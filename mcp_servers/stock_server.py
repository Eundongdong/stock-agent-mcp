import os
import warnings
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()
warnings.filterwarnings("ignore")

mcp = FastMCP("stock-analyzer")

_krx_tickers: dict | None = None


def _load_krx_tickers() -> dict:
    global _krx_tickers
    if _krx_tickers is not None:
        return _krx_tickers
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("KRX")
        _krx_tickers = dict(zip(df["Name"].str.strip(), df["Code"].str.strip()))
    except Exception:
        _krx_tickers = {}
    return _krx_tickers


def _get_ticker_with_llm(company: str) -> str:
    from llm_provider import get_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm("default")
    response = llm.invoke([
        SystemMessage(content=(
            "You are a stock market expert. "
            "When given a company name, respond ONLY with its ticker symbol. "
            "For Korean stocks, return the 6-digit code only (e.g. 005930). "
            "For US stocks, return the ticker symbol only (e.g. AAPL). "
            "If unknown, respond with UNKNOWN."
        )),
        HumanMessage(content=f"What is the ticker symbol for {company}?"),
    ])
    ticker = response.content.strip().upper()
    return ticker if ticker != "UNKNOWN" else company


def _get_ticker(company: str, market: str) -> str:
    if market == "domestic":
        tickers = _load_krx_tickers()

        # 1. 완전 일치
        if company in tickers:
            return tickers[company] + ".KS"

        # 2. 부분 일치
        matches = [
            (name, code)
            for name, code in tickers.items()
            if company in name
        ]
        if matches:
            best = min(matches, key=lambda x: len(x[0]))
            return best[1] + ".KS"

        # 3. LLM 폴백
        code = _get_ticker_with_llm(company)
        if code.isdigit():
            return code + ".KS"
        return code

    # 국외: yfinance 직접 처리
    return company


def _fetch_stock_data(ticker: str, days: int) -> list[dict]:
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=days + 10)

    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
    except Exception:
        return []

    if df.empty:
        return []

    df = df.tail(days)
    df.index = df.index.strftime("%Y-%m-%d")

    records = []
    prev_close = None
    for date, row in df.iterrows():
        def get_val(col):
            v = row[col]
            return float(v.iloc[0] if hasattr(v, "iloc") else v)

        close  = get_val("Close")
        high   = get_val("High")
        low    = get_val("Low")
        volume = get_val("Volume")

        change_pct = None
        if prev_close and prev_close != 0:
            change_pct = round((close - prev_close) / prev_close * 100, 4)

        records.append({
            "date": date,
            "close": round(close, 4),
            "high": round(high, 4),
            "low": round(low, 4),
            "volume": int(volume),
            "change_pct": change_pct,
            "volatility": round(high - low, 4),
        })
        prev_close = close

    return records


def _calc_correlation(stock_data: list[dict], sentiment_by_date: dict) -> dict:
    label_to_score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    pairs = []
    for record in stock_data:
        date = record["date"]
        if date in sentiment_by_date and record["change_pct"] is not None:
            dominant = sentiment_by_date[date].get("dominant", "neutral")
            pairs.append((label_to_score.get(dominant, 0.0), record["change_pct"]))

    if len(pairs) < 3:
        return {
            "pearson_r": None,
            "matched_days": len(pairs),
            "note": "상관관계 계산에 필요한 데이터 부족 (최소 3일)",
        }

    import statistics
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    mean_x, mean_y = statistics.mean(x), statistics.mean(y)
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = statistics.stdev(x) if len(x) > 1 else 0
    std_y = statistics.stdev(y) if len(y) > 1 else 0

    pearson_r = 0.0 if (std_x == 0 or std_y == 0) else \
        round(cov / ((len(pairs) - 1) * std_x * std_y), 4)

    abs_r = abs(pearson_r)
    strength  = "강한" if abs_r >= 0.7 else "중간" if abs_r >= 0.4 else \
                "약한" if abs_r >= 0.2 else "없음"
    direction = "양의" if pearson_r > 0 else "음의"

    return {
        "pearson_r": pearson_r,
        "matched_days": len(pairs),
        "interpretation": f"{direction} {strength} 상관관계",
        "note": "감성점수(positive=1, neutral=0, negative=-1)와 당일 주가 등락률",
    }


@mcp.tool()
def get_stock_data(
    company: str,
    market: str = "domestic",
    days: int = 30,
) -> dict:
    """
    기업의 주가 데이터를 조회합니다.
    종목코드를 몰라도 기업명으로 자동 조회합니다.

    Args:
        company: 기업명 (예: 삼성전자, Apple)
        market: 시장 구분 (domestic=국내, foreign=국외)
        days: 조회 기간 (기본 30일)

    Returns:
        ticker: 사용된 티커 심볼
        prices: 일별 주가 데이터
        summary: 기간 요약
    """
    ticker = _get_ticker(company, market)
    prices = _fetch_stock_data(ticker, days)

    if not prices:
        return {
            "ticker": ticker,
            "prices": [],
            "summary": {},
            "error": f"\'{ticker}\' 주가 데이터를 가져올 수 없습니다.",
        }

    closes  = [p["close"] for p in prices]
    changes = [p["change_pct"] for p in prices if p["change_pct"] is not None]

    return {
        "ticker": ticker,
        "prices": prices,
        "summary": {
            "period": f"{prices[0]['date']} ~ {prices[-1]['date']}",
            "trading_days": len(prices),
            "close_latest": closes[-1],
            "close_high": round(max(closes), 4),
            "close_low": round(min(closes), 4),
            "close_avg": round(sum(closes) / len(closes), 4),
            "change_avg_pct": round(sum(changes) / len(changes), 4) if changes else None,
        },
    }


@mcp.tool()
def analyze_correlation(stock_data: dict, sentiment_results: dict) -> dict:
    """
    주가 변동률과 감성 분석 결과의 상관관계를 분석합니다.
    get_stock_data와 analyze_sentiment Tool의 출력을 그대로 받습니다.

    Args:
        stock_data: get_stock_data Tool의 출력
        sentiment_results: analyze_sentiment Tool의 출력

    Returns:
        correlation: Pearson 상관계수 및 해석
        matched_dates: 분석에 사용된 날짜 목록
        insight: 상관관계 요약 설명
    """
    prices  = stock_data.get("prices", [])
    by_date = sentiment_results.get("by_date", {})

    if not prices or not by_date:
        return {
            "correlation": {"pearson_r": None, "interpretation": "데이터 없음"},
            "matched_dates": [],
            "insight": "주가 또는 감성 데이터가 없습니다.",
        }

    correlation   = _calc_correlation(prices, by_date)
    matched_dates = sorted({p["date"] for p in prices} & set(by_date.keys()))

    r = correlation.get("pearson_r")
    if r is None:
        insight = "데이터가 부족하여 상관관계를 계산할 수 없습니다."
    elif r > 0.4:
        insight = f"뉴스 감성이 긍정적일수록 주가가 오르는 경향이 있습니다. (r={r})"
    elif r < -0.4:
        insight = f"뉴스 감성이 긍정적일수록 주가가 내리는 역의 관계가 있습니다. (r={r})"
    else:
        insight = f"뉴스 감성과 주가 변동 간 뚜렷한 상관관계가 없습니다. (r={r})"

    return {
        "correlation": correlation,
        "matched_dates": matched_dates,
        "insight": insight,
    }


if __name__ == "__main__":
    mcp.run()
