import os
import warnings
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()
warnings.filterwarnings("ignore")

mcp = FastMCP("sentiment-analyzer")

# 모델은 첫 호출 시 한 번만 로드 (전역 캐싱)
_pipeline = None


def _get_pipeline():
    """감성 분석 파이프라인 로드 (최초 1회만 실행)"""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "text-classification",
            model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            top_k=None,
        )
    return _pipeline


def _has_korean(text: str) -> bool:
    """한글 포함 여부 확인 (유니코드 한글 음절 범위)"""
    return any('\uac00' <= c <= '\ud7a3' for c in text)


def _split_text(text: str, chunk_size: int = 512) -> list[str]:
    """
    토크나이저 최대 길이(512) 초과 시 텍스트 분할.
    단어 기준으로 분할하여 문장 중간에서 잘리지 않도록 함.
    """
    words = text.split()
    chunks, current = [], []
    current_len = 0

    for word in words:
        estimated_tokens = len(word) // 4 + 1
        if current_len + estimated_tokens > chunk_size:
            if current:
                chunks.append(" ".join(current))
            current, current_len = [word], estimated_tokens
        else:
            current.append(word)
            current_len += estimated_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


def _analyze_with_model(text: str) -> dict:
    """영어 텍스트 — distilroberta 모델로 분석"""
    pipe = _get_pipeline()
    chunks = _split_text(text)

    label_scores: dict[str, list[float]] = {
        "positive": [], "negative": [], "neutral": []
    }

    for chunk in chunks:
        results = pipe(chunk)[0]
        for item in results:
            label = item["label"].lower()
            if label in label_scores:
                label_scores[label].append(item["score"])

    avg_scores = {
        label: sum(scores) / len(scores)
        for label, scores in label_scores.items()
        if scores
    }

    best_label = max(avg_scores, key=avg_scores.get)

    return {
        "label": best_label,
        "score": round(avg_scores[best_label], 4),
        "confidence": round(avg_scores[best_label], 4),
        "all_scores": {k: round(v, 4) for k, v in avg_scores.items()},
        "method": "distilroberta",
    }


def _analyze_with_llm(text: str) -> dict:
    """
    한국어 텍스트 — LLM으로 감성 분석.
    LLM_PROVIDER_SENTIMENT 또는 LLM_PROVIDER 기본값 사용.
    """
    from llm_provider import get_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm("sentiment")

    system_prompt = (
        "You are a financial news sentiment analyzer. "
        "Analyze the sentiment of the given news text and respond ONLY with a JSON object. "
        "Format: "
        '{"label": "positive" or "negative" or "neutral", '
        '"score": float between 0.0 and 1.0, '
        '"reason": "one sentence explanation in Korean"}'
    )

    user_prompt = f"다음 금융 뉴스의 감성을 분석해줘:\n\n{text[:1000]}"

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        import json, re
        # 응답에서 JSON 부분만 추출
        raw = response.content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            raise ValueError("JSON 파싱 실패")

        parsed = json.loads(match.group())
        label = parsed.get("label", "neutral").lower()
        score = float(parsed.get("score", 0.5))
        reason = parsed.get("reason", "")

        return {
            "label": label,
            "score": round(score, 4),
            "confidence": round(score, 4),
            "all_scores": {},
            "reason": reason,
            "method": "llm",
        }

    except Exception as e:
        # LLM 호출 실패 시 neutral로 fallback
        return {
            "label": "neutral",
            "score": 0.5,
            "confidence": 0.0,
            "all_scores": {},
            "reason": f"분석 실패: {str(e)}",
            "method": "llm_fallback",
        }


def _analyze_single(text: str) -> dict:
    """
    텍스트 하나에 대한 감성 분석.
    한국어 포함 시 LLM으로 분기, 영어는 distilroberta 모델 사용.
    """
    if not text or not text.strip():
        return {
            "label": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "method": "empty",
        }

    if _has_korean(text):
        return _analyze_with_llm(text)
    else:
        return _analyze_with_model(text)


# ── MCP Tool ──────────────────────────────────────────────────────────────

@mcp.tool()
def analyze_sentiment(articles: list[dict]) -> dict:
    """
    뉴스 기사 목록에 대한 감성 분석 수행.
    collect_news Tool의 출력을 그대로 입력으로 받습니다.
    한국어 기사는 LLM, 영어 기사는 distilroberta 모델로 분석합니다.

    Args:
        articles: 기사 목록 (title, description, content, pub_date 포함)

    Returns:
        results: 기사별 감성 분석 결과
        overall: 전체 감성 요약 (label, score)
        by_date: 날짜별 감성 분포
        summary: 긍정/부정/중립 기사 수
    """
    if not articles:
        return {
            "results": [],
            "overall": {"label": "neutral", "score": 0.0},
            "by_date": {},
            "summary": {"positive": 0, "negative": 0, "neutral": 0},
        }

    results = []
    summary = {"positive": 0, "negative": 0, "neutral": 0}
    by_date: dict[str, list[str]] = {}

    for article in articles:
        # content 우선, 없으면 description, 없으면 title 사용
        text = (
            article.get("content")
            or article.get("description")
            or article.get("title")
            or ""
        )

        analysis = _analyze_single(text)
        label = analysis["label"]
        summary[label] = summary.get(label, 0) + 1

        pub_date = article.get("pub_date", "unknown")
        if pub_date not in by_date:
            by_date[pub_date] = []
        by_date[pub_date].append(label)

        results.append({
            "title": article.get("title", ""),
            "pub_date": pub_date,
            "sentiment": analysis,
        })

    # 날짜별 감성 분포 집계
    by_date_summary = {}
    for date, labels in sorted(by_date.items()):
        by_date_summary[date] = {
            "positive": labels.count("positive"),
            "negative": labels.count("negative"),
            "neutral": labels.count("neutral"),
            "dominant": max(set(labels), key=labels.count),
        }

    # 전체 감성: 최빈 레이블 + 평균 점수
    all_scores = [r["sentiment"]["score"] for r in results]
    all_labels = [r["sentiment"]["label"] for r in results]
    overall_label = max(set(all_labels), key=all_labels.count)
    overall_score = round(sum(all_scores) / len(all_scores), 4)

    return {
        "results": results,
        "overall": {"label": overall_label, "score": overall_score},
        "by_date": by_date_summary,
        "summary": summary,
    }


if __name__ == "__main__":
    mcp.run()
