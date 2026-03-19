import os
import re
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("news-collector")

# ── 내부 유틸 ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """HTML 태그, 이메일, URL, 특수문자 제거"""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # HTML 태그
    text = re.sub(r"&[a-zA-Z]+;", " ", text)       # HTML 엔티티
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # URL
    text = re.sub(r"\S+@\S+", " ", text)           # 이메일
    text = re.sub(r"[^\w\s]", " ", text)           # 특수문자
    text = re.sub(r"\s+", " ", text).strip()       # 중복 공백
    return text


def _fetch_naver(company: str, days: int) -> list[dict]:
    """Naver 뉴스 API로 기사 수집"""
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        return []

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }

    cutoff = datetime.now() - timedelta(days=days)
    articles = []

    for page in range(1, 11):  # 최대 10페이지
        params = {
            "query": company,
            "display": 100,
            "start": (page - 1) * 100 + 1,
            "sort": "date",
        }
        try:
            res = requests.get(
                "https://openapi.naver.com/v1/search/news.json",
                headers=headers,
                params=params,
                timeout=10,
            )
            res.raise_for_status()
            items = res.json().get("items", [])
        except Exception:
            break

        for item in items:
            # 기업명이 제목에 포함된 경우만
            title = _clean_text(item.get("title", ""))
            if company not in title:
                continue

            pub_date_str = item.get("pubDate", "")
            try:
                pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")
                pub_date = pub_date.replace(tzinfo=None)
            except Exception:
                continue

            if pub_date < cutoff:
                # 날짜 오래된 것 나오면 중단
                return articles

            articles.append({
                "source": "naver",
                "title": title,
                "description": _clean_text(item.get("description", "")),
                "link": item.get("link", ""),
                "pub_date": pub_date.strftime("%Y-%m-%d"),
                "content": "",
            })

            if len(articles) >= days * 30:
                return articles

    return articles


def _fetch_newsapi(company: str, days: int) -> list[dict]:
    """NewsAPI로 기사 수집"""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return []

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "q": company,
        "from": cutoff,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": min(days * 30, 100),
        "apiKey": api_key,
    }

    try:
        res = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=10,
        )
        res.raise_for_status()
        raw_articles = res.json().get("articles", [])
    except Exception:
        return []

    articles = []
    for item in raw_articles:
        title = _clean_text(item.get("title", ""))
        if company.lower() not in title.lower():
            continue

        pub_date_str = item.get("publishedAt", "")
        try:
            pub_date = datetime.strptime(pub_date_str, "%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            continue

        articles.append({
            "source": "newsapi",
            "title": title,
            "description": _clean_text(item.get("description", "")),
            "link": item.get("url", ""),
            "pub_date": pub_date.strftime("%Y-%m-%d"),
            "content": _clean_text(item.get("content", "")),
        })

    return articles


# ── MCP Tool ──────────────────────────────────────────────────────────────

@mcp.tool()
def collect_news(company: str, days: int = 7) -> dict:
    """
    기업명으로 최근 뉴스를 수집합니다.
    Naver API(국내)와 NewsAPI(국외)를 모두 조회하여 통합 반환합니다.

    Args:
        company: 검색할 기업명 (예: 삼성전자, Apple)
        days: 수집 기간 (기본 7일, 최대 30일)

    Returns:
        articles: 수집된 기사 목록
        total: 총 기사 수
        by_date: 날짜별 기사 수
        sources: 출처별 기사 수
    """
    days = min(days, 30)  # 최대 30일 제한

    naver_articles = _fetch_naver(company, days)
    newsapi_articles = _fetch_newsapi(company, days)
    all_articles = naver_articles + newsapi_articles

    # 날짜별 집계
    by_date: dict[str, int] = {}
    for a in all_articles:
        by_date[a["pub_date"]] = by_date.get(a["pub_date"], 0) + 1

    # 출처별 집계
    sources: dict[str, int] = {}
    for a in all_articles:
        sources[a["source"]] = sources.get(a["source"], 0) + 1

    return {
        "articles": all_articles,
        "total": len(all_articles),
        "by_date": dict(sorted(by_date.items())),
        "sources": sources,
    }


if __name__ == "__main__":
    mcp.run()
