import os
import json
import asyncio
from dotenv import load_dotenv
from llm_provider import get_llm

load_dotenv()

# ── MCP Tool 정의 ─────────────────────────────────────────────────────────
# 오케스트레이터는 MCP 서버 내부를 모릅니다.
# Tool 이름, 설명, 입력 스키마만 알고 있습니다.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "collect_news",
            "description": (
                "기업명으로 최근 뉴스를 수집합니다. "
                "Naver API(국내)와 NewsAPI(국외)를 통합 조회합니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": "검색할 기업명 (예: 삼성전자, Apple)",
                    },
                    "days": {
                        "type": "integer",
                        "description": "수집 기간 (기본 7일, 최대 30일)",
                        "default": 7,
                    },
                },
                "required": ["company"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sentiment",
            "description": (
                "뉴스 기사 목록의 감성을 분석합니다. "
                "한국어는 LLM, 영어는 distilroberta 모델로 분석합니다. "
                "collect_news의 articles 결과를 그대로 입력으로 사용합니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "articles": {
                        "type": "array",
                        "description": "collect_news Tool이 반환한 articles 배열",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title":       {"type": "string"},
                                "description": {"type": "string"},
                                "content":     {"type": "string"},
                                "pub_date":    {"type": "string"},
                                "source":      {"type": "string"},
                                "link":        {"type": "string"}
                            }
                        }
                    },
                },
                "required": ["articles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_data",
            "description": (
                "기업의 주가 데이터를 조회합니다. "
                "종목코드 없이 기업명만으로 자동 조회합니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": "기업명 (예: 삼성전자, Apple)",
                    },
                    "market": {
                        "type": "string",
                        "enum": ["domestic", "foreign"],
                        "description": "시장 구분 (domestic=국내, foreign=국외)",
                        "default": "domestic",
                    },
                    "days": {
                        "type": "integer",
                        "description": "조회 기간 (기본 30일)",
                        "default": 30,
                    },
                },
                "required": ["company"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_correlation",
            "description": (
                "주가 변동률과 감성 분석 결과의 상관관계를 분석합니다. "
                "get_stock_data와 analyze_sentiment의 결과를 입력으로 사용합니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_data": {
                        "type": "object",
                        "description": "get_stock_data Tool의 반환값 전체",
                    },
                    "sentiment_results": {
                        "type": "object",
                        "description": "analyze_sentiment Tool의 반환값 전체",
                    },
                },
                "required": ["stock_data", "sentiment_results"],
            },
        },
    },
]


# ── Tool 실행 레이어 ──────────────────────────────────────────────────────
# 에이전트가 호출을 결정하면 실제 MCP 서버 함수를 실행합니다.

def _execute_tool(name: str, arguments: dict) -> dict:
    """Tool 이름과 인자를 받아 실제 MCP 서버 함수 실행"""

    if name == "collect_news":
        from mcp_servers.news_server import collect_news
        return collect_news(**arguments)

    if name == "analyze_sentiment":
        from mcp_servers.sentiment_server import analyze_sentiment
        return analyze_sentiment(**arguments)

    if name == "get_stock_data":
        from mcp_servers.stock_server import get_stock_data
        return get_stock_data(**arguments)

    if name == "analyze_correlation":
        from mcp_servers.stock_server import analyze_correlation
        return analyze_correlation(**arguments)

    raise ValueError(f"알 수 없는 Tool: {name}")


# ── 오케스트레이터 ────────────────────────────────────────────────────────

def run(user_query: str, verbose: bool = True) -> str:
    """
    사용자 질의를 받아 에이전트 루프를 실행합니다.

    에이전트 루프:
    1. LLM이 질의를 보고 어떤 Tool을 호출할지 결정
    2. Tool 실행 후 결과를 메시지에 추가
    3. LLM이 결과를 보고 다음 Tool 또는 최종 답변 결정
    4. stop_reason == "stop" 이 될 때까지 반복
    """
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    messages = [
        {
            "role": "system",
            "content": (
                "당신은 금융 분석 전문가 AI입니다. "
                "주어진 도구를 활용하여 기업의 뉴스와 주가를 분석하고 "
                "투자 판단에 도움이 되는 인사이트를 제공합니다. "
                "분석 시 반드시 수집된 데이터를 근거로 답변하세요."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    if verbose:
        print(f"\n[오케스트레이터] 질의: {user_query}")
        print("=" * 60)

    step = 0

    while True:
        step += 1

        # LLM 호출 — 다음 행동 결정
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Tool 호출이 없으면 최종 답변
        if finish_reason == "stop" or not msg.tool_calls:
            if verbose:
                print(f"\n[Step {step}] 최종 답변 생성")
                print("-" * 60)
            return msg.content

        # Tool 호출 처리
        messages.append(msg)

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"\n[Step {step}] Tool 호출: {tool_name}")
                print(f"           인자: {json.dumps(tool_args, ensure_ascii=False)}")

            # 실제 Tool 실행
            try:
                result = _execute_tool(tool_name, tool_args)
                result_str = json.dumps(result, ensure_ascii=False)

                if verbose:
                    # 결과 요약 출력 (전체 출력 시 너무 길어서 일부만)
                    preview = result_str[:200] + "..." if len(result_str) > 200 else result_str
                    print(f"           결과: {preview}")

            except Exception as e:
                result_str = json.dumps({"error": str(e)})
                if verbose:
                    print(f"           오류: {e}")

            # Tool 결과를 메시지에 추가
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str,
            })
