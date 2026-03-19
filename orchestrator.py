import os
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

# ── Tool 정의 (Phase 2와 동일) ────────────────────────────────────────────

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
                    "company": {"type": "string", "description": "기업명 (예: 삼성전자, Apple)"},
                    "days":    {"type": "integer", "description": "수집 기간 (기본 7일)", "default": 7},
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
                "한국어는 LLM, 영어는 distilroberta 모델로 분석합니다."
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
                                "link":        {"type": "string"},
                            },
                        },
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
                    "company": {"type": "string", "description": "기업명 (예: 삼성전자, Apple)"},
                    "market":  {"type": "string", "enum": ["domestic", "foreign"], "default": "domestic"},
                    "days":    {"type": "integer", "description": "조회 기간 (기본 30일)", "default": 30},
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

def _execute_tool(name: str, arguments: dict) -> dict:
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


async def _execute_tool_async(name: str, arguments: dict) -> dict:
    """비동기 Tool 실행 — 동기 함수를 스레드풀에서 실행"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _execute_tool, name, arguments)


# ── 병렬 실행 판단 ────────────────────────────────────────────────────────
# 서로 의존하지 않는 Tool은 동시에 실행합니다.
# 의존 관계:
#   collect_news     ─┐
#   get_stock_data   ─┘ → 독립 (병렬 가능)
#   analyze_sentiment   → collect_news 결과 필요
#   analyze_correlation → sentiment + stock 결과 모두 필요

PARALLEL_SAFE = {"collect_news", "get_stock_data"}  # 독립 실행 가능한 Tool


async def _execute_tool_calls_async(
    tool_calls: list,
    verbose: bool,
) -> list[dict]:
    """
    tool_calls 목록을 받아 병렬/순차 실행 결정 후 처리.
    PARALLEL_SAFE에 해당하는 Tool이 2개 이상이면 병렬 실행.
    """
    parallel = [tc for tc in tool_calls if tc.function.name in PARALLEL_SAFE]
    sequential = [tc for tc in tool_calls if tc.function.name not in PARALLEL_SAFE]

    results = {}

    # 병렬 실행
    if len(parallel) > 1:
        if verbose:
            names = [tc.function.name for tc in parallel]
            print(f"\n[병렬 실행] {names}")

        tasks = {
            tc.id: _execute_tool_async(
                tc.function.name,
                json.loads(tc.function.arguments),
            )
            for tc in parallel
        }
        done = await asyncio.gather(*tasks.values())
        for tc, result in zip(parallel, done):
            results[tc.id] = result
            if verbose:
                preview = json.dumps(result, ensure_ascii=False)[:150] + "..."
                print(f"  └ {tc.function.name} 완료: {preview}")

    elif len(parallel) == 1:
        tc = parallel[0]
        args = json.loads(tc.function.arguments)
        if verbose:
            print(f"\n[Step] Tool 호출: {tc.function.name}")
            print(f"       인자: {json.dumps(args, ensure_ascii=False)}")
        result = _execute_tool(tc.function.name, args)
        results[tc.id] = result
        if verbose:
            preview = json.dumps(result, ensure_ascii=False)[:150] + "..."
            print(f"       결과: {preview}")

    # 순차 실행
    for tc in sequential:
        args = json.loads(tc.function.arguments)
        if verbose:
            print(f"\n[순차 실행] Tool 호출: {tc.function.name}")
            print(f"            인자: {json.dumps(args, ensure_ascii=False)}")
        try:
            result = _execute_tool(tc.function.name, args)
            results[tc.id] = result
            if verbose:
                preview = json.dumps(result, ensure_ascii=False)[:150] + "..."
                print(f"            결과: {preview}")
        except Exception as e:
            results[tc.id] = {"error": str(e)}
            if verbose:
                print(f"            오류: {e}")

    return results


# ── HIL (Human-in-the-Loop) ───────────────────────────────────────────────

def _check_confidence(result: dict) -> tuple[bool, str]:
    """
    분석 결과의 신뢰도를 확인합니다.
    낮은 신뢰도 기준:
      - 상관관계 데이터 부족 (matched_days < 3)
      - Pearson r 절댓값이 0.2 미만 (상관관계 없음)
      - 뉴스 기사 수가 5개 미만
    반환: (신뢰도_낮음: bool, 사유: str)
    """
    # 뉴스 수 확인
    if "total" in result and result["total"] < 5:
        return True, f"수집된 뉴스가 {result['total']}개로 부족합니다 (최소 5개 권장)"

    # 상관관계 확인
    if "correlation" in result:
        corr = result["correlation"]
        if corr.get("pearson_r") is None:
            return True, corr.get("note", "상관관계 데이터 부족")
        if abs(corr["pearson_r"]) < 0.2:
            return True, f"뉴스 감성과 주가의 상관관계가 매우 낮습니다 (r={corr['pearson_r']})"

    return False, ""


def _ask_human(question: str) -> str:
    """
    사용자에게 확인을 요청합니다.
    Streamlit 환경에서는 UI 컴포넌트로 교체됩니다.
    """
    print(f"\n[HIL] {question}")
    print("      계속하려면 Enter, 중단하려면 'n' 입력: ", end="")
    answer = input().strip().lower()
    return answer


# ── 메인 오케스트레이터 ───────────────────────────────────────────────────

async def _run_async(
    user_query: str,
    verbose: bool = True,
    hil: bool = True,
) -> str:
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
                "독립적인 작업(뉴스 수집, 주가 조회)은 가능하면 함께 요청하세요. "
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

        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "stop" or not msg.tool_calls:
            if verbose:
                print(f"\n[Step {step}] 최종 답변 생성")
                print("-" * 60)
            return msg.content

        messages.append(msg)

        # 병렬/순차 실행
        results = await _execute_tool_calls_async(msg.tool_calls, verbose)

        # HIL 체크 — 신뢰도 낮은 결과 감지
        if hil:
            for result in results.values():
                low_confidence, reason = _check_confidence(result)
                if low_confidence:
                    answer = _ask_human(
                        f"신뢰도가 낮습니다: {reason}\n"
                        "분석을 계속 진행할까요?"
                    )
                    if answer == "n":
                        return f"분석을 중단했습니다. 사유: {reason}"

        # Tool 결과를 messages에 추가
        for tool_call in msg.tool_calls:
            result = results.get(tool_call.id, {"error": "실행 실패"})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })


def run(
    user_query: str,
    verbose: bool = True,
    hil: bool = True,
) -> str:
    """
    사용자 질의를 받아 멀티 에이전트 루프를 실행합니다.

    Args:
        user_query: 사용자 질의 (자연어)
        verbose: Tool 호출 과정 출력 여부
        hil: Human-in-the-Loop 활성화 여부
    """
    return asyncio.run(_run_async(user_query, verbose, hil))
