import streamlit as st
import json
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI 주식 분석 에이전트",
    page_icon="📈",
    layout="wide",
)

# ── 스타일 ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.tool-call {
    background-color: #f0f2f6;
    border-left: 4px solid #4CAF50;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 0 4px 4px 0;
    font-size: 0.85em;
    font-family: monospace;
}
.tool-parallel {
    border-left-color: #2196F3;
}
.tool-error {
    border-left-color: #f44336;
}
.hil-box {
    background-color: #fff3cd;
    border: 1px solid #ffc107;
    padding: 12px;
    border-radius: 4px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ── 세션 상태 초기화 ──────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "hil_pending" not in st.session_state:
    st.session_state.hil_pending = None  # HIL 대기 중인 질문

if "agent_log" not in st.session_state:
    st.session_state.agent_log = []


# ── 사이드바 ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ 설정")

    st.markdown("**LLM Provider**")
    provider = os.getenv("LLM_PROVIDER", "azure")
    st.info(f"현재: `{provider}`")
    st.caption(".env의 LLM_PROVIDER 값으로 변경")

    st.divider()

    st.markdown("**분석 옵션**")
    hil_enabled = st.toggle("Human-in-the-Loop", value=True)
    verbose_enabled = st.toggle("에이전트 로그 표시", value=True)

    st.divider()

    st.markdown("**빠른 질의 예시**")
    examples = [
        "삼성전자 최근 뉴스 감성 분석해줘",
        "삼성전자 주가와 뉴스 상관관계 7일치 분석해줘",
        "Apple 최근 뉴스 어때?",
        "SK하이닉스 투자해도 될까?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.quick_query = ex

    st.divider()
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_log = []
        st.session_state.hil_pending = None
        st.rerun()


# ── 메인 레이아웃 ─────────────────────────────────────────────────────────

st.title("📈 AI 주식 분석 에이전트")
st.caption("뉴스 감성 분석 · 주가 조회 · 상관관계 분석을 자연어로")

# 채팅 영역과 로그 영역 분리
col_chat, col_log = st.columns([3, 2])

with col_chat:
    st.markdown("### 💬 대화")

    # 대화 히스토리 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

with col_log:
    st.markdown("### 🔍 에이전트 실행 로그")
    log_container = st.container()
    # rerun 후에도 이전 로그 유지
    if st.session_state.get("log_items"):
        for item_text, item_style in st.session_state.log_items:
            css_class = f"tool-call {item_style}"
            log_container.markdown(
                f'<div class="{css_class}">{item_text}</div>',
                unsafe_allow_html=True,
            )


# ── 에이전트 실행 (스트리밍 버전) ─────────────────────────────────────────

def run_agent_streaming(user_query: str, log_placeholder):
    """
    오케스트레이터를 실행하면서 각 단계를 실시간으로 UI에 표시.
    Phase 3의 verbose 출력을 Streamlit 컴포넌트로 교체.
    """
    from openai import AzureOpenAI
    from orchestrator import TOOLS, _execute_tool

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

    log_items = []

    def add_log(text: str, style: str = ""):
        log_items.append((text, style))
        st.session_state.log_items = log_items
        # 로그 실시간 업데이트
        with log_placeholder:
            for item_text, item_style in log_items:
                css_class = f"tool-call {item_style}"
                st.markdown(
                    f'<div class="{css_class}">{item_text}</div>',
                    unsafe_allow_html=True,
                )

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
            add_log(f"✅ Step {step}: 최종 답변 생성 완료")
            return msg.content

        messages.append(msg)

        # 병렬 여부 판단
        PARALLEL_SAFE = {"collect_news", "get_stock_data"}
        parallel = [tc for tc in msg.tool_calls if tc.function.name in PARALLEL_SAFE]
        sequential = [tc for tc in msg.tool_calls if tc.function.name not in PARALLEL_SAFE]

        # 병렬 실행 로그
        if len(parallel) > 1:
            names = [tc.function.name for tc in parallel]
            add_log(f"⚡ 병렬 실행: {' + '.join(names)}", "tool-parallel")

            results = {}
            for tc in parallel:
                args = json.loads(tc.function.arguments)
                try:
                    result = _execute_tool(tc.function.name, args)
                    results[tc.id] = result
                    add_log(f"  └ {tc.function.name} 완료", "tool-parallel")
                except Exception as e:
                    results[tc.id] = {"error": str(e)}
                    add_log(f"  └ {tc.function.name} 오류: {e}", "tool-error")

        elif len(parallel) == 1:
            tc = parallel[0]
            args = json.loads(tc.function.arguments)
            add_log(f"🔧 Step {step}: {tc.function.name}({json.dumps(args, ensure_ascii=False)})")
            try:
                results = {tc.id: _execute_tool(tc.function.name, args)}
                add_log(f"  └ 완료")
            except Exception as e:
                results = {tc.id: {"error": str(e)}}
                add_log(f"  └ 오류: {e}", "tool-error")
        else:
            results = {}

        # 순차 실행
        for tc in sequential:
            args = json.loads(tc.function.arguments)
            add_log(f"🔧 Step {step}: {tc.function.name}({json.dumps(args, ensure_ascii=False)[:60]}...)")
            try:
                result = _execute_tool(tc.function.name, args)
                results[tc.id] = result

                # HIL 체크
                if hil_enabled:
                    from orchestrator import _check_confidence
                    low_conf, reason = _check_confidence(result)
                    if low_conf:
                        st.session_state.hil_pending = reason
                        add_log(f"  ⚠️ 신뢰도 낮음: {reason}", "tool-error")

                add_log(f"  └ 완료")
            except Exception as e:
                results[tc.id] = {"error": str(e)}
                add_log(f"  └ 오류: {e}", "tool-error")

        # Tool 결과 messages에 추가
        for tc in msg.tool_calls:
            result = results.get(tc.id, {"error": "실행 실패"})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })


# ── 입력 처리 ─────────────────────────────────────────────────────────────

# 빠른 질의 버튼 처리
query_input = None
if "quick_query" in st.session_state:
    query_input = st.session_state.pop("quick_query")

# 채팅 입력
chat_input = st.chat_input("기업명과 분석 내용을 입력하세요 (예: 삼성전자 최근 뉴스 어때?)")
if chat_input:
    query_input = chat_input

# 질의 실행
if query_input:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": query_input})
    st.session_state.agent_log = []
    st.session_state.hil_pending = None

    with col_chat:
        with st.chat_message("user"):
            st.markdown(query_input)

        with st.chat_message("assistant"):
            with st.spinner("분석 중..."):
                if "log_items" not in st.session_state:
                    st.session_state.log_items = []
                st.session_state.log_items = []
                log_placeholder = col_log.container()

                # HIL 처리
                if st.session_state.hil_pending:
                    with col_chat:
                        st.markdown(
                            f'<div class="hil-box">⚠️ <b>신뢰도 확인 필요</b><br>{st.session_state.hil_pending}</div>',
                            unsafe_allow_html=True,
                        )
                        c1, c2 = st.columns(2)
                        if c1.button("계속 진행"):
                            st.session_state.hil_pending = None
                        if c2.button("분석 중단"):
                            st.stop()

                answer = run_agent_streaming(query_input, log_placeholder)
                st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
