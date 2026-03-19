# 📈 AI 주식 분석 에이전트

> **MCP(Model Context Protocol) + AI 오케스트레이션 기반 주식 뉴스 감성 분석 시스템**
>
> 자연어 한 마디로 뉴스 수집 → 감성 분석 → 주가 조회 → 상관관계 분석까지 자동으로 수행
> 
> (기존 프로젝트에서 업그레이드하여 프로젝트 진행)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastMCP](https://img.shields.io/badge/FastMCP-MCP%20Server-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-blue)

---

## 📌 프로젝트 소개

이 프로젝트는 **LangChain 기반 순차 파이프라인**으로 구축했던 기존 주식 예측 프로그램을 **MCP + AI 오케스트레이션** 구조로 업그레이드한 실습 프로젝트입니다.

### 핵심 특징

- 🤖 **자율 에이전트** — "이 회사에 투자해도 될까?"라고 물으면 에이전트가 스스로 필요한 도구를 선택하고 순서를 결정
- 🔌 **MCP 표준 인터페이스** — 각 기능이 독립적인 MCP 서버로 분리되어 언어/환경에 무관하게 호출 가능
- ⚡ **병렬 실행** — 서로 의존하지 않는 작업(뉴스 수집 + 주가 조회)을 동시에 실행
- 🔀 **멀티 LLM 라우팅** — task별로 다른 LLM provider 지정 가능 (Azure / OpenAI / Anthropic)
- 🇰🇷 **한/영 감성 분석** — 한국어는 LLM, 영어는 distilroberta 모델로 자동 분기
- 🏢 **KRX 전체 종목** — 종목코드 없이 기업명만으로 2,878개 국내 종목 자동 조회
- 🛡️ **Human-in-the-Loop** — 신뢰도 낮은 분석 결과 감지 시 사용자 확인 요청

---

## 🏗️ 아키텍처
<img width="1440" height="974" alt="image" src="https://github.com/user-attachments/assets/63f87978-cc57-4d1b-b83c-db85d5622385" />



---

## 🆚 기존 프로젝트 대비 개선점

| 항목 | 기존 (LangChain) | 현재 (MCP + 오케스트레이션) |
|---|---|---|
| 실행 방식 | 하드코딩 순차 실행 | 에이전트 자율 판단 |
| 도구 인터페이스 | Python import 직접 호출 | MCP JSON-RPC 표준 |
| 환경 분리 | Google Colab 수동 실행 | MCP 서버 자동 호출 |
| 병렬 처리 | 없음 | asyncio.gather 병렬화 |
| 한국어 감성 분석 | 오분류 (neutral) | LLM 분기로 정확도 개선 |
| 종목 조회 | 하드코딩 10개 | KRX 2,878개 + LLM 폴백 |
| LLM 교체 | 전체 코드 수정 | `.env` 한 줄 변경 |
| 사용자 인터페이스 | 정적 결과 출력 | 자연어 + 실시간 스트리밍 |

---

## 📁 프로젝트 구조
```
stock-agent-mcp/
├── .env                        # API 키 및 LLM Provider 설정 (gitignore)
├── .python-version             # Python 3.12.9 고정
├── llm_provider.py             # task별 LLM 라우팅 (Azure / OpenAI / Anthropic)
├── orchestrator.py             # 오케스트레이터 에이전트 (tool_use 루프 + 병렬화 + HIL)
├── main.py                     # Streamlit UI 진입점
├── requirements.txt
├── mcp_servers/
│   ├── news_server.py          # 뉴스 수집 MCP 서버 (Naver API + NewsAPI)
│   ├── sentiment_server.py     # 감성 분석 MCP 서버 (한/영 자동 분기)
│   └── stock_server.py         # 주가 조회 + 상관관계 분석 MCP 서버
├── docs/
│   └── architecture.md
└── tests/
```

---

## ⚙️ 설치 및 실행

### 사전 요구사항

- Python 3.12 이상
- pyenv (버전 관리)
- API 키: Azure OpenAI, Naver API, NewsAPI

### 1. 레포지토리 클론
```bash
git clone https://github.com/Eundongdong/stock-agent-mcp.git
cd stock-agent-mcp
```

### 2. Python 환경 설정
```bash
pyenv install 3.12.9
pyenv local 3.12.9
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

### 3. 의존성 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 환경변수 설정

`.env` 파일을 생성하고 아래 내용을 채웁니다.
```bash
# LLM Provider 설정
LLM_PROVIDER=azure

# task별 provider 지정 (선택, 미설정 시 LLM_PROVIDER 기본값 사용)
# LLM_PROVIDER_SUMMARIZE=azure
# LLM_PROVIDER_SENTIMENT=azure
# LLM_PROVIDER_PREDICT=anthropic

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
AZURE_OPENAI_API_VERSION=version_name

# 뉴스 수집 API
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_client_secret
NEWS_API_KEY=your_newsapi_key
```

### 5. 실행
```bash
streamlit run main.py
```

브라우저에서 `http://localhost:8501` 접속

---

## 💬 사용 예시

브라우저 입력창에 자연어로 질의합니다.
```
삼성전자 최근 뉴스 감성 분석해줘
SK하이닉스 투자해도 될까?
Apple 최근 뉴스 어때?
삼성전자 주가와 뉴스 상관관계를 7일치로 분석해줘
```

에이전트가 스스로 필요한 도구를 선택하고 실행합니다.
```
[병렬 실행] collect_news + get_stock_data  ← 동시 실행
  └ collect_news 완료
  └ get_stock_data 완료
[순차 실행] analyze_sentiment
[순차 실행] analyze_correlation
```

---

## 🔧 MCP 서버 개별 테스트

각 서버는 독립적으로 실행하고 테스트할 수 있습니다.
```bash
# MCP Inspector로 JSON-RPC 통신 직접 확인
npx @modelcontextprotocol/inspector python mcp_servers/news_server.py
npx @modelcontextprotocol/inspector python mcp_servers/sentiment_server.py
npx @modelcontextprotocol/inspector python mcp_servers/stock_server.py
```

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|---|---|
| LLM | Azure OpenAI (GPT), 확장 가능: OpenAI / Anthropic Claude |
| MCP 서버 | FastMCP |
| 감성 분석 | mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis |
| 주가 데이터 | yfinance, FinanceDataReader |
| 뉴스 수집 | Naver News API, NewsAPI |
| UI | Streamlit |
| 비동기 처리 | asyncio |
| 언어 | Python 3.12 |

---

## 🗺️ LLM Provider 전환

`.env`에서 한 줄만 바꾸면 코드 수정 없이 LLM을 교체할 수 있습니다.
```bash
# Azure OpenAI 사용 (현재)
LLM_PROVIDER=azure

# Claude로 전환
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key

# task별로 다른 LLM 지정
LLM_PROVIDER_SUMMARIZE=azure       # 뉴스 요약 → Azure (비용 효율)
LLM_PROVIDER_PREDICT=anthropic     # 주가 예측 판단 → Claude (추론 능력)
```

---

## 📋 향후 개선 예정 (리팩토링 과제)

- [ ] **날짜 불일치 해결** — 뉴스 수집 기간을 어제 기준으로 조정, lag 분석 추가
- [ ] **상관관계 고도화** — Granger 인과성 검정, rolling 상관계수
- [ ] **예측 모델 에이전트화** — XGBoost 모델을 MCP Tool로 래핑, 신뢰도 구간 포함
- [ ] **MCP 프로세스 완전 분리** — 현재 함수 직접 호출 → 실제 MCP 프로세스 JSON-RPC 통신
- [ ] **국외 종목 티커 개선** — yfinance 검색 실패 시 LLM 폴백 고도화

---

## 📚 관련 자료

- [FastMCP 공식 문서](https://github.com/jlowin/fastmcp)
- [MCP 공식 스펙](https://modelcontextprotocol.io)
- [기존 실습 프로젝트 (LangChain 버전)](https://github.com/Eundongdong/stock_prediction_public)
