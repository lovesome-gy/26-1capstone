# 💧 실시간 데이터 기반 지능형 수위 예측 및 AI 의사결정 지원 시스템

**컴퓨터공학부 | 산학협력 캡스톤디자인2 | 물결탐사대 | 2026-1**

> 경기도 여주시 여주보(남한강)를 테스트베드로, 실시간 수위 데이터를 수집·예측하고
> LLM 기반 자연어 보고서와 AI 의사결정 지원을 통해 수해 대응의 골든타임을 확보하는 시스템

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 과제 번호 | P08 |
| 지도교수 | 김성완 교수 (삼육대학교 컴퓨터공학부) |
| 산업체 멘토 | 장일식 (시그마케이주식회사) |
| 테스트베드 | 경기도 여주시 여주보 (관측소 코드: 3008680) |
| 성능 목표 | 수위 예측 정확도 90% 이상, 시스템 응답 5초 이내 |

---

## 👩‍💻 팀 구성 및 역할

| 이름 | 담당 |
|---|---|
| **신가연** (팀장) | 시스템 전체 설계 및 구현 총괄 *(아래 상세 참고)* |
| 정휘수 | Hydro-MAST 딥러닝 예측 모델 개발 (Graph-GRU + Advective Delay) |
| 문성빈 | Streamlit 프론트엔드 UI/UX 및 Plotly 시각화 컴포넌트 초안 |
| 김민준 | LSTM/XGBoost 앙상블 예측 모델 학습 및 서버 초안 |

### 신가연 담당 상세

**시스템 설계 및 인프라**
- 5-레이어 마이크로서비스 아키텍처 설계
- Docker Compose 기반 8개 컨테이너 통합 오케스트레이션
- PostgreSQL + TimescaleDB 스키마 설계 (5개 테이블)
- 전체 환경변수 체계 및 네트워크 구성

**백엔드 개발 (Part④ LLM 서비스)**
- FastAPI LLM 서비스 전체 구현 (`services/llm_service/`)
- Ollama/Qwen3-8b 온프레미스 LLM 연동 클라이언트
- 보고서 생성 모듈 (시간별/일간/주간/월간/긴급 5종)
- 의사결정 지원 모듈 (경보 단계별 수문제어/대피/모니터링)
- 보고서·의사결정 프롬프트 설계 및 XML 파싱
- SQLAlchemy async ORM 모델 (`make_report_tb`, `decision_support_tb`)
- Pydantic 스키마 및 REST API 엔드포인트

**팀원 코드 통합 (신가연 직접 수행)**
- 정휘수 Hydro-MAST → 신가연 표준 API 어댑터 작성 (`services/predictor/`)
- 김민준 LSTM/XGB 서버 → 신가연 표준 API 명세 맞춤 수정 (`services/lstm_xgb/app/main.py`)
- 두 모델 통합 어댑터: `?model=hydro_mast|lstm_xgb|both` 파라미터 구현
- HRFCO 실시간 데이터 수집 모듈 작성 (`services/lstm_xgb/app/hrfco_collector.py`)

**데이터 파이프라인**
- 실시간 수위 데이터 수집 서비스 구현 (`services/data_collector/`)
- 한강홍수통제소(HRFCO) + K-water API 연동
- 2년치 전처리 CSV → PostgreSQL 적재 스크립트 (`scripts/load_data.py`)

**프론트엔드 통합**
- 문성빈 UI 컴포넌트 + 신가연 LLM API 통합 대시보드 완성
- 5개 탭 구성: 실시간 현황 / 모델 비교 / 보고서 생성 / 의사결정 / 이력
- 멘토 피드백 반영: 모델 좌우 분할 비교, 월간 통계 입력, 의사결정 독립 분리

**프로젝트 문서화 (전체)**
- 과제신청서 작성
- 중간보고서 작성
- 기업실습멘토보고서 작성 (1~6회차 전체)
- 결과보고서 작성
- 팀원 통합 가이드라인 및 API 명세서 작성
- 개인 WBS(작업분류체계) 및 간트차트 작성

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network (yeoju_net)                │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  TimescaleDB │   │    Ollama    │   │ data_collector │  │
│  │  (postgres)  │   │  Qwen3-8b   │   │   (신가연)     │  │
│  │  :5432       │   │  :11434     │   │                │  │
│  └──────┬───────┘   └──────┬───────┘   └────────────────┘  │
│         │                  │                                 │
│  ┌──────▼──────────────────▼──────────────────────────┐    │
│  │           llm_service (신가연 Part④)                │    │
│  │  POST /api/v1/reports/generate                      │    │
│  │  POST /api/v1/decisions/generate    :8002           │    │
│  └─────────────────────┬───────────────────────────────┘    │
│                         │                                    │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │           predictor (신가연, 통합 어댑터)            │   │
│  │  ?model=hydro_mast | lstm_xgb | both   :8001        │   │
│  └────────────┬─────────────────────────┬──────────────┘   │
│               │                         │                    │
│  ┌────────────▼────────┐   ┌────────────▼────────┐         │
│  │  hydro_mast (정휘수) │   │  lstm_xgb (김민준)  │         │
│  │  :8787              │   │  :8000              │         │
│  └─────────────────────┘   └─────────────────────┘         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      frontend — Streamlit (문성빈 UI + 신가연 통합)   │  │
│  │      :8501                                           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|---|---|
| Language | Python 3.11 |
| Backend | FastAPI, SQLAlchemy (async), asyncpg |
| Frontend | Streamlit, Plotly |
| Database | PostgreSQL 15 + TimescaleDB |
| AI/ML | PyTorch (Graph-GRU, LSTM), XGBoost, Scikit-learn |
| LLM | Ollama / Qwen3-8b (온프레미스) |
| Infrastructure | Docker, docker-compose |
| Data Sources | 한강홍수통제소(HRFCO), K-water 공공데이터포털 |
| Server | NVIDIA RTX 5090 (시그마케이 제공) |

---

## 📁 디렉토리 구조

```
26-1capstone/
├── docker-compose.yml              # 8개 컨테이너 오케스트레이션 (신가연)
├── .env.example                    # 환경변수 템플릿 (신가연)
├── scripts/
│   └── load_data.py                # CSV → PostgreSQL 적재 (신가연)
├── services/
│   ├── db/
│   │   └── init.sql                # TimescaleDB 초기화, 5개 테이블 DDL (신가연)
│   ├── llm_service/                # 신가연 Part④ — LLM 보고서/의사결정
│   │   └── app/
│   │       ├── main.py             # FastAPI 진입점
│   │       ├── core/               # config.py, database.py
│   │       ├── models/             # MakeReportTB, DecisionSupportTB ORM
│   │       ├── schemas/            # Pydantic 요청/응답
│   │       ├── prompts/            # 보고서/의사결정 프롬프트
│   │       ├── services/           # ollama_client, report_generator, decision_support
│   │       └── api/                # /reports, /decisions 라우터
│   ├── predictor/                  # 신가연 — 두 모델 통합 어댑터
│   │   └── app/main.py
│   ├── hydro_mast/                 # 정휘수 — Hydro-MAST 예측 서버
│   ├── lstm_xgb/                   # 김민준 모델 + 신가연 어댑터/수집기
│   │   └── app/
│   │       ├── main.py             # 신가연 API 명세 맞춤 수정
│   │       └── hrfco_collector.py  # HRFCO 실시간 수집 (신가연)
│   ├── data_collector/             # 신가연 — 실시간 수위 수집 서비스
│   │   └── app/main.py
│   └── frontend/                   # 문성빈 UI + 신가연 LLM API 통합
│       └── app/dashboard.py
└── docs/
    └── team_integration_guide.md   # 팀원 통합 가이드라인 (신가연)
```

---

## 🚀 빠른 시작

### 1. 환경변수 설정

```bash
cp .env.example .env
# .env에서 아래 값 입력
# POSTGRES_PASSWORD, DATABASE_URL
# HRFCO_SERVICE_KEY, DATA_GO_KR_SERVICE_KEY
```

### 2. Hydro-MAST 코드 배치 (정휘수)

```bash
cd services/hydro_mast
git clone https://github.com/hwisu-jung/flood-forecast-project .
# services/hydro_mast/.env에 API 키 설정
```

### 3. LSTM/XGB 모델 파일 배치 (김민준)

```bash
# services/lstm_xgb/models/ 에 복사:
# total_scaler.pkl, target_scaler.pkl
# yeoju_lstm_model.keras / _1h.keras / _3h.keras
# yeoju_xgb_model.pkl / _1h.pkl / _3h.pkl
```

### 4. 전체 서비스 실행

```bash
docker compose up postgres ollama llm_service -d
docker compose up hydro_mast -d
docker compose up predictor frontend data_collector lstm_xgb -d --build
```

### 5. 초기 데이터 적재

```bash
# HRFCO 초기 히스토리 CSV 생성
docker exec -e HRFCO_SERVICE_KEY=키값 yeoju_lstm_xgb python -m app.hrfco_collector --init

# 2년치 과거 데이터 적재
python scripts/load_data.py --db-url "postgresql://yeoju_admin:비밀번호@localhost:5432/yeoju_water"
```

### 6. 접속 URL

| 서비스 | URL |
|---|---|
| 통합 대시보드 | http://localhost:8501 |
| LLM API 문서 | http://localhost:8002/docs |
| Predictor 상태 | http://localhost:8001/health |
| Hydro-MAST | http://localhost:8787 |
| LSTM/XGB | http://localhost:8000/health |

---

## 📊 대시보드 기능

| 탭 | 기능 |
|---|---|
| 📊 실시간 현황 | Hydro-MAST 4지평 예측 + 경보 단계 배너 + Plotly 바 차트 |
| 🤖 모델 비교 | Hydro-MAST vs LSTM/XGB 좌우 분할 비교 (멘토 피드백 반영) |
| 📄 AI 보고서 | 시간별/일간/주간/월간 LLM 자연어 보고서 자동 생성 |
| 🧭 의사결정 지원 | 예측 수위 기반 수문 제어/대피/모니터링 조치 안내 |
| 📋 보고서 이력 | 생성된 보고서 조회 및 확인 처리 |

---

## 🔌 API 명세

### LLM Service (포트 8002)

```
POST /api/v1/reports/generate    보고서 생성
GET  /api/v1/reports/            보고서 목록
GET  /api/v1/reports/{id}        보고서 단건 조회
POST /api/v1/decisions/generate  의사결정 생성
GET  /api/v1/decisions/          의사결정 목록
PATCH /api/v1/decisions/acknowledge  확인 처리
GET  /health
```

### Predictor 어댑터 (포트 8001)

```
POST /api/v1/predict?model=hydro_mast   Hydro-MAST 단독
POST /api/v1/predict?model=lstm_xgb    LSTM/XGB 단독
POST /api/v1/predict?model=both        두 모델 동시 비교
GET  /health
```

---

## 🗄️ DB 테이블

| 테이블 | 설명 | 담당 |
|---|---|---|
| `water_level_raw` | HRFCO/K-water 10분 수위 (TimescaleDB) | 신가연 |
| `model_registry` | 학습 모델 메타데이터 (RMSE, NSE) | 신가연 |
| `prediction_result` | 실시간 예측 결과 | 신가연 |
| `make_report_tb` | LLM 자연어 보고서 | 신가연 |
| `decision_support_tb` | LLM 의사결정 지원 | 신가연 |

---

## 📈 모델 성능

| 모델 | 담당 | h1 NSE | h6 NSE | h18 NSE | h36 NSE |
|---|---|---|---|---|---|
| Hydro-MAST | 정휘수 | 0.939 | 0.924 | 0.902 | 0.868 |
| LSTM/XGB | 김민준 | 측정 중 | 측정 중 | 측정 중 | 측정 중 |

---

## 🔑 환경변수

| 변수 | 설명 |
|---|---|
| `POSTGRES_PASSWORD` | PostgreSQL 비밀번호 |
| `DATABASE_URL` | SQLAlchemy async 연결 URL |
| `OLLAMA_MODEL` | LLM 모델명 (기본: qwen3:8b) |
| `OLLAMA_TIMEOUT` | LLM 응답 대기 시간(초, 기본: 300) |
| `HRFCO_SERVICE_KEY` | 한강홍수통제소 API 인증키 |
| `DATA_GO_KR_SERVICE_KEY` | 공공데이터포털 K-water 인증키 |
| `SKIP_API` | true=캐시 사용(로컬), false=실시간 API |
| `DEFAULT_MODEL` | 기본 예측 모델 (hydro_mast/lstm_xgb/both) |

---

## 📋 프로젝트 문서

아래 문서는 모두 신가연이 작성하였습니다.

| 문서 | 내용 |
|---|---|
| 과제신청서 | 프로젝트 개요, 개발 목표, 기대효과 기술 |
| 중간보고서 | 5-레이어 아키텍처, 기술 스택, 일정, 진행 현황 |
| 기업실습멘토보고서 (1~6회차) | 멘토링 내용 및 피드백 적용 사항 기록 |
| 결과보고서 | 시스템 전체 구현 결과, 성능 평가, 향후 과제 |
| 개인 WBS / 간트차트 | 7개 작업 그룹, 공휴일 반영 NETWORKDAYS 기반 일정 |
| API 명세서 | LLM Service / Predictor 엔드포인트 전체 정의 |
| 팀원 통합 가이드 | 팀원 코드 배치 및 통합 절차 문서 |

---

## 📝 데이터 출처

- **한강홍수통제소(HRFCO)**: 여주보 10분 수위/유량 (관측소 1007639)
- **K-water 공공데이터포털**: 댐 수위/방류량 (관측소 1007602)
- **학습 데이터**: 2024-01-01 ~ 2025-12-31 (105,264건, 10분 간격)

---

## 📄 라이선스

본 프로젝트는 삼육대학교 캡스톤디자인 과제물로 작성되었으며, 학술적 목적으로만 사용됩니다.
