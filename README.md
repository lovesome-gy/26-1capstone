# 실시간 데이터 기반 지능형 수위 예측 및 AI 의사결정 지원 시스템

**삼육대학교 컴퓨터공학부 캡스톤디자인2 | 물결탐사대 | 2026**

---

## 팀 구성 및 역할

| 이름 | 역할 | 담당 서비스 |
|------|------|-------------|
| 신가연 (팀장) | LLM 통합 & 자연어 보고서 | `llm_service` (Part④) |
| 정휘수 | 데이터 수집 | `data_collector` (Part①) |
| 문성빈 | ML 모델 학습 | `model_trainer` (Part②) |
| 김민준 | 예측 서빙 | `predictor` (Part③) |

---

## 디렉토리 구조

```
yeoju-water-ai/
├── docker-compose.yml          # 7 컨테이너 오케스트레이션
├── .env.example                # 환경변수 템플릿
├── services/
│   ├── db/
│   │   └── init.sql            # PostgreSQL 초기화 (TimescaleDB)
│   ├── llm_service/            # ← Part④ 신가연
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── app/
│   │       ├── main.py         # FastAPI 진입점
│   │       ├── core/
│   │       │   ├── config.py   # 환경변수 (pydantic-settings)
│   │       │   └── database.py # SQLAlchemy async 엔진
│   │       ├── models/
│   │       │   ├── report.py   # MakeReportTB ORM
│   │       │   └── decision.py # DecisionSupportTB ORM
│   │       ├── schemas/
│   │       │   ├── report.py   # Pydantic 요청/응답
│   │       │   └── decision.py
│   │       ├── prompts/
│   │       │   ├── report_prompt.py    # 보고서 생성 프롬프트
│   │       │   └── decision_prompt.py  # 의사결정 지원 프롬프트
│   │       ├── services/
│   │       │   ├── ollama_client.py    # Ollama HTTP 클라이언트
│   │       │   ├── report_generator.py # 보고서 생성 비즈니스 로직
│   │       │   └── decision_support.py # 의사결정 생성 비즈니스 로직
│   │       ├── api/
│   │       │   ├── report.py   # POST /api/v1/reports/generate
│   │       │   └── decision.py # POST /api/v1/decisions/generate
│   │       └── utils/
│   │           └── formatter.py
│   ├── data_collector/         # Part① 정휘수
│   ├── model_trainer/          # Part② 문성빈
│   ├── predictor/              # Part③ 김민준
│   └── frontend/               # Streamlit 대시보드
└── data/
    └── merged/                 # 2년치 수위 데이터 (gitignore)
```

---

## 빠른 시작

```bash
# 1. 환경변수 설정
cp .env.example .env
# .env 파일에서 POSTGRES_PASSWORD 등 실제 값 입력

# 2. 전체 서비스 실행
docker-compose up -d

# 3. Ollama 모델 pull (최초 1회, 자동으로도 실행됨)
docker exec yeoju_ollama ollama pull qwen3:8b

# 4. API 문서 확인
# http://localhost:8002/docs  ← LLM Service
# http://localhost:8001/docs  ← Predictor
# http://localhost:8501       ← Streamlit 대시보드
```

---

## API 엔드포인트 (LLM Service - Part④)

### 보고서 생성
```
POST /api/v1/reports/generate
GET  /api/v1/reports/
GET  /api/v1/reports/{report_id}
```

### 의사결정 지원
```
POST  /api/v1/decisions/generate
GET   /api/v1/decisions/
PATCH /api/v1/decisions/acknowledge
```

### 헬스체크
```
GET /health
```

---

## DB 주요 테이블

| 테이블 | 설명 | 담당 |
|--------|------|------|
| `water_level_raw` | 10분 단위 원시 수위 관측값 (TimescaleDB) | Part① |
| `model_registry` | 학습 모델 메타데이터 (RMSE, MAE, R²) | Part② |
| `prediction_result` | 실시간 수위 예측 결과 | Part③ |
| `make_report_tb` | LLM 자연어 보고서 | **Part④** |
| `decision_support_tb` | LLM 의사결정 지원 항목 | **Part④** |

---

## 데이터 출처

- **한강홍수통제소(HRFC)**: 여주보 10분 수위
- **K-water**: 수문 운영 정보 (10분/시간/일)
- **병합 파일**: `data/merged/kwater_hrfc_10min_2024-01-01_2025-12-31_preprocessed.csv`
