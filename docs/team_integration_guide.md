# 팀원 코드 통합 가이드라인
**작성: 신가연 (팀장) | 2026-05-28**

이 문서는 각 팀원이 자신의 코드를 통합 시스템에 연결하기 위해 **반드시 지켜야 할 명세**입니다.
내 llm_service와 docker-compose가 기준이며, 이 명세에 맞게 각자 코드를 수정해주세요.

---

## 정휘수 (flood-forecast-project / Hydro-MAST)

### 해야 할 것: 딱 1가지

네 `flood-forecast-project` 레포 **전체 내용**을 아래 경로에 복사해줘:

```
26-1 capstone/services/hydro_mast/
```

복사 후 구조:
```
services/hydro_mast/
├── Dockerfile              ← 네 레포에 있는 그대로
├── requirements.txt        (또는 requirements-lock.txt)
├── 01_data_pipeline/
├── 02_model_development/
├── 03_realtime_pipeline/
├── 04_artifacts/
│   ├── models/hydro_mast_v2.pt    ← 필수
│   ├── models/feature_scaler_v2.pkl
│   ├── models/target_scaler_v2.pkl
│   └── data/
└── ...
```

### 확인해야 할 것

네 서버가 **포트 8787**에서 실행되고 아래 두 엔드포인트가 동작하는지 확인:

**1. 헬스체크**
```
GET http://localhost:8787/api/health
```
응답 예시:
```json
{"status": "ok", "model_loaded": true, ...}
```

**2. 예측 API**
```
POST http://localhost:8787/api/predict
Content-Type: application/json

{"skip_api": true}
```
응답 **반드시 포함해야 할 필드:**
```json
{
  "predictions_m": {
    "h1": 5.23,
    "h6": 5.45,
    "h18": 5.87,
    "h36": 6.10
  },
  "meta": {
    "bucket_t": "2025-07-15T14:00:00+09:00"
  }
}
```

`predictions_m`의 키가 `h1/h6/h18/h36`이 아니거나 이름이 다르면 알려줘. 내가 어댑터 코드 수정할게.

### 수정 필요 없음

네 서버 코드 자체는 **수정 안 해도 돼**. 내 어댑터(predictor)가 네 API를 감싸서 변환함.

---

## 김민준 (AI_LSTM_XGB)

### 역할 확인

네 레포(`AI_LSTM_XGB`)의 `Server/` 폴더가 어떤 역할인지 알려줘:

1. **예측 API 서버**라면 → 정휘수처럼 docker-compose에 추가 가능
2. **모델 학습 코드**라면 → `services/model_trainer/` 에 넣으면 됨
3. **둘 다**라면 → 각각 경로 분리

### Server 폴더가 API 서버인 경우

아래 명세에 맞는 엔드포인트가 있는지 확인하고, 없으면 추가해줘:

**필수 엔드포인트 1: 헬스체크**
```
GET http://localhost:{네포트}/health
```
응답:
```json
{"status": "ok", "service": "lstm_xgb"}
```

**필수 엔드포인트 2: 예측**
```
POST http://localhost:{네포트}/api/v1/predict
```
요청 본문:
```json
{
  "station_id": "3008680",
  "horizon_minutes": 60
}
```
응답 **반드시 포함해야 할 필드:**
```json
{
  "prediction_id": 12345,
  "station_id": "3008680",
  "predicted_level": 5.87,
  "confidence_low": 5.80,
  "confidence_high": 5.94,
  "horizon_minutes": 60,
  "target_time": "2025-07-15T15:00:00+09:00",
  "model_name": "LSTM"
}
```

### AI_model_create 폴더가 학습 코드인 경우

아래 경로에 복사:
```
26-1 capstone/services/model_trainer/
```

학습 완료된 모델 파일(`.pt` 또는 `.pkl`)은:
```
26-1 capstone/services/hydro_mast/04_artifacts/models/
```
에 같이 넣어줘 (정휘수 artifacts 폴더 공유).

### 모델 비교 결과 DB 저장 (선택사항)

LSTM/XGB/Hydro-MAST 비교를 위해 아래 테이블에 학습 결과를 저장하면 좋아:

```sql
-- 이미 DB에 있는 테이블
INSERT INTO model_registry (model_name, version, rmse, mae, r2_score, accuracy_pct, artifact_path)
VALUES ('LSTM', 'v1.0', 0.025, 0.018, 0.91, 91.0, '/artifacts/lstm_v1.pt');
```

Python으로 삽입하려면:
```python
import psycopg2
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
# INSERT 쿼리 실행
```

---

## 공통 규칙

### 환경변수

각자 서비스에서 필요한 환경변수는 **루트의 `.env` 파일**에 추가해줘.
`.env.example`도 같이 업데이트.

### Docker 네트워크

모든 컨테이너는 `yeoju_net` 브리지 네트워크 안에서 실행됨.
서비스끼리 통신할 때는 `localhost` 대신 **컨테이너 이름**으로 호출:

| 서비스 | 내부 주소 | 외부 포트 |
|--------|-----------|----------|
| postgres | `postgres:5432` | 5432 |
| ollama | `ollama:11434` | 11434 |
| hydro_mast | `hydro_mast:8787` | 8787 |
| predictor | `predictor:8001` | 8001 |
| llm_service | `llm_service:8002` | 8002 |
| frontend | `frontend:8501` | 8501 |

### 브랜치 규칙

각자 브랜치에서 작업하고 `dev`로 PR:
```
feat/data      ← 정휘수
feat/model     ← 김민준
feat/llm       ← 신가연
```

---

## 통합 테스트 순서 (전원 합류 후)

```bash
# 1. 전체 서비스 실행
docker compose up -d

# 2. 각 서비스 헬스체크
curl http://localhost:8787/api/health   # Hydro-MAST
curl http://localhost:8001/health       # Predictor 어댑터
curl http://localhost:8002/health       # LLM Service
curl http://localhost:8501              # Frontend

# 3. 예측 → 보고서 생성 파이프라인 테스트
python scripts/validate_pipeline.py
```

---

## 문의

명세가 안 맞거나 수정이 필요하면 바로 알려줘. 내가 어댑터 코드 수정할게.
