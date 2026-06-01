# 수문 예측 프로젝트 (평가용 API 서버 패키지)

Hydro-MAST 모델을 **API 서버 형태**로 실행해 성능평가/재현이 가능하도록 정리한 패키지입니다.

## 1) 폴더 구성

- `01_data_pipeline`: 전처리/데이터 유틸
- `02_model_development`: 모델/검증 코드
- `03_realtime_pipeline`: API 서버 및 실시간 추론 파이프라인
- `04_artifacts`:
  - `models/`: `hydro_mast_v2.pt`, 스케일러
  - `data/`: `features_v2_train.csv`, `features_v2_test.csv`, 최신 추론 JSON
  - `docs/`: 성능 지표 및 전처리 스펙

## 2) 로컬 실행 (권장)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-lock.txt
run_api_server.bat
```

브라우저/평가 도구:
- 대시보드: `http://127.0.0.1:8787/`
- API health: `GET http://127.0.0.1:8787/api/health`
- API 예측: `POST http://127.0.0.1:8787/api/predict` (`{"skip_api": true}`)

### 예측 API 호출 예시

```powershell
curl -X POST "http://127.0.0.1:8787/api/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"skip_api\": true}"
```

## 3) Docker 실행

```powershell
docker compose up --build
```

실행 후 동일하게 `http://127.0.0.1:8787`에서 접근 가능합니다.

## 4) 평가용 최소 검증

1. `GET /api/health` 응답에서 필수 파일 존재 여부 확인
2. `POST /api/predict` 호출 후 `predictions_m`와 `meta` 확인
3. 필요 시 `skip_api=false`로 실측 API 연동 검증

자동 스모크 테스트:

```powershell
.\smoke_test_api.ps1
```

## 5) 환경 변수

- `.env`는 로컬 전용이며 저장소 커밋 금지
- `.env.example`를 복사해 `.env` 생성 후 사용
- API 키가 없으면 `skip_api=true`로 구조 검증 가능
