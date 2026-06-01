# LSTM/XGB 여주보 수위 예측 서버 (김민준)

## 필수 모델 파일
`services/lstm_xgb/models/` 폴더에 아래 파일 복사:
```
total_scaler.pkl
target_scaler.pkl
yeoju_lstm_model.keras
yeoju_xgb_model.pkl
yeoju_lstm_model_1h.keras
yeoju_xgb_model_1h.pkl
yeoju_lstm_model_3h.keras
yeoju_xgb_model_3h.pkl
```

## 환경변수 설정
프로젝트 루트 `.env`에 추가:
```
HRFCO_SERVICE_KEY=한강홍수통제소_키
```

## 초기 히스토리 CSV 생성 (최초 1회)
```bash
docker exec yeoju_lstm_xgb python -m app.hrfco_collector --init
```

## API 엔드포인트
- `POST /predict`          → 김민준 원본 형식 `{"status": "success", "predictions": {"10m":x,"1h":y,"3h":z}}`
- `POST /api/v1/predict`   → 신가연 표준 형식 (predictor 어댑터 연결용)
- `GET  /health`

## 신가연 API 매핑
| 김민준 | 신가연 표준 | 의미 |
|--------|------------|------|
| 10m    | h1_pred_m  | 10분 후 |
| 1h     | h6_pred_m  | 1시간 후 |
| 3h     | h18_pred_m | 3시간 후 |
| (없음) | h36_pred_m | 6시간 후 → 3h 근사값 사용 |
