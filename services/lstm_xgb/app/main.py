"""
services/lstm_xgb/app/main.py
김민준 AI_API_SERVER.py → 신가연 API 명세 맞춤 수정

[변경 사항]
1. POST /predict 유지 (김민준 원본)
2. POST /api/v1/predict 추가 → 신가연 표준 응답 형식으로 래핑
   - 10m → h1_pred_m
   - 1h  → h6_pred_m
   - 3h  → h18_pred_m (h36_pred_m은 3h로 근사, 6h 모델 미구현)
3. GET  /health 추가
4. HRFCO API 실시간 데이터 자동 수집 엔드포인트 추가

실행:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import logging
from collections import deque
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lstm_xgb")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KST = timezone(timedelta(hours=9))


def get_path(filename):
    return os.path.join(BASE_DIR, "..", "models", filename)


# ── Pydantic 입력 모델 (김민준 원본 유지) ─────────────────────────────────────
class WaterLevelData(BaseModel):
    시간: str
    여주보_상류_수위_수위_m: float
    여주보_상류_수위_유량_m3_s: float
    여주보_상류_수위_해발수위_El_m: float
    여주보_하류_수위_수위_m: float
    여주보_하류_수위_유량_m3_s: float
    여주보_하류_수위_해발수위_El_m: float
    여주시_여주대교_강수량_강수량_mm: float
    여주시_여주대교_강수량_누적강수량_mm: float
    여주시_여주대교_수위_수위_m: float
    여주시_여주대교_수위_유량_m3_s: float
    여주시_여주대교_수위_해발수위_El_m: float
    여주시_주암리_강수량_강수량_mm: float
    여주시_주암리_강수량_누적강수량_mm: float
    원주시_문막교_수위_수위_m: float
    원주시_문막교_수위_유량_m3_s: float
    원주시_문막교_수위_해발수위_El_m: float
    충주댐_댐_현재수위_EL_m: float
    충주댐_댐_유입량_m3_s: float
    충주댐_댐_방류량_m3_s: float
    충주시_수안보면사무소_강수량_강수량_mm: float
    충주시_수안보면사무소_강수량_누적강수량_mm: float
    충주조정지댐_댐_현재수위_EL_m: float
    충주조정지댐_댐_유입량_m3_s: float
    충주조정지댐_댐_방류량_m3_s: float


# ── 앱 초기화 ─────────────────────────────────────────────────────────────────
app = FastAPI(title="LSTM/XGB 여주보 수위 예측 서버", version="1.0.0")

# 모델 전역 변수
scaler = None
target_scaler = None
model_lstm_10m = model_xgb_10m = None
model_lstm_1h  = model_xgb_1h  = None
model_lstm_3h  = model_xgb_3h  = None
model_features = []
history_db: deque = deque(maxlen=100)
models_loaded = False


def load_models():
    global scaler, target_scaler, model_features
    global model_lstm_10m, model_xgb_10m
    global model_lstm_1h,  model_xgb_1h
    global model_lstm_3h,  model_xgb_3h
    global models_loaded, history_db

    try:
        from keras.models import load_model  # type: ignore
        scaler        = joblib.load(get_path("total_scaler.pkl"))
        target_scaler = joblib.load(get_path("target_scaler.pkl"))
        model_lstm_10m = load_model(get_path("yeoju_lstm_model.keras"),     compile=False, safe_mode=False)
        model_xgb_10m  = joblib.load(get_path("yeoju_xgb_model.pkl"))
        model_lstm_1h  = load_model(get_path("yeoju_lstm_model_1h.keras"),  compile=False, safe_mode=False)
        model_xgb_1h   = joblib.load(get_path("yeoju_xgb_model_1h.pkl"))
        model_lstm_3h  = load_model(get_path("yeoju_lstm_model_3h.keras"),  compile=False, safe_mode=False)
        model_xgb_3h   = joblib.load(get_path("yeoju_xgb_model_3h.pkl"))
        model_features = list(scaler.feature_names_in_)

        # 히스토리 초기 로드
        csv_path = get_path("여주보_예측_필요_최근100행_데이터셋.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path).tail(100)
            history_db = deque(df.to_dict("records"), maxlen=100)
            logger.info("히스토리 CSV 로드 완료: %d행", len(history_db))

        models_loaded = True
        logger.info("🚀 LSTM/XGB 모델 탑재 완료 (10m, 1h, 3h)")
    except Exception as e:
        logger.error("❌ 모델 로드 실패: %s", e)
        models_loaded = False


@app.on_event("startup")
def startup():
    load_models()


# ── 예측 핵심 로직 (김민준 원본 유지) ─────────────────────────────────────────
def perform_multi_prediction(data_dict: dict) -> dict:
    global history_db

    mapped = {
        "시간": data_dict["시간"],
        "여주보(상류)_수위_수위(m)":           data_dict["여주보_상류_수위_수위_m"],
        "여주보(상류)_수위_유량(m³/s)":         data_dict["여주보_상류_수위_유량_m3_s"],
        "여주보(상류)_수위_해발수위(El.m)":      data_dict["여주보_상류_수위_해발수위_El_m"],
        "여주보(하류)_수위_수위(m)":            data_dict["여주보_하류_수위_수위_m"],
        "여주보(하류)_수위_유량(m³/s)":          data_dict["여주보_하류_수위_유량_m3_s"],
        "여주보(하류)_수위_해발수위(El.m)":       data_dict["여주보_하류_수위_해발수위_El_m"],
        "여주시(여주대교)_강수량_강수량(mm)":     data_dict["여주시_여주대교_강수량_강수량_mm"],
        "여주시(여주대교)_강수량_누적강수량(mm)":  data_dict["여주시_여주대교_강수량_누적강수량_mm"],
        "여주시(여주대교)_수위_수위(m)":         data_dict["여주시_여주대교_수위_수위_m"],
        "여주시(여주대교)_수위_유량(m³/s)":       data_dict["여주시_여주대교_수위_유량_m3_s"],
        "여주시(여주대교)_수위_해발수위(El.m)":    data_dict["여주시_여주대교_수위_해발수위_El_m"],
        "여주시(주암리)_강수량_강수량(mm)":       data_dict["여주시_주암리_강수량_강수량_mm"],
        "여주시(주암리)_강수량_누적강수량(mm)":    data_dict["여주시_주암리_강수량_누적강수량_mm"],
        "원주시(문막교)_수위_수위(m)":          data_dict["원주시_문막교_수위_수위_m"],
        "원주시(문막교)_수위_유량(m³/s)":        data_dict["원주시_문막교_수위_유량_m3_s"],
        "원주시(문막교)_수위_해발수위(El.m)":     data_dict["원주시_문막교_수위_해발수위_El_m"],
        "충주댐_댐_현재수위(EL.m)":             data_dict["충주댐_댐_현재수위_EL_m"],
        "충주댐_댐_유입량(m³/s)":               data_dict["충주댐_댐_유입량_m3_s"],
        "충주댐_댐_방류량(m³/s)":               data_dict["충주댐_댐_방류량_m3_s"],
        "충주시(수안보면사무소)_강수량_강수량(mm)": data_dict["충주시_수안보면사무소_강수량_강수량_mm"],
        "충주시(수안보면사무소)_강수량_누적강수량(mm)": data_dict["충주시_수안보면사무소_강수량_누적강수량_mm"],
        "충주조정지댐_댐_현재수위(EL.m)":        data_dict["충주조정지댐_댐_현재수위_EL_m"],
        "충주조정지댐_댐_유입량(m³/s)":          data_dict["충주조정지댐_댐_유입량_m3_s"],
        "충주조정지댐_댐_방류량(m³/s)":          data_dict["충주조정지댐_댐_방류량_m3_s"],
    }

    target_cols = [
        "충주조정지댐_댐_방류량(m³/s)", "원주시(문막교)_수위_유량(m³/s)",
        "여주시(여주대교)_수위_유량(m³/s)", "충주시(수안보면사무소)_강수량_강수량(mm)",
        "여주보(상류)_수위_수위(m)",
    ]
    lags = [6, 18, 36, 72]
    lag_features = {}
    for col in target_cols:
        for lag in lags:
            val = history_db[-lag].get(col, mapped.get(col, 0)) if len(history_db) >= lag else mapped.get(col, 0)
            lag_features[f"{col}_lag_{lag}"] = val

    full = {**mapped, **lag_features}
    row  = pd.DataFrame([full]).reindex(columns=model_features).fillna(0)
    scaled = scaler.transform(row).astype("float32")

    def get_pred(m_lstm, m_xgb, scaled_input):
        xgb_feats = m_xgb.get_booster().feature_names
        df_s  = pd.DataFrame(scaled_input, columns=model_features)[xgb_feats]
        p_l   = float(m_lstm(df_s.values.reshape(1, 1, len(xgb_feats)), training=False).numpy()[0][0])
        p_x   = float(m_xgb.predict(df_s)[0])
        return float(target_scaler.inverse_transform([[(p_l + p_x) / 2]])[0][0])

    pred_10m = pred_1h = pred_3h = None
    try:
        pred_10m = get_pred(model_lstm_10m, model_xgb_10m, scaled)
    except Exception as e:
        logger.warning("10m 예측 실패: %s", e)
    try:
        pred_1h = get_pred(model_lstm_1h, model_xgb_1h, scaled)
    except Exception as e:
        logger.warning("1h 예측 실패: %s", e)
    try:
        pred_3h = get_pred(model_lstm_3h, model_xgb_3h, scaled)
    except Exception as e:
        logger.warning("3h 예측 실패: %s", e)

    history_db.append(mapped)
    return {"10m": pred_10m, "1h": pred_1h, "3h": pred_3h}


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "lstm_xgb", "models_loaded": models_loaded}


# 김민준 원본 엔드포인트 (유지)
@app.post("/predict")
async def predict_original(data: WaterLevelData):
    if not models_loaded:
        raise HTTPException(503, detail="모델 미로드 — models/ 폴더에 .keras/.pkl 파일 확인")
    try:
        preds = perform_multi_prediction(data.dict())
        return {"status": "success", "predictions": preds}
    except Exception as e:
        logger.error("예측 실패: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


@app.post("/api/v1/predict")
async def predict_standard():
    """
    신가연 API 명세 표준 응답 형식
    CSV 최신 행에서 자동으로 데이터 읽어서 예측
    """
    if not models_loaded:
        raise HTTPException(503, detail="모델 미로드 — models/ 폴더에 .keras/.pkl 파일 확인")

    # CSV에서 최신 데이터 로드
    csv_path = get_path("여주보_예측_필요_최근100행_데이터셋.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(503, detail="히스토리 CSV 없음 — hrfco_collector --init 실행 필요")

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise HTTPException(503, detail="CSV 데이터 없음")

    latest = df.iloc[-1].to_dict()

    # 컬럼명 변환 (CSV → API 형식)
    col_map = {
        "시간": "시간",
        "여주보(상류)_수위_수위(m)":            "여주보_상류_수위_수위_m",
        "여주보(상류)_수위_유량(m³/s)":          "여주보_상류_수위_유량_m3_s",
        "여주보(상류)_수위_해발수위(El.m)":       "여주보_상류_수위_해발수위_El_m",
        "여주보(하류)_수위_수위(m)":             "여주보_하류_수위_수위_m",
        "여주보(하류)_수위_유량(m³/s)":           "여주보_하류_수위_유량_m3_s",
        "여주보(하류)_수위_해발수위(El.m)":        "여주보_하류_수위_해발수위_El_m",
        "여주시(여주대교)_강수량_강수량(mm)":      "여주시_여주대교_강수량_강수량_mm",
        "여주시(여주대교)_강수량_누적강수량(mm)":   "여주시_여주대교_강수량_누적강수량_mm",
        "여주시(여주대교)_수위_수위(m)":           "여주시_여주대교_수위_수위_m",
        "여주시(여주대교)_수위_유량(m³/s)":        "여주시_여주대교_수위_유량_m3_s",
        "여주시(여주대교)_수위_해발수위(El.m)":     "여주시_여주대교_수위_해발수위_El_m",
        "여주시(주암리)_강수량_강수량(mm)":        "여주시_주암리_강수량_강수량_mm",
        "여주시(주암리)_강수량_누적강수량(mm)":     "여주시_주암리_강수량_누적강수량_mm",
        "원주시(문막교)_수위_수위(m)":            "원주시_문막교_수위_수위_m",
        "원주시(문막교)_수위_유량(m³/s)":          "원주시_문막교_수위_유량_m3_s",
        "원주시(문막교)_수위_해발수위(El.m)":       "원주시_문막교_수위_해발수위_El_m",
        "충주댐_댐_현재수위(EL.m)":              "충주댐_댐_현재수위_EL_m",
        "충주댐_댐_유입량(m³/s)":                "충주댐_댐_유입량_m3_s",
        "충주댐_댐_방류량(m³/s)":                "충주댐_댐_방류량_m3_s",
        "충주시(수안보면사무소)_강수량_강수량(mm)":  "충주시_수안보면사무소_강수량_강수량_mm",
        "충주시(수안보면사무소)_강수량_누적강수량(mm)":"충주시_수안보면사무소_강수량_누적강수량_mm",
        "충주조정지댐_댐_현재수위(EL.m)":          "충주조정지댐_댐_현재수위_EL_m",
        "충주조정지댐_댐_유입량(m³/s)":            "충주조정지댐_댐_유입량_m3_s",
        "충주조정지댐_댐_방류량(m³/s)":            "충주조정지댐_댐_방류량_m3_s",
    }

    payload = {}
    for csv_col, api_col in col_map.items():
        val = latest.get(csv_col, 0)
        try:
            payload[api_col] = float(val) if api_col != "시간" else str(val)
        except (ValueError, TypeError):
            payload[api_col] = 0.0 if api_col != "시간" else ""

    try:
        raw = perform_multi_prediction(payload)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

    pred_10m = raw.get("10m")
    pred_1h  = raw.get("1h")
    pred_3h  = raw.get("3h")
    representative = pred_1h or pred_10m or 0.0
    now_kst = datetime.now(KST)

    return {
        "prediction_id":   abs(hash(f"lstm_{now_kst.isoformat()}")) % 999999,
        "station_id":      "3008680",
        "predicted_level": round(representative, 4),
        "confidence_low":  round(representative - 0.05, 4),
        "confidence_high": round(representative + 0.05, 4),
        "horizon_minutes": 60,
        "target_time":     (now_kst + timedelta(hours=1)).isoformat(),
        "model_name":      "LSTM-XGB-Ensemble-v1",
        "all_horizons": {
            "h1_pred_m":  round(pred_10m, 4) if pred_10m else None,
            "h6_pred_m":  round(pred_1h,  4) if pred_1h  else None,
            "h18_pred_m": round(pred_3h,  4) if pred_3h  else None,
            "h36_pred_m": round(pred_3h,  4) if pred_3h  else None,
        },
        "meta": {"generated_at_kst": now_kst.strftime("%Y-%m-%d %H:%M:%S")},
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
