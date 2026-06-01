"""
predictor/app/main.py
정휘수(Hydro-MAST) + 김민준(LSTM/XGB) 예측 서비스 어댑터

역할:
  - 정휘수의 Hydro-MAST API (포트 8787) 래핑
  - 김민준의 LSTM/XGB API (포트 8000) 래핑
  - llm_service는 항상 POST /api/v1/predict 만 호출
  - ?model=hydro_mast|lstm_xgb|both 파라미터로 선택 가능
"""

import os
import logging
from datetime import datetime, timedelta, timezone

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query

logging.basicConfig(level="INFO")
logger = logging.getLogger("predictor")

HYDRO_MAST_URL = os.getenv("HYDRO_MAST_URL", "http://hydro_mast:8787")
LSTM_XGB_URL   = os.getenv("LSTM_XGB_URL",   "http://lstm_xgb:8000")
SKIP_API       = os.getenv("SKIP_API", "true").lower() == "true"
DEFAULT_MODEL  = os.getenv("DEFAULT_MODEL", "hydro_mast")

KST = timezone(timedelta(hours=9))

app = FastAPI(
    title="여주보 수위 예측 어댑터",
    description="Hydro-MAST + LSTM/XGB를 내부 표준 스키마로 변환하는 어댑터",
    version="2.0.0",
)

# 지평별 RMSE (신뢰구간 계산용)
RMSE_HYDRO = {"h1_pred_m":0.0177,"h6_pred_m":0.0198,"h18_pred_m":0.0225,"h36_pred_m":0.0261}
HORIZON_KEY_MAP = {10:"h1_pred_m", 60:"h6_pred_m", 180:"h18_pred_m", 360:"h36_pred_m"}


@app.get("/health")
async def health():
    hydro_ok = lstm_ok = False
    async with httpx.AsyncClient(timeout=5) as c:
        try:
            r = await c.get(f"{HYDRO_MAST_URL}/api/health")
            hydro_ok = r.status_code == 200
        except Exception:
            pass
        try:
            r = await c.get(f"{LSTM_XGB_URL}/health")
            lstm_ok = r.status_code == 200
        except Exception:
            pass
    return {
        "status": "ok",
        "service": "predictor_adapter",
        "hydro_mast_url": HYDRO_MAST_URL,
        "lstm_xgb_url": LSTM_XGB_URL,
        "hydro_mast_available": hydro_ok,
        "lstm_xgb_available": lstm_ok,
        "skip_api": SKIP_API,
        "default_model": DEFAULT_MODEL,
    }


# ── Hydro-MAST 호출 (기존 코드 유지) ────────────────────────────────────────
async def _call_hydro_mast(horizon_minutes: int = 60) -> dict:
    closest = min(HORIZON_KEY_MAP.keys(), key=lambda x: abs(x - horizon_minutes))
    h_key = HORIZON_KEY_MAP[closest]

    try:
        async with httpx.AsyncClient(timeout=30) as c:
            resp = await c.post(
                f"{HYDRO_MAST_URL}/api/predict",
                json={"skip_api": SKIP_API},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        raise HTTPException(503, detail=f"Hydro-MAST 연결 실패: {HYDRO_MAST_URL}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, detail=f"Hydro-MAST 오류: {e}")

    # hydro_mast 응답: {"ok": true, "payload": {"predictions_m": {...}, "meta": {...}}}
    payload = data.get("payload", data)  # payload 래핑 여부 자동 처리
    predictions_m: dict = payload.get("predictions_m", {})
    meta: dict          = payload.get("meta", {})

    predicted_level = predictions_m.get(h_key)
    if predicted_level is None:
        raise HTTPException(502, detail=f"Hydro-MAST 응답에 {h_key} 없음: {list(predictions_m.keys())}")

    rmse = RMSE_HYDRO.get(h_key, 0.03)
    bucket_t = meta.get("bucket_t") or meta.get("bucket_start_kst", datetime.now(KST).isoformat())
    try:
        base_dt = datetime.fromisoformat(str(bucket_t).replace(" ", "T"))
        if base_dt.tzinfo is None:
            base_dt = base_dt.replace(tzinfo=KST)
        target_time = (base_dt + timedelta(minutes=closest)).isoformat()
    except Exception:
        target_time = (datetime.now(KST) + timedelta(minutes=closest)).isoformat()

    return {
        "prediction_id":   abs(hash(f"{3008680}_{target_time}")) % 999999,
        "station_id":      "3008680",
        "predicted_level": round(float(predicted_level), 4),
        "confidence_low":  round(float(predicted_level) - 1.96 * rmse, 4),
        "confidence_high": round(float(predicted_level) + 1.96 * rmse, 4),
        "horizon_minutes": closest,
        "target_time":     target_time,
        "model_name":      "Hydro-MAST-v2",
        "all_horizons":    {k: round(float(v), 4) for k,v in predictions_m.items() if isinstance(v,(int,float))},
        "meta":            meta,
    }


# ── LSTM/XGB 호출 (김민준 신규) ──────────────────────────────────────────────
async def _call_lstm_xgb() -> dict:
    """
    김민준 LSTM/XGB 서버의 /api/v1/predict 호출.
    신가연 표준 형식으로 응답하는 엔드포인트.
    """
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            resp = await c.post(f"{LSTM_XGB_URL}/api/v1/predict")
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(503, detail=f"LSTM/XGB 연결 실패: {LSTM_XGB_URL}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, detail=f"LSTM/XGB 오류: {e}")


# ── 통합 예측 엔드포인트 ─────────────────────────────────────────────────────
@app.post("/api/v1/predict")
async def predict(
    station_id: str = "3008680",
    horizon_minutes: int = 60,
    model: str = Query(default=None, description="hydro_mast | lstm_xgb | both"),
):
    """
    모델 선택:
      - model=hydro_mast (기본): Hydro-MAST 단독
      - model=lstm_xgb: LSTM/XGB 단독
      - model=both: 두 모델 동시 호출, 가능한 것 반환
    """
    use_model = model or DEFAULT_MODEL

    if use_model == "hydro_mast":
        return await _call_hydro_mast(horizon_minutes)

    elif use_model == "lstm_xgb":
        return await _call_lstm_xgb()

    elif use_model == "both":
        results = {}
        try:
            results["hydro_mast"] = await _call_hydro_mast(horizon_minutes)
        except Exception as e:
            results["hydro_mast"] = {"error": str(e)}
        try:
            results["lstm_xgb"] = await _call_lstm_xgb()
        except Exception as e:
            results["lstm_xgb"] = {"error": str(e)}

        # hydro_mast 우선, 없으면 lstm_xgb
        primary = next(
            (r for r in [results.get("hydro_mast"), results.get("lstm_xgb")]
             if r and "error" not in r),
            None,
        )
        if not primary:
            raise HTTPException(502, detail=f"두 모델 모두 실패: {results}")
        primary["model_comparison"] = results
        return primary

    else:
        raise HTTPException(400, detail=f"model 파라미터 오류. hydro_mast|lstm_xgb|both 중 선택")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=False)
