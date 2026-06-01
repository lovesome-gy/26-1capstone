# -*- coding: utf-8 -*-
"""홍수 위험도(확률) 계산 — 발표용"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# 단기 = 1시간 후(h6), 장기 = 6시간 후(h36)
SHORT_H = 6
LONG_H = 36

# 모델 RMSE(m) — holdout 기준 (unified_model_metrics.json, 없으면 기본값)
def _load_sigma_m() -> dict[int, float]:
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parent / "docs" / "hydro_mast_metrics.json"
    if not p.exists():
        return {6: 0.031, 18: 0.068, 36: 0.096}
    with open(p, encoding="utf-8") as f:
        m = json.load(f)
    block = m.get("hydro_mast", {})
    labels = {6: "1시간", 18: "3시간", 36: "6시간"}
    out = {}
    for h, lab in labels.items():
        key = f"h{h}_{lab}"
        if key in block:
            out[h] = float(block[key]["rmse"])
    return out or {6: 0.031, 18: 0.068, 36: 0.096}


SIGMA_M = _load_sigma_m()

DEFAULT_ALERT_PROB_PCT = 60.0


def train_levels(train_csv: str | pd.DataFrame) -> np.ndarray:
    if isinstance(train_csv, pd.DataFrame):
        return train_csv["target_수위_m"].astype(float).values
    return pd.read_csv(train_csv, usecols=["target_수위_m"])["target_수위_m"].astype(float).values


def default_flood_threshold_m(train_csv: str | pd.DataFrame) -> float:
    """학습 구간 90% 수위 — 여주 데이터는 변동폭이 작아 p90이 발표용으로 적당"""
    s = train_levels(train_csv)
    return float(np.round(np.quantile(s, 0.90), 2))


def flood_probability_pct(
    pred_m: float,
    threshold_m: float,
    sigma_m: float | None = None,
) -> float:
    """
    예측 불확실성을 반영한 '임계치 초과 확률' (%).
    예측 수위 ~ N(pred, sigma) 일 때 P(수위 > 임계치).
    """
    sig = max(float(sigma_m or 0.08), 0.02)
    z = (float(threshold_m) - float(pred_m)) / sig
    # P(X > threshold) = Φ(-z)
    p = 0.5 * (1.0 + math.erf(-z / math.sqrt(2.0)))
    return float(np.clip(100.0 * p, 0.0, 100.0))


def risk_level(prob_pct: float, alert_prob_pct: float) -> str:
    if prob_pct >= alert_prob_pct:
        return "alert"
    if prob_pct >= max(15.0, alert_prob_pct * 0.4):
        return "watch"
    return "safe"


def risk_label(level: str) -> str:
    return {"safe": "안전", "watch": "주의", "alert": "경고"}[level]
