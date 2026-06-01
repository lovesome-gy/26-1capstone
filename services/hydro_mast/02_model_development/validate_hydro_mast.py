# -*- coding: utf-8 -*-
"""
Hydro-MAST 검증 — holdout 10% 구간, 미터 단위 지표, 추론 시간

실행: .venv\\Scripts\\python validate_hydro_mast.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config_v2 import DATA_DIR, DOCS_DIR, MODEL_DIR, MULTI_HORIZON_LABELS, MULTI_HORIZONS
from hydro_mast_data import (
    HydroMASTDataset,
    load_train_holdout_frames,
    make_tensor_spec,
    precompute_sequences,
    target_cols_scaled,
)
from hydro_mast_net import HydroMAST
from hydro_mast_predict import load_model

DOCS_DIR.mkdir(parents=True, exist_ok=True)


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return float("nan")
    return 1 - np.sum((y_true - y_pred) ** 2) / denom


def metrics_m(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "nse": float(nse(y_true, y_pred)),
        "rmse_m": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae_m": float(np.mean(np.abs(y_true - y_pred))),
        "bias_m": float(np.mean(y_pred - y_true)),
    }


def main():
    t_all = time.perf_counter()
    with open(DOCS_DIR / "preprocess_v2_spec.json", encoding="utf-8") as f:
        pspec = json.load(f)

    train_df, hold_df = load_train_holdout_frames()
    spec = make_tensor_spec()
    tcols = target_cols_scaled()
    scaler = joblib.load(MODEL_DIR / "target_scaler_v2.pkl")

    hold_start = pspec.get("holdout_start")
    hold_end = str(hold_df["시간"].max()) if "시간" in hold_df.columns else None
    n_hold = len(hold_df)

    print("=" * 60)
    print("Hydro-MAST 검증 (holdout 10%)")
    print("=" * 60)
    print(f"학습 구간: {pspec['data_start']} ~ {pspec['train_end']}")
    print(f"검증 구간: {hold_start} ~ {hold_end}")
    print(f"검증 샘플: {n_hold:,}행 (시퀀스 유효 샘플은 lookback 제외 후 계산)")
    print(f"입력 윈도우: 직전 {spec.seq_len}스텝 = {spec.seq_len / 6:.0f}시간")
    print()

    t0 = time.perf_counter()
    print("[1] holdout 시퀀스 생성...")
    hold_cache = precompute_sequences(hold_df, spec, tcols)
    t_cache = time.perf_counter() - t0

    hold_ds = HydroMASTDataset(hold_df, spec, tcols, mask_prob=0.0, cache=hold_cache)
    hold_loader = DataLoader(hold_ds, batch_size=512, shuffle=False, num_workers=0)
    n_valid = len(hold_ds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, spec = load_model(device)

    t1 = time.perf_counter()
    print("[2] holdout 예측 (scaled)...")
    model.eval()
    n_dam = spec.dam_channels
    ys, ps = [], []
    with torch.no_grad():
        for nodes, time_g, y in hold_loader:
            pred = model(nodes.to(device), time_g.to(device), n_dam)
            ys.append(y.numpy())
            ps.append(pred.cpu().numpy())
    y_s = np.vstack(ys)
    p_s = np.vstack(ps)
    t_infer = time.perf_counter() - t1

    idx = hold_cache["indices"]
    times = pd.to_datetime(hold_df.iloc[idx]["시간"]).astype(str).tolist()

    report = {
        "model": "hydro_mast_v2",
        "validation_type": "holdout_10pct",
        "train_period": f"{pspec['data_start']} ~ {pspec['train_end']}",
        "validation_period": {"start": hold_start, "end": hold_end},
        "n_holdout_rows": n_hold,
        "n_validation_samples": n_valid,
        "seq_lookback_steps": spec.seq_len,
        "seq_lookback_hours": spec.seq_len / 6,
        "timing_sec": {
            "precompute_sequences": round(t_cache, 2),
            "inference_holdout": round(t_infer, 2),
            "per_sample_ms": round(1000 * t_infer / max(n_valid, 1), 3),
            "total": 0.0,
        },
        "scaled": {},
        "meters": {},
        "peak_top10pct_meters": {},
    }

    print()
    print("[3] 지표 (미터 단위)")
    print("-" * 60)
    for i, h in enumerate(MULTI_HORIZONS):
        lab = MULTI_HORIZON_LABELS[h]
        yt_m = scaler.inverse_transform(y_s[:, i].reshape(-1, 1)).ravel()
        yp_m = scaler.inverse_transform(p_s[:, i].reshape(-1, 1)).ravel()
        m_s = metrics_m(yt_m, yp_m)
        key = f"h{h}_{lab}"
        report["scaled"][key] = {
            "nse": float(nse(y_s[:, i], p_s[:, i])),
            "rmse": float(np.sqrt(np.mean((y_s[:, i] - p_s[:, i]) ** 2))),
        }
        report["meters"][key] = m_s
        print(
            f"  {lab:>6} (h{h:>2})  NSE={m_s['nse']:.4f}  "
            f"RMSE={m_s['rmse_m']*100:.2f}cm  MAE={m_s['mae_m']*100:.2f}cm  "
            f"bias={m_s['bias_m']*100:+.2f}cm"
        )

        q90 = np.quantile(yt_m, 0.90)
        mask = yt_m >= q90
        if mask.sum() >= 10:
            mp = metrics_m(yt_m[mask], yp_m[mask])
            report["peak_top10pct_meters"][key] = mp
            print(
                f"         [상위10% 수위] NSE={mp['nse']:.4f}  "
                f"RMSE={mp['rmse_m']*100:.2f}cm  (n={int(mask.sum())})"
            )

    report["timing_sec"]["total"] = round(time.perf_counter() - t_all, 2)

    print()
    print("[4] 검증 소요 시간")
    print(f"  시퀀스 생성: {report['timing_sec']['precompute_sequences']:.1f}s")
    print(f"  holdout 추론: {report['timing_sec']['inference_holdout']:.1f}s  "
          f"({report['timing_sec']['per_sample_ms']:.2f} ms/샘플)")
    print(f"  전체: {report['timing_sec']['total']:.1f}s")

    out = DOCS_DIR / "hydro_mast_validation_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print()
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
