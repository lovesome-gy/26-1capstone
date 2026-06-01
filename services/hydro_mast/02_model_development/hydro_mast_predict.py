# -*- coding: utf-8 -*-
"""Hydro-MAST 추론 (발표 서버·배치 예측)"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config_v2 import MODEL_DIR, MULTI_HORIZONS
from hydro_mast_data import (
    HydroTensorSpec,
    load_train_holdout_frames,
    make_tensor_spec,
    precompute_sequences,
    target_cols_scaled,
)
from hydro_mast_net import HydroMAST

ROOT = Path(__file__).resolve().parent


def load_model(device: torch.device | None = None) -> tuple[HydroMAST, dict, HydroTensorSpec]:
    device = device or torch.device("cpu")
    ckpt = torch.load(MODEL_DIR / "hydro_mast_v2.pt", map_location=device, weights_only=False)
    sp = ckpt["spec"]
    c_pad = sp.get("c_pad") or max(sp["hrfc_channels"], sp["dam_channels"])
    spec = HydroTensorSpec(
        seq_len=sp["seq_len"],
        n_nodes=sp["n_nodes"],
        hrfc_channels=sp["hrfc_channels"],
        dam_channels=sp["dam_channels"],
        time_dim=sp["time_dim"],
        horizons=sp["horizons"],
        wl_cols=sp["wl_cols"],
        dam_cols=sp["dam_cols"],
        c_pad=c_pad,
    )
    model = HydroMAST(
        hrfc_channels=spec.hrfc_channels,
        dam_channels=spec.dam_channels,
        time_dim=spec.time_dim,
        hidden=sp.get("hidden", 64),
        n_horizons=len(spec.horizons),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt, spec


def predict_cache(
    cache: dict,
    model: HydroMAST | None = None,
    spec: HydroTensorSpec | None = None,
    device: torch.device | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    device = device or torch.device("cpu")
    if model is None or spec is None:
        model, _, spec = load_model(device)
    n_dam = spec.dam_channels
    nodes, times = cache["nodes"], cache["times"]
    preds = []
    for start in range(0, len(nodes), batch_size):
        nt = torch.from_numpy(nodes[start : start + batch_size]).to(device)
        tt = torch.from_numpy(times[start : start + batch_size]).to(device)
        with torch.no_grad():
            preds.append(model(nt, tt, n_dam).cpu().numpy())
    return np.vstack(preds)


def build_prediction_dataframe(
    df: pd.DataFrame | None = None,
    use_train_only: bool = True,
) -> pd.DataFrame:
    """발표용: train 전 구간에 대한 다중 시계 예측 프레임"""
    import joblib

    if df is None:
        train, _ = load_train_holdout_frames()
        df = train if use_train_only else train

    spec = make_tensor_spec()
    tcols = target_cols_scaled()
    cache = precompute_sequences(df, spec, tcols)
    n = len(cache["indices"])
    if n > 25000:
        cache = {
            "nodes": cache["nodes"][-25000:],
            "times": cache["times"][-25000:],
            "targets": cache["targets"][-25000:],
            "indices": cache["indices"][-25000:],
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, spec = load_model(device)
    pred = predict_cache(cache, model, spec, device)
    indices = cache["indices"]

    scaler = joblib.load(MODEL_DIR / "target_scaler_v2.pkl")
    meta = df.iloc[indices].reset_index(drop=True)

    out = pd.DataFrame({"time": pd.to_datetime(meta["시간"]).astype(str)})
    for i, h in enumerate(MULTI_HORIZONS):
        out[f"h{h}_actual_m"] = scaler.inverse_transform(
            meta[f"target_수위_m_h{h}_scaled"].values.reshape(-1, 1)
        ).ravel()
        out[f"h{h}_pred_m"] = scaler.inverse_transform(pred[:, i].reshape(-1, 1)).ravel()
    if "target_수위_m" in meta.columns:
        out["current_m"] = meta["target_수위_m"].values
    return out.sort_values("time").reset_index(drop=True)
