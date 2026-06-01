# -*- coding: utf-8 -*-
"""Hydro-MAST — 그래프 노드 시퀀스 데이터 (v2 CSV, 직전 SEQ_LOOKBACK만 사용)"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config_v2 import (
    DATA_DIR,
    DOCS_DIR,
    MULTI_HORIZONS,
    SEQ_LOOKBACK,
)

HRFC_TOPO_ORDER = [
    "1007615",
    "1007617",
    "1007620",
    "1007625",
    "1007626",
    "1007633",
    "1007635",
    "1007637",
    "1007640",
    "1007650",
    "1007655",
    "1007656",
    "1007662",
    "1007664",
    "1007639",
    "1007641",
]

DAM_COLS = [
    "kwater_댐수위_m",
    "kwater_강우량_mm",
    "kwater_유입량_m3s",
    "kwater_총방류량_m3s",
]

TIME_COLS = ["time_sin_hour", "time_cos_hour", "time_sin_doy", "time_cos_doy"]

N_HRFC = len(HRFC_TOPO_ORDER)
N_NODES = N_HRFC + 1
TARGET_NODE = HRFC_TOPO_ORDER.index("1007639")


def hrfc_wl_col(station_id: str) -> str:
    return f"hrfc_{station_id}_수위_m"


def build_edge_index() -> tuple[np.ndarray, np.ndarray]:
    edges: list[tuple[int, int]] = []
    for i in range(N_HRFC - 1):
        edges.append((i, i + 1))
    dam_idx = N_HRFC
    edges.append((dam_idx, TARGET_NODE))
    src = np.array([e[0] for e in edges], dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64)
    return src, dst


@dataclass
class HydroTensorSpec:
    seq_len: int
    n_nodes: int
    hrfc_channels: int
    dam_channels: int
    time_dim: int
    horizons: list[int]
    wl_cols: list[str]
    dam_cols: list[str]
    c_pad: int


def load_spec_columns() -> tuple[list[str], list[str]]:
    with open(DOCS_DIR / "preprocess_v2_spec.json", encoding="utf-8") as f:
        spec = json.load(f)
    feats = set(spec["feature_columns"])
    wl = [hrfc_wl_col(s) for s in HRFC_TOPO_ORDER]
    dam = [c for c in DAM_COLS if c in feats]
    return wl, dam


def make_tensor_spec() -> HydroTensorSpec:
    wl, dam = load_spec_columns()
    c_pad = max(2, len(dam))
    return HydroTensorSpec(
        seq_len=SEQ_LOOKBACK,
        n_nodes=N_NODES,
        hrfc_channels=2,
        dam_channels=len(dam),
        time_dim=len(TIME_COLS),
        horizons=list(MULTI_HORIZONS),
        wl_cols=wl,
        dam_cols=dam,
        c_pad=c_pad,
    )


def target_cols_scaled() -> list[str]:
    return [f"target_수위_m_h{h}_scaled" for h in MULTI_HORIZONS]


def load_train_holdout_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "features_v2_train.csv", encoding="utf-8-sig")
    holdout = pd.read_csv(DATA_DIR / "features_v2_test.csv", encoding="utf-8-sig")
    return train, holdout


def _to_mat(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)


def precompute_sequences(df: pd.DataFrame, spec: HydroTensorSpec, tcols: list[str]) -> dict:
    """전체 시퀀스를 sliding_window로 벡터 생성"""
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(df)
    t, c_pad = spec.seq_len, spec.c_pad
    wl_mat = _to_mat(df, spec.wl_cols)
    wl_ok = df[spec.wl_cols].notna().values.astype(np.float32)
    dam_mat = _to_mat(df, spec.dam_cols) if spec.dam_cols else None
    time_mat = _to_mat(df, TIME_COLS)
    y_mat = df[tcols].astype(float).values.astype(np.float32)

    wl_w = sliding_window_view(wl_mat, window_shape=t, axis=0)
    ok_w = sliding_window_view(wl_ok, window_shape=t, axis=0)
    valid = np.arange(t, n)
    valid = valid[np.all(np.isfinite(y_mat[valid]), axis=1)]
    wi = valid - t

    m = len(valid)
    nodes = np.zeros((m, t, N_NODES, c_pad), dtype=np.float32)
    nodes[:, :, :N_HRFC, 0] = np.transpose(wl_w[wi], (0, 2, 1))
    nodes[:, :, :N_HRFC, 1] = np.transpose(ok_w[wi], (0, 2, 1))
    if dam_mat is not None and spec.dam_channels:
        dam_w = sliding_window_view(dam_mat, window_shape=t, axis=0)
        nodes[:, :, N_HRFC, : spec.dam_channels] = np.transpose(dam_w[wi], (0, 2, 1))
    times = time_mat[valid - 1]
    targets = y_mat[valid]

    return {"nodes": nodes, "times": times, "targets": targets, "indices": valid}


def subsample_cache(cache: dict, stride: int = 1) -> dict:
    if stride <= 1:
        return cache
    sl = slice(None, None, stride)
    return {
        "nodes": cache["nodes"][sl],
        "times": cache["times"][sl],
        "targets": cache["targets"][sl],
        "indices": cache["indices"][sl],
    }


class HydroMASTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        spec: HydroTensorSpec,
        target_cols_scaled: list[str],
        mask_prob: float = 0.0,
        cache: dict | None = None,
        stride: int = 1,
    ):
        self.mask_prob = mask_prob
        self.n_hrfc = N_HRFC
        if cache is None:
            cache = precompute_sequences(df, spec, target_cols_scaled)
        cache = subsample_cache(cache, stride)
        self.nodes = torch.from_numpy(cache["nodes"])
        self.times = torch.from_numpy(cache["times"])
        self.targets = torch.from_numpy(cache["targets"])

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int):
        nodes = self.nodes[i]
        if self.mask_prob > 0:
            nodes = nodes.clone()
            for k in range(self.n_hrfc):
                if torch.rand(1).item() < self.mask_prob:
                    nodes[:, k, 0] = 0.0
                    nodes[:, k, 1] = 0.0
        return nodes, self.times[i], self.targets[i]


def build_node_tensor_window(
    df: pd.DataFrame,
    end_idx: int,
    spec: HydroTensorSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """단건 추론용 (발표 서버 배치는 precompute 권장)"""
    start = end_idx - spec.seq_len
    win = df.iloc[start:end_idx]
    t = spec.seq_len
    nodes = np.zeros((t, N_NODES, spec.c_pad), dtype=np.float32)
    for i, wl_c in enumerate(spec.wl_cols):
        s = win[wl_c].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0).values
        m = (win[wl_c].notna() & np.isfinite(s)).astype(np.float32).values
        nodes[:, i, 0] = s.astype(np.float32)
        nodes[:, i, 1] = m
    if spec.dam_channels:
        nodes[:, N_HRFC, : spec.dam_channels] = _to_mat(win, spec.dam_cols)
    time_g = win.iloc[-1][TIME_COLS].astype(float).values.astype(np.float32)
    return nodes, time_g, np.ones(N_NODES, dtype=np.float32)
