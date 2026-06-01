# -*- coding: utf-8 -*-
"""
Hydro-MAST 학습 — 2년 train CSV, 직전 SEQ_LOOKBACK(12h)만 입력, 단기+장기 동시 예측

산출:
  models/hydro_mast_v2.pt
  docs/hydro_mast_metrics.json
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from config_v2 import (
    DATA_START,
    DOCS_DIR,
    MODEL_DIR,
    MULTI_HORIZON_LABELS,
    MULTI_HORIZONS,
    SEQ_LOOKBACK,
    TRAIN_END,
    VAL_HOLDOUT_FRACTION,
)
from hydro_mast_data import (
    HydroMASTDataset,
    HydroTensorSpec,
    load_train_holdout_frames,
    make_tensor_spec,
    precompute_sequences,
    target_cols_scaled,
)
from hydro_mast_net import HydroMAST, huber_loss

DOCS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024
EPOCHS = 8
LR = 1e-3
MASK_PROB = 0.2
HIDDEN = 48
TRAIN_STRIDE = 2
EVAL_EVERY = 2


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return float("nan")
    return float(1 - np.sum((y_true - y_pred) ** 2) / denom)


def eval_loader(model: HydroMAST, loader: DataLoader, spec: HydroTensorSpec) -> dict:
    model.eval()
    ys, ps = [], []
    n_dam = spec.dam_channels
    with torch.no_grad():
        for nodes, time_g, y in loader:
            nodes = nodes.to(DEVICE)
            time_g = time_g.to(DEVICE)
            pred = model(nodes, time_g, n_dam)
            ys.append(y.numpy())
            ps.append(pred.cpu().numpy())
    y_all = np.vstack(ys)
    p_all = np.vstack(ps)
    out = {}
    tcols = target_cols_scaled()
    for i, h in enumerate(MULTI_HORIZONS):
        yt, yp = y_all[:, i], p_all[:, i]
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        out[f"h{h}_{MULTI_HORIZON_LABELS[h]}"] = {
            "rmse": rmse,
            "mae": float(np.mean(np.abs(yt - yp))),
            "nse": nse(yt, yp),
        }
    out["mean_nse"] = float(np.mean([out[k]["nse"] for k in out if k.startswith("h")]))
    return out, y_all, p_all


def train_loop(
    model: HydroMAST,
    train_loader: DataLoader,
    hold_loader: DataLoader,
    spec: HydroTensorSpec,
) -> tuple[HydroMAST, dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    n_dam = spec.dam_channels
    best_nse = -1e9
    best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        n_batch = 0
        t0 = time.time()
        for nodes, time_g, y in train_loader:
            nodes = nodes.to(DEVICE, non_blocking=True)
            time_g = time_g.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(nodes, time_g, n_dam)
            loss = huber_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            n_batch += 1

        do_eval = epoch == EPOCHS or epoch % EVAL_EVERY == 0
        if do_eval:
            metrics, _, _ = eval_loader(model, hold_loader, spec)
            h36 = metrics.get(f"h36_{MULTI_HORIZON_LABELS[36]}", {}).get("nse", 0)
            sched.step(h36)
            history.append({"epoch": epoch, "loss": loss_sum / max(n_batch, 1), "holdout": metrics})
            print(
                f"epoch {epoch}/{EPOCHS} loss={loss_sum/n_batch:.5f} "
                f"holdout mean_nse={metrics['mean_nse']:.4f} h36_nse={h36:.4f} "
                f"({time.time()-t0:.1f}s)",
                flush=True,
            )
            if h36 > best_nse:
                best_nse = h36
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            print(
                f"epoch {epoch}/{EPOCHS} loss={loss_sum/n_batch:.5f} (eval skip) ({time.time()-t0:.1f}s)",
                flush=True,
            )

    if best_state:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_h36_nse": best_nse}


def save_bundle(model: HydroMAST, spec: HydroTensorSpec, metrics: dict) -> Path:
    path = MODEL_DIR / "hydro_mast_v2.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "spec": {
                "seq_len": spec.seq_len,
                "n_nodes": spec.n_nodes,
                "hrfc_channels": spec.hrfc_channels,
                "dam_channels": spec.dam_channels,
                "time_dim": spec.time_dim,
                "horizons": spec.horizons,
                "wl_cols": spec.wl_cols,
                "dam_cols": spec.dam_cols,
                "c_pad": spec.c_pad,
                "hidden": HIDDEN,
            },
            "horizons": MULTI_HORIZONS,
            "horizon_labels": MULTI_HORIZON_LABELS,
            "target_columns": target_cols_scaled(),
            "type": "hydro_mast",
        },
        path,
    )
    return path


def main():
    import sys

    print("=" * 60, flush=True)
    print("Hydro-MAST 학습 (2년, lookback=%d steps = %.1fh)" % (SEQ_LOOKBACK, SEQ_LOOKBACK / 6), flush=True)
    print("device:", DEVICE, flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()

    train_df, hold_df = load_train_holdout_frames()
    spec = make_tensor_spec()
    tcols = target_cols_scaled()

    print("[cache] train sequences...")
    train_cache = precompute_sequences(train_df, spec, tcols)
    print("[cache] holdout sequences...")
    hold_cache = precompute_sequences(hold_df, spec, tcols)
    train_ds = HydroMASTDataset(
        train_df, spec, tcols, mask_prob=MASK_PROB, cache=train_cache, stride=TRAIN_STRIDE
    )
    hold_ds = HydroMASTDataset(hold_df, spec, tcols, mask_prob=0.0, cache=hold_cache, stride=1)
    print(
        f"train samples={len(train_ds):,} (stride={TRAIN_STRIDE}, 2년 전체 커버), "
        f"holdout={len(hold_ds):,}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    hold_loader = DataLoader(hold_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = HydroMAST(
        hrfc_channels=spec.hrfc_channels,
        dam_channels=spec.dam_channels,
        time_dim=spec.time_dim,
        hidden=HIDDEN,
        n_horizons=len(MULTI_HORIZONS),
    ).to(DEVICE)

    model, train_info = train_loop(model, train_loader, hold_loader, spec)
    hold_metrics, _, _ = eval_loader(model, hold_loader, spec)

    pkl_path = save_bundle(model, spec, hold_metrics)
    results = {
        "model": "hydro_mast",
        "horizons": MULTI_HORIZONS,
        "labels": MULTI_HORIZON_LABELS,
        "train_period": f"{DATA_START} ~ {TRAIN_END}",
        "holdout_fraction": VAL_HOLDOUT_FRACTION,
        "seq_lookback_steps": SEQ_LOOKBACK,
        "seq_lookback_hours": SEQ_LOOKBACK / 6,
        "note": "2년 학습(stride=2 샘플링), 추론 시 직전 12시간(72스텝)만 사용",
        "train_stride": TRAIN_STRIDE,
        "hydro_mast": hold_metrics,
        "training": train_info,
        "recommended": "hydro_mast",
        "model_file": str(pkl_path.name),
        "usage": (
            "추론: 직전 72스텝 노드 텐서 + 시간부호화 -> 4 horizon scaled 수위. "
            "target_scaler_v2.pkl 역변환."
        ),
    }

    out_json = DOCS_DIR / "hydro_mast_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] {pkl_path}")
    print(f"[OK] {out_json}")
    print(json.dumps(hold_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
