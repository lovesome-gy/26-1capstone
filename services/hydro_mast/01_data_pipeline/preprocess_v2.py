# -*- coding: utf-8 -*-
"""
yeoju handoff 원본 → v2 병합·전처리·피처·정규화
논문 반영 요약:
  - 강우 결측 0 (CNN-GRU 2023)
  - 수위 ffill + 짧은 구간 보간 (CNN-GRU / 다변량 Spline)
  - 수위 IQR clip k=1.5 (yeoju handoff)
  - MinMaxScaler train 구간만 fit (결측 LSTM 2022)
  - lag + 1차차분 + cyclical time (다변량 2024)
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

from config_v2 import (
    DATA_DIR,
    DOCS_DIR,
    FFILL_LIMIT,
    HANDOFF,
    HORIZON_STEPS,
    HRFC_STATIONS,
    INTERP_LIMIT,
    IQR_K,
    KEY_SERIES,
    LAG_STEPS,
    MODEL_DIR,
    MULTI_HORIZONS,
    TARGET_COL,
    TARGET_FALLBACK,
    DATA_START,
    TRAIN_END,
    VAL_HOLDOUT_FRACTION,
)

warnings.filterwarnings("ignore")

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_kwater_time(s: str) -> pd.Timestamp:
    """예: '01-01 00시 10분' + 파일 날짜"""
    s = str(s).strip()
    m = re.match(r"(\d{2})-(\d{2})\s+(\d{2})시\s+(\d{2})분", s)
    if not m:
        return pd.NaT
    return m.group(1), m.group(2), m.group(3), m.group(4)


def load_kwater_10min() -> pd.DataFrame:
    folder = HANDOFF / "kwater" / "mntlist"
    parts = []
    for fp in sorted(folder.glob("K-water_여주보_수문현황10분_*.csv")):
        df = pd.read_csv(fp, encoding="utf-8-sig")
        if df.empty:
            continue
        rows = []
        for _, r in df.iterrows():
            traw = r.get("관측일시_10분", "")
            parsed = _parse_kwater_time(traw)
            if not isinstance(parsed, tuple):
                continue
            mm, dd, hh, mi = parsed
            ref = str(r.get("조회_종료일", r.get("조회_시작일", "")))[:10]
            year = int(ref[:4]) if ref[:4].isdigit() else int(fp.stem.split("_")[-1][:4])
            hh, mi = int(hh), int(mi)
            if hh >= 24:
                hh = 0
            try:
                ts = pd.Timestamp(year=year, month=int(mm), day=int(dd), hour=hh, minute=mi)
            except ValueError:
                continue
            rows.append(
                {
                    "시간": ts,
                    "kwater_댐수위_m": pd.to_numeric(r.get("댐수위_m(EL.m)"), errors="coerce"),
                    "kwater_강우량_mm": pd.to_numeric(r.get("강우량_mm"), errors="coerce"),
                    "kwater_유입량_m3s": pd.to_numeric(r.get("유입량_m3초"), errors="coerce"),
                    "kwater_총방류량_m3s": pd.to_numeric(r.get("총방류량_m3초"), errors="coerce"),
                    "kwater_저수량_백만m3": pd.to_numeric(r.get("저수량_백만m3"), errors="coerce"),
                    "kwater_저수율_pct": pd.to_numeric(r.get("저수율_퍼센트"), errors="coerce"),
                }
            )
        if rows:
            parts.append(pd.DataFrame(rows))
    if not parts:
        raise FileNotFoundError("K-water 10분 파일 없음")
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["시간"]).sort_values("시간")
    return out


def load_hrfc_10min() -> pd.DataFrame:
    folder = HANDOFF / "hrfc" / "waterlevel_10m"
    chunks = []
    for fp in folder.glob("HRFCO_수위10분_*.csv"):
        code_m = re.search(r"_(\d{7})_", fp.name)
        if not code_m:
            continue
        code = code_m.group(1)
        df = pd.read_csv(fp, encoding="utf-8-sig", usecols=["관측시각_년월일시분", "수위_m", "유량_m3초"])
        df["시간"] = pd.to_datetime(df["관측시각_년월일시분"].astype(str), format="%Y%m%d%H%M", errors="coerce")
        df["code"] = code
        df["수위_m"] = pd.to_numeric(df["수위_m"], errors="coerce")
        df["유량_m3초"] = pd.to_numeric(df["유량_m3초"], errors="coerce")
        chunks.append(df[["시간", "code", "수위_m", "유량_m3초"]].dropna(subset=["시간"]))
    if not chunks:
        raise FileNotFoundError("HRFC 10분 파일 없음")
    long_df = pd.concat(chunks, ignore_index=True)
    long_df = long_df.sort_values(["code", "시간"]).drop_duplicates(["code", "시간"])

    wl = long_df.pivot_table(index="시간", columns="code", values="수위_m", aggfunc="mean")
    fl = long_df.pivot_table(index="시간", columns="code", values="유량_m3초", aggfunc="mean")
    wl.columns = [f"hrfc_{c}_수위_m" for c in wl.columns]
    fl.columns = [f"hrfc_{c}_유량_m3s" for c in fl.columns]
    wide = wl.join(fl, how="outer").reset_index()

    wl_cols = [c for c in wide.columns if c.endswith("_수위_m")]
    wide["hrfc_수위_m_평균"] = wide[wl_cols].mean(axis=1, skipna=True)
    wide["hrfc_관측소수"] = wide[wl_cols].notna().sum(axis=1)
    return wide.sort_values("시간")


def build_merged_raw() -> pd.DataFrame:
    print("[1] K-water 10분 로드...")
    kw = load_kwater_10min()
    print(f"  K-water: {len(kw):,}행")
    print("[2] HRFC 10분 로드 (관측소별 pivot)...")
    hr = load_hrfc_10min()
    print(f"  HRFC wide: {len(hr):,}행 × {len(hr.columns)}열")
    grid_start = min(kw["시간"].min(), hr["시간"].min())
    grid_end = max(kw["시간"].max(), hr["시간"].max())
    grid = pd.date_range(grid_start.floor("10min"), grid_end.ceil("10min"), freq="10min")
    base = pd.DataFrame({"시간": grid})
    merged = base.merge(kw, on="시간", how="left").merge(hr, on="시간", how="left")
    return merged


def iqr_clip(s: pd.Series, k: float = IQR_K) -> pd.Series:
    x = s.astype(float)
    valid = x.dropna()
    if len(valid) < 10:
        return x
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return x
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return x.clip(lo, hi)


def preprocess_series(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = [c for c in out.columns if c != "시간"]

    rain_cols = [c for c in num_cols if "강우" in c]
    flow_cols = [c for c in num_cols if "유량" in c or "방류" in c or "유입" in c]
    wl_cols = [c for c in num_cols if "수위" in c or "댐수위" in c]

    for c in rain_cols + flow_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    for c in wl_cols:
        s = pd.to_numeric(out[c], errors="coerce")
        s = iqr_clip(s)
        s = s.interpolate(method="linear", limit=INTERP_LIMIT, limit_direction="both")
        s = s.ffill(limit=FFILL_LIMIT).bfill(limit=FFILL_LIMIT)
        out[c] = s

    for c in num_cols:
        if c not in rain_cols + flow_cols + wl_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c].interpolate(method="linear", limit=INTERP_LIMIT).ffill(limit=FFILL_LIMIT)

    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["시간"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    doy = ts.dt.dayofyear
    out["time_sin_hour"] = np.sin(2 * np.pi * hour / 24)
    out["time_cos_hour"] = np.cos(2 * np.pi * hour / 24)
    out["time_sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    out["time_cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def add_diff_lag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    series = [c for c in KEY_SERIES if c in out.columns]
    for col in series:
        s = out[col]
        out[f"{col}_diff1"] = s.diff(1)
        for lag in LAG_STEPS:
            out[f"{col}_lag_{lag}"] = s.shift(lag)
    return out


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if TARGET_COL in out.columns and out[TARGET_COL].notna().sum() > 100:
        out["target_수위_m"] = out[TARGET_COL]
    else:
        out["target_수위_m"] = out[TARGET_FALLBACK]
    for h in MULTI_HORIZONS:
        out[f"target_수위_m_h{h}"] = out["target_수위_m"].shift(-h)
    # 하위 호환
    out["target_수위_m_h36"] = out[f"target_수위_m_h{HORIZON_STEPS}"]
    return out


def split_train_holdout(df: pd.DataFrame):
    """2년 전체 = 학습 CSV. 말미 10% = 지표용 holdout(test 파일명 유지)."""
    ts = pd.to_datetime(df["시간"])
    all_data = df[(ts >= DATA_START) & (ts <= TRAIN_END)].copy()
    n = len(all_data)
    cut = max(1, int(n * (1 - VAL_HOLDOUT_FRACTION)))
    holdout = all_data.iloc[cut:].copy()
    train = all_data.copy()
    return train, holdout


def fit_scalers(train: pd.DataFrame, feature_cols: list):
    X = train[feature_cols].astype(float)
    y = train[["target_수위_m"]].astype(float)
    fx = MinMaxScaler()
    fy = MinMaxScaler()
    fx.fit(X)
    fy.fit(y)
    return fx, fy


def main():
    print("=" * 60)
    print("yeoju v2 전처리 시작")
    print("=" * 60)

    raw = build_merged_raw()
    print(f"[OK] 병합: {len(raw):,}행")

    prep = preprocess_series(raw)
    print(f"[OK] 결측·이상치 처리 완료")

    feat = add_time_features(prep)
    feat = add_diff_lag(feat)
    feat = build_target(feat)

    meta_cols = {"시간", "target_수위_m", "target_수위_m_h36"}
    meta_cols |= {f"target_수위_m_h{h}" for h in MULTI_HORIZONS}
    feature_cols = [c for c in feat.columns if c not in meta_cols and feat[c].dtype != "object"]

    train, holdout = split_train_holdout(feat)
    usable = train.dropna(subset=["target_수위_m"]).index
    train = train.loc[usable]
    holdout = holdout.dropna(subset=["target_수위_m"])

    fx, fy = fit_scalers(train, feature_cols)
    joblib.dump(fx, MODEL_DIR / "feature_scaler_v2.pkl")
    joblib.dump(fy, MODEL_DIR / "target_scaler_v2.pkl")

    def scale_split(df):
        d = df.copy()
        d[feature_cols] = fx.transform(d[feature_cols].astype(float))
        d["target_수위_m_scaled"] = fy.transform(d[["target_수위_m"]].astype(float))
        for h in MULTI_HORIZONS:
            col = f"target_수위_m_h{h}"
            if col in d.columns:
                m = d[col].notna()
                v = d.loc[m, col].astype(float).values.reshape(-1, 1)
                d.loc[m, f"{col}_scaled"] = fy.transform(v).ravel()
        return d

    train_s = scale_split(train)
    holdout_s = scale_split(holdout)

    train_path = DATA_DIR / "features_v2_train.csv"
    test_path = DATA_DIR / "features_v2_test.csv"  # holdout (말미 10%, 지표용)
    train_s.to_csv(train_path, index=False, encoding="utf-8-sig")
    holdout_s.to_csv(test_path, index=False, encoding="utf-8-sig")

    holdout_start = str(holdout["시간"].min()) if len(holdout) else None
    spec = {
        "target_col": TARGET_COL,
        "target_fallback": TARGET_FALLBACK,
        "horizon_steps": HORIZON_STEPS,
        "horizon_hours": HORIZON_STEPS / 6,
        "multi_horizons": MULTI_HORIZONS,
        "lag_steps": LAG_STEPS,
        "data_start": DATA_START,
        "train_end": TRAIN_END,
        "train_years": "2024-2025",
        "holdout_fraction": VAL_HOLDOUT_FRACTION,
        "holdout_start": holdout_start,
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "n_train": len(train_s),
        "n_holdout": len(holdout_s),
        "n_test": len(holdout_s),
        "hrfc_stations": HRFC_STATIONS,
        "preprocessing_rules": {
            "rain_missing": "0",
            "water_level": f"IQR_clip_k={IQR_K} -> linear_interp_limit={INTERP_LIMIT} -> ffill_limit={FFILL_LIMIT}",
            "scaling": "MinMaxScaler on 2024-2025 train (2 years)",
            "split": "train=full 2y, test csv=last 10% holdout for metrics only",
            "time_encoding": "sin/cos hour and day-of-year",
            "diff": "diff1 on key series",
        },
    }
    with open(DOCS_DIR / "preprocess_v2_spec.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)

    print(f"[OK] 저장: {train_path} / {test_path}")
    print(f"[OK] 피처 수: {len(feature_cols)}, train(2y)={len(train_s):,}, holdout={len(holdout_s):,}")
    print("완료.")


if __name__ == "__main__":
    main()
