# -*- coding: utf-8 -*-
r"""
실시간 실측 -> v2(72열) 맞춤 -> Hydro-MAST 단건 추론 파이프라인.

목표:
- API에서 최신 실측값(K-water + HRFC)을 가져온다.
- 학습 v2 스펙(72열, 동일 열 순서/전처리)으로 1행을 만든다.
- 직전 72스텝 윈도우를 구성해 모델 예측을 1회 수행한다.

실행:
  .venv\Scripts\python realtime_pipeline_v2.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, unquote
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for p in (PROJECT / "01_data_pipeline", PROJECT / "02_model_development"):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from config_v2 import DOCS_DIR, KEY_SERIES, MODEL_DIR, MULTI_HORIZONS
from hydro_mast_data import TIME_COLS, build_node_tensor_window, load_train_holdout_frames
from hydro_mast_predict import load_model

DATA_DIR = PROJECT / "04_artifacts" / "data"
ENV_PATH = PROJECT / ".env"
TAIL_WINDOW = 120  # 실시간 1행 생성 시 역정규화할 최근 구간만 사용
LOOKBACK_STEPS = 72  # 10분 * 72 = 직전 12시간
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f".env not found: {path}")
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def decode_service_key(k: str) -> str:
    return unquote(k) if "%" in k else k


def floor_10m_kst(ts: pd.Timestamp | None = None) -> pd.Timestamp:
    now = pd.Timestamp.now(tz="Asia/Seoul") if ts is None else ts.tz_localize("Asia/Seoul")
    return now.floor("10min")


def to_bucket_strings(ts_kst: pd.Timestamp) -> tuple[str, str]:
    kst = ts_kst.tz_convert("Asia/Seoul").replace(second=0, microsecond=0)
    utc = kst.tz_convert("UTC")
    return kst.strftime("%Y-%m-%d %H:%M:%S"), utc.strftime("%Y-%m-%dT%H:%M:%SZ")


@lru_cache(maxsize=1)
def get_spec() -> dict[str, Any]:
    return json.loads((DOCS_DIR / "preprocess_v2_spec.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_feature_scaler():
    return joblib.load(MODEL_DIR / "feature_scaler_v2.pkl")


@lru_cache(maxsize=1)
def get_target_scaler():
    return joblib.load(MODEL_DIR / "target_scaler_v2.pkl")


@lru_cache(maxsize=1)
def get_model_bundle():
    model, _, spec = load_model(torch.device("cpu"))
    return model, spec


@lru_cache(maxsize=1)
def get_train_hist_sorted() -> pd.DataFrame:
    train_df, _ = load_train_holdout_frames()
    return train_df.sort_values("시간").reset_index(drop=True)


def _extract_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    cur: Any = data
    # HRFCO 기본 응답: {"content":[...]}
    if isinstance(cur, dict) and isinstance(cur.get("content"), list):
        return [x for x in cur["content"] if isinstance(x, dict)]
    for k in ("response", "body", "items", "item"):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
    if isinstance(cur, list):
        return [x for x in cur if isinstance(x, dict)]
    if isinstance(cur, dict):
        return [cur]
    return []


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(str(v).replace(",", "").strip())
        if math.isfinite(x):
            return x
        return None
    except Exception:
        return None


def _parse_time_any(v: Any, year_hint: int | None = None) -> pd.Timestamp | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    for fmt in ("%Y%m%d%H%M", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt), tz="Asia/Seoul")
        except Exception:
            pass
    # 예: "12-31 01시", "12-31 01"
    m = re.search(r"(?P<mo>\d{1,2})-(?P<dd>\d{1,2})\s+(?P<hh>\d{1,2})(?::(?P<mm>\d{1,2}))?", s)
    if m and year_hint is not None:
        mo = int(m.group("mo"))
        dd = int(m.group("dd"))
        hh = int(m.group("hh"))
        mm = int(m.group("mm") or 0)
        try:
            return pd.Timestamp(datetime(year_hint, mo, dd, hh, mm), tz="Asia/Seoul")
        except Exception:
            pass
    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t):
        return None
    if t.tzinfo is None:
        t = t.tz_localize("Asia/Seoul")
    return t


def _build_url(url: str, params: dict[str, Any]) -> str:
    parts: list[str] = []
    for k, v in params.items():
        if v is None:
            continue
        kk = quote_plus(str(k))
        if str(k).lower() == "servicekey":
            vv = str(v)
        else:
            vv = quote_plus(str(v))
        parts.append(f"{kk}={vv}")
    q = "&".join(parts)
    return f"{url}?{q}"


def http_get_items(url: str, params: dict[str, Any], timeout: int = 12) -> list[dict[str, Any]]:
    full = _build_url(url, params)
    return http_get_items_by_url(full, timeout=timeout)


def http_get_items_by_url(full_url: str, timeout: int = 12) -> list[dict[str, Any]]:
    with urlopen(full_url, timeout=timeout) as r:
        body = r.read().decode("utf-8", errors="ignore").strip()
    # JSON 응답 우선
    try:
        data = json.loads(body)
        return _extract_items(data)
    except Exception:
        pass
    # XML 응답 fallback
    try:
        root = ET.fromstring(body)
        items: list[dict[str, Any]] = []
        for it in root.findall(".//item"):
            rec: dict[str, Any] = {}
            for c in list(it):
                rec[c.tag.lower()] = c.text
            if rec:
                items.append(rec)
        return items
    except Exception:
        return []


def fetch_kwater_latest(env: dict[str, str], bucket_kst: pd.Timestamp) -> tuple[dict[str, float | None], pd.Timestamp | None]:
    key_raw = env["DATA_GO_KR_SERVICE_KEY"]
    key_dec = decode_service_key(key_raw)
    dam_code = env["KWATER_DAM_CODE_YEOJU"]
    urls = [
        env.get("KWATER_API_URL", "").strip(),
        "http://apis.data.go.kr/B500001/dam/sluicePresentCondition/hourlist",
        "https://apis.data.go.kr/B500001/dam/sluicePresentCondition/hourlist",
        "https://apis.data.go.kr/B500001/dam/sluicePresentCondition/minlist",
        "http://apis.data.go.kr/B500001/dam/sluicePresentCondition/minlist",
        "https://apis.data.go.kr/B500001/waterinfoDam/getWaterinfoDam10m",
        "http://apis.data.go.kr/B500001/waterinfoDam/getWaterinfoDam10m",
        "https://apis.data.go.kr/B500001/dam/getWaterinfoDam10m",
    ]
    urls = [u for u in urls if u]

    key_candidates = [key_dec, key_raw]
    day = bucket_kst.strftime("%Y-%m-%d")
    params_candidates = []
    for key in key_candidates:
        params_candidates += [
            {"serviceKey": key, "pageNo": 1, "numOfRows": 300, "resultType": "json", "dmobscd": dam_code},
            {"serviceKey": key, "pageNo": 1, "numOfRows": 300, "resultType": "json", "damcd": dam_code},
            {"serviceKey": key, "pageNo": 1, "numOfRows": 300, "_type": "json", "damcode": dam_code, "stdt": day, "eddt": day},
            {"ServiceKey": key, "pageNo": 1, "numOfRows": 300, "resultType": "json", "dmobscd": dam_code},
            {"ServiceKey": key, "pageNo": 1, "numOfRows": 300, "resultType": "json", "damcd": dam_code},
            {"ServiceKey": key, "pageNo": 1, "numOfRows": 300, "_type": "json", "damcode": dam_code, "stdt": day, "eddt": day},
        ]

    best: dict[str, float | None] | None = None
    best_t: pd.Timestamp | None = None
    for u in urls:
        for p in params_candidates:
            try:
                items = http_get_items(u, p)
            except Exception:
                continue
            if not items:
                continue
            for it in items:
                low = {str(k).lower(): v for k, v in it.items()}
                t = None
                for tk in ("obsymdhm", "ymdhm", "obstm", "obsdt", "obsdate", "checkymdhm", "base_time", "obsrdt"):
                    if tk in low:
                        t = _parse_time_any(low[tk], year_hint=bucket_kst.year)
                        if t is not None:
                            break
                if t is None:
                    continue
                if t > bucket_kst:
                    continue
                row = {
                    "kwater_댐수위_m": _safe_float(
                        low.get("dmwlv") or low.get("dmwl") or low.get("waterlevel") or low.get("wl") or low.get("lowlevel")
                    ),
                    "kwater_강우량_mm": _safe_float(low.get("rf") or low.get("rainfall") or low.get("rnfal") or low.get("rfall")),
                    "kwater_유입량_m3s": _safe_float(
                        low.get("inf") or low.get("inflow") or low.get("inflowq") or low.get("inqty") or low.get("inflowqy")
                    ),
                    "kwater_총방류량_m3s": _safe_float(
                        low.get("tototf")
                        or low.get("outflow")
                        or low.get("tototfq")
                        or low.get("tototfqty")
                        or low.get("totdcwtrqy")
                    ),
                    "kwater_저수량_백만m3": _safe_float(low.get("stg") or low.get("storage") or low.get("rsvwtqty")),
                    "kwater_저수율_pct": _safe_float(low.get("rsvwt") or low.get("rate") or low.get("rsvwtrt")),
                }
                if best_t is None or t > best_t:
                    best_t = t
                    best = row
    if best is None:
        raise RuntimeError("K-water API 최신값 파싱 실패 (URL/파라미터 확인 필요)")
    return best, best_t


def fetch_hrfc_latest(
    env: dict[str, str], bucket_kst: pd.Timestamp, wl_cols: list[str]
) -> tuple[dict[str, float | None], pd.Timestamp | None]:
    """
    HRFCO API는 서비스별 응답 키 차이가 크므로, URL 오버라이드 가능하게 구성.
    기본 URL 실패 시 호출 결과를 경고하고 빈 dict를 반환(직전값 ffill로 보완).
    """
    key = env["HRFCO_SERVICE_KEY"]
    if env.get("HRFCO_WLOBSCD_LIST"):
        station_ids = [x.strip() for x in env["HRFCO_WLOBSCD_LIST"].split(",") if x.strip()]
    else:
        station_ids = [c.split("_")[1] for c in wl_cols if c.startswith("hrfc_") and c.endswith("_수위_m")]

    override = env.get("HRFCO_API_URL", "").strip()

    out: dict[str, float | None] = {}

    # 1) 우선 전체 최신값 일괄 조회 (속도 개선)
    bulk_items: list[dict[str, Any]] = []
    bulk_urls: list[str]
    if override and "{sid}" not in override:
        if "{key}" in override:
            bulk_urls = [override.format(key=key)]
        else:
            base = override.rstrip("/")
            bulk_urls = [
                f"{base}/{key}/waterlevel/list/10M.json",
                f"{base}/{key}/waterlevel/list/10M.xml",
            ]
    elif override and "{sid}" in override:
        bulk_urls = []
    else:
        bulk_urls = [
            f"https://api.hrfco.go.kr/{key}/waterlevel/list/10M.json",
            f"http://api.hrfco.go.kr/{key}/waterlevel/list/10M.json",
        ]
    for u in bulk_urls:
        try:
            bulk_items = http_get_items_by_url(u)
        except Exception:
            bulk_items = []
        if bulk_items:
            break

    bulk_best: dict[str, float] = {}
    if bulk_items:
        for it in bulk_items:
            low = {str(k).lower(): v for k, v in it.items()}
            sid0 = str(low.get("wlobscd", "")).strip()
            if not sid0:
                continue
            t = _parse_time_any(low.get("ymdhm") or low.get("obsymdhm") or low.get("tm"))
            if t is None or t > bucket_kst:
                continue
            v = _safe_float(low.get("wl") or low.get("waterlevel") or low.get("swl") or low.get("수위_m"))
            if v is None:
                continue
            if sid0 not in bulk_best:
                bulk_best[sid0] = v

    latest_any: pd.Timestamp | None = None
    for sid in station_ids:
        col = f"hrfc_{sid}_수위_m"
        if sid in bulk_best:
            out[col] = bulk_best[sid]
            continue
        val: float | None = None
        if override:
            if "{key}" in override or "{sid}" in override:
                urls = [override.format(key=key, sid=sid)]
            else:
                base = override.rstrip("/")
                urls = [f"{base}/{key}/waterlevel/list/10M/{sid}.json"]
        else:
            urls = [
                f"https://api.hrfco.go.kr/{key}/waterlevel/list/10M/{sid}.json",
                f"http://api.hrfco.go.kr/{key}/waterlevel/list/10M/{sid}.json",
            ]
        for u in urls:
            try:
                items = http_get_items_by_url(u)
            except Exception:
                continue
            best_t: pd.Timestamp | None = None
            best_v: float | None = None
            for it in items:
                low = {str(k).lower(): v for k, v in it.items()}
                t = None
                for tk in ("ymdhm", "obsymdhm", "obstm", "obsdt", "tm"):
                    if tk in low:
                        t = _parse_time_any(low[tk])
                        if t is not None:
                            break
                if t is None or t > bucket_kst:
                    continue
                v = _safe_float(low.get("wl") or low.get("waterlevel") or low.get("swl") or low.get("수위_m"))
                if v is None:
                    continue
                if best_t is None or t > best_t:
                    best_t, best_v = t, v
            if best_v is not None:
                val = best_v
                if latest_any is None or (best_t is not None and best_t > latest_any):
                    latest_any = best_t
                break
        out[col] = val
    return out, latest_any


def _kwater_urls_and_params(env: dict[str, str], bucket_kst: pd.Timestamp) -> tuple[list[str], list[dict[str, Any]]]:
    key_raw = env["DATA_GO_KR_SERVICE_KEY"]
    key_dec = decode_service_key(key_raw)
    dam_code = env["KWATER_DAM_CODE_YEOJU"]
    urls = [
        env.get("KWATER_API_URL", "").strip(),
        "http://apis.data.go.kr/B500001/dam/sluicePresentCondition/hourlist",
        "https://apis.data.go.kr/B500001/dam/sluicePresentCondition/hourlist",
        "https://apis.data.go.kr/B500001/dam/sluicePresentCondition/minlist",
        "http://apis.data.go.kr/B500001/dam/sluicePresentCondition/minlist",
        "https://apis.data.go.kr/B500001/waterinfoDam/getWaterinfoDam10m",
        "http://apis.data.go.kr/B500001/waterinfoDam/getWaterinfoDam10m",
        "https://apis.data.go.kr/B500001/dam/getWaterinfoDam10m",
    ]
    urls = [u for u in urls if u]
    key_candidates = [key_dec, key_raw]
    day_start = (bucket_kst - pd.Timedelta(minutes=10 * (LOOKBACK_STEPS - 1))).strftime("%Y-%m-%d")
    day_end = bucket_kst.strftime("%Y-%m-%d")
    params_candidates: list[dict[str, Any]] = []
    for key in key_candidates:
        params_candidates += [
            {"serviceKey": key, "pageNo": 1, "numOfRows": 500, "resultType": "json", "dmobscd": dam_code},
            {"serviceKey": key, "pageNo": 1, "numOfRows": 500, "resultType": "json", "damcd": dam_code},
            {
                "serviceKey": key,
                "pageNo": 1,
                "numOfRows": 500,
                "_type": "json",
                "damcode": dam_code,
                "stdt": day_start,
                "eddt": day_end,
            },
            {"ServiceKey": key, "pageNo": 1, "numOfRows": 500, "resultType": "json", "dmobscd": dam_code},
            {"ServiceKey": key, "pageNo": 1, "numOfRows": 500, "resultType": "json", "damcd": dam_code},
            {
                "ServiceKey": key,
                "pageNo": 1,
                "numOfRows": 500,
                "_type": "json",
                "damcode": dam_code,
                "stdt": day_start,
                "eddt": day_end,
            },
        ]
    return urls, params_candidates


def _extract_kwater_row(it: dict[str, Any], year_hint: int) -> tuple[pd.Timestamp | None, dict[str, float | None]]:
    low = {str(k).lower(): v for k, v in it.items()}
    t = None
    for tk in ("obsymdhm", "ymdhm", "obstm", "obsdt", "obsdate", "checkymdhm", "base_time", "obsrdt"):
        if tk in low:
            t = _parse_time_any(low[tk], year_hint=year_hint)
            if t is not None:
                break
    row = {
        "kwater_댐수위_m": _safe_float(low.get("dmwlv") or low.get("dmwl") or low.get("waterlevel") or low.get("wl") or low.get("lowlevel")),
        "kwater_강우량_mm": _safe_float(low.get("rf") or low.get("rainfall") or low.get("rnfal") or low.get("rfall")),
        "kwater_유입량_m3s": _safe_float(low.get("inf") or low.get("inflow") or low.get("inflowq") or low.get("inqty") or low.get("inflowqy")),
        "kwater_총방류량_m3s": _safe_float(
            low.get("tototf") or low.get("outflow") or low.get("tototfq") or low.get("tototfqty") or low.get("totdcwtrqy")
        ),
        "kwater_저수량_백만m3": _safe_float(low.get("stg") or low.get("storage") or low.get("rsvwtqty")),
        "kwater_저수율_pct": _safe_float(low.get("rsvwt") or low.get("rate") or low.get("rsvwtrt")),
    }
    return t, row


def fetch_kwater_series(
    env: dict[str, str], bucket_kst: pd.Timestamp, lookback_steps: int = LOOKBACK_STEPS
) -> tuple[dict[pd.Timestamp, dict[str, float | None]], pd.Timestamp | None]:
    start_kst = bucket_kst - pd.Timedelta(minutes=10 * (lookback_steps - 1))
    urls, params_candidates = _kwater_urls_and_params(env, bucket_kst)
    series: dict[pd.Timestamp, dict[str, float | None]] = {}
    for u in urls:
        for p in params_candidates:
            try:
                items = http_get_items(u, p)
            except Exception:
                continue
            if not items:
                continue
            for it in items:
                t, row = _extract_kwater_row(it, bucket_kst.year)
                if t is None or t < start_kst or t > bucket_kst:
                    continue
                if t not in series:
                    series[t] = row
                else:
                    for k, v in row.items():
                        if series[t].get(k) is None and v is not None:
                            series[t][k] = v
    if not series:
        raise RuntimeError("K-water API 12시간 시계열 파싱 실패 (URL/파라미터 확인 필요)")
    return series, max(series.keys())


def fetch_hrfc_series(
    env: dict[str, str], bucket_kst: pd.Timestamp, wl_cols: list[str], lookback_steps: int = LOOKBACK_STEPS
) -> tuple[dict[pd.Timestamp, dict[str, float | None]], pd.Timestamp | None]:
    start_kst = bucket_kst - pd.Timedelta(minutes=10 * (lookback_steps - 1))
    key = env["HRFCO_SERVICE_KEY"]
    if env.get("HRFCO_WLOBSCD_LIST"):
        station_ids = [x.strip() for x in env["HRFCO_WLOBSCD_LIST"].split(",") if x.strip()]
    else:
        station_ids = [c.split("_")[1] for c in wl_cols if c.startswith("hrfc_") and c.endswith("_수위_m")]
    override = env.get("HRFCO_API_URL", "").strip()

    out: dict[pd.Timestamp, dict[str, float | None]] = {}
    latest_any: pd.Timestamp | None = None
    seen_station_ids: set[str] = set()

    bulk_items: list[dict[str, Any]] = []
    bulk_urls: list[str]
    if override and "{sid}" not in override:
        if "{key}" in override:
            bulk_urls = [override.format(key=key)]
        else:
            base = override.rstrip("/")
            bulk_urls = [
                f"{base}/{key}/waterlevel/list/10M.json",
                f"{base}/{key}/waterlevel/list/10M.xml",
            ]
    elif override and "{sid}" in override:
        bulk_urls = []
    else:
        bulk_urls = [
            f"https://api.hrfco.go.kr/{key}/waterlevel/list/10M.json",
            f"http://api.hrfco.go.kr/{key}/waterlevel/list/10M.json",
        ]
    for u in bulk_urls:
        try:
            bulk_items = http_get_items_by_url(u)
        except Exception:
            bulk_items = []
        if bulk_items:
            break

    for it in bulk_items:
        low = {str(k).lower(): v for k, v in it.items()}
        sid0 = str(low.get("wlobscd", "")).strip()
        if not sid0 or sid0 not in station_ids:
            continue
        t = _parse_time_any(low.get("ymdhm") or low.get("obsymdhm") or low.get("tm"), year_hint=bucket_kst.year)
        if t is None or t < start_kst or t > bucket_kst:
            continue
        v = _safe_float(low.get("wl") or low.get("waterlevel") or low.get("swl") or low.get("수위_m"))
        if v is None:
            continue
        col = f"hrfc_{sid0}_수위_m"
        row = out.setdefault(t, {})
        row[col] = v
        seen_station_ids.add(sid0)
        if latest_any is None or t > latest_any:
            latest_any = t

    for sid in station_ids:
        # bulk 응답에 없던 관측소는 개별 endpoint로 보완
        if sid in seen_station_ids:
            continue
        if override:
            if "{key}" in override or "{sid}" in override:
                urls = [override.format(key=key, sid=sid)]
            else:
                base = override.rstrip("/")
                urls = [f"{base}/{key}/waterlevel/list/10M/{sid}.json"]
        else:
            urls = [
                f"https://api.hrfco.go.kr/{key}/waterlevel/list/10M/{sid}.json",
                f"http://api.hrfco.go.kr/{key}/waterlevel/list/10M/{sid}.json",
            ]
        for u in urls:
            try:
                items = http_get_items_by_url(u)
            except Exception:
                continue
            hit = False
            for it in items:
                low = {str(k).lower(): v for k, v in it.items()}
                t = None
                for tk in ("ymdhm", "obsymdhm", "obstm", "obsdt", "tm"):
                    if tk in low:
                        t = _parse_time_any(low[tk], year_hint=bucket_kst.year)
                        if t is not None:
                            break
                if t is None or t < start_kst or t > bucket_kst:
                    continue
                v = _safe_float(low.get("wl") or low.get("waterlevel") or low.get("swl") or low.get("수위_m"))
                if v is None:
                    continue
                col = f"hrfc_{sid}_수위_m"
                row = out.setdefault(t, {})
                row[col] = v
                hit = True
                if latest_any is None or t > latest_any:
                    latest_any = t
            if hit:
                break
    return out, latest_any


def _recompute_row_features(tmp: pd.DataFrame, idx: int) -> None:
    wl_cols = [c for c in tmp.columns if c.startswith("hrfc_") and c.endswith("_수위_m") and "평균" not in c]
    tmp.at[idx, "hrfc_수위_m_평균"] = pd.to_numeric(tmp.loc[idx, wl_cols], errors="coerce").mean()
    tmp.at[idx, "hrfc_관측소수"] = int(pd.to_numeric(tmp.loc[idx, wl_cols], errors="coerce").notna().sum())

    ts = pd.Timestamp(tmp.at[idx, "시간"])
    hour = ts.hour + ts.minute / 60.0
    doy = ts.dayofyear
    tmp.at[idx, "time_sin_hour"] = np.sin(2 * np.pi * hour / 24)
    tmp.at[idx, "time_cos_hour"] = np.cos(2 * np.pi * hour / 24)
    tmp.at[idx, "time_sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    tmp.at[idx, "time_cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    for col in KEY_SERIES:
        if col not in tmp.columns:
            continue
        now_v = pd.to_numeric(tmp.at[idx, col], errors="coerce")
        prev_v = pd.to_numeric(tmp.at[idx - 1, col], errors="coerce") if idx > 0 else np.nan
        tmp.at[idx, f"{col}_diff1"] = now_v - prev_v if pd.notna(now_v) and pd.notna(prev_v) else np.nan
        for lag in (6, 18, 36, 72):
            li = idx - lag
            lv = pd.to_numeric(tmp.at[li, col], errors="coerce") if li >= 0 else np.nan
            tmp.at[idx, f"{col}_lag_{lag}"] = lv


def build_realtime_feature_window(
    df_scaled_hist: pd.DataFrame, env: dict[str, str], requested_bucket_kst: pd.Timestamp
) -> tuple[pd.DataFrame, pd.Timestamp, dict[str, Any]]:
    spec = get_spec()
    feat_cols: list[str] = spec["feature_columns"]
    bucket = requested_bucket_kst

    fx = get_feature_scaler()
    hist_scaled = df_scaled_hist.copy()
    hist_scaled[feat_cols] = hist_scaled[feat_cols].astype(float)

    ctx_len = max(TAIL_WINDOW, LOOKBACK_STEPS + 80)
    hist_tail_scaled = hist_scaled.iloc[-ctx_len:].copy()
    hist_raw = hist_tail_scaled.copy()
    hist_raw[feat_cols] = fx.inverse_transform(hist_tail_scaled[feat_cols].values)
    hist_raw = hist_raw.reset_index(drop=True)

    if len(hist_raw) <= LOOKBACK_STEPS:
        raise RuntimeError("실시간 윈도우 구성을 위한 히스토리 길이가 부족합니다.")

    timeline = list(pd.date_range(end=bucket, periods=LOOKBACK_STEPS, freq="10min", tz="Asia/Seoul"))
    start_idx = len(hist_raw) - LOOKBACK_STEPS
    time_to_idx = {ts: start_idx + i for i, ts in enumerate(timeline)}

    for i, ts in enumerate(timeline):
        idx = start_idx + i
        kst_str, utc_str = to_bucket_strings(ts)
        hist_raw.at[idx, "시간"] = kst_str
        hist_raw.at[idx, "bucket_start_kst"] = kst_str
        hist_raw.at[idx, "bucket_start_utc"] = utc_str

    warn_msgs: list[str] = []
    kw_t: pd.Timestamp | None = None
    hr_t: pd.Timestamp | None = None
    kw_points = 0
    hr_points = 0

    try:
        kw_series, kw_t = fetch_kwater_series(env, bucket, lookback_steps=LOOKBACK_STEPS)
    except Exception as e:
        kw_series = {}
        warn_msgs.append(f"K-water API fallback(prev used): {e}")
    for t, row in kw_series.items():
        idx = time_to_idx.get(t)
        if idx is None:
            continue
        for k, v in row.items():
            if k in hist_raw.columns and v is not None:
                hist_raw.at[idx, k] = v
                kw_points += 1
    if kw_series and kw_points == 0:
        warn_msgs.append("K-water API returned no in-range values; previous values retained")

    wl_cols = [c for c in feat_cols if c.startswith("hrfc_") and c.endswith("_수위_m") and "평균" not in c]
    try:
        hr_series, hr_t = fetch_hrfc_series(env, bucket, wl_cols, lookback_steps=LOOKBACK_STEPS)
    except Exception as e:
        hr_series = {}
        warn_msgs.append(f"HRFCO API fallback(prev used): {e}")
    for t, row in hr_series.items():
        idx = time_to_idx.get(t)
        if idx is None:
            continue
        for k, v in row.items():
            if k in hist_raw.columns and v is not None:
                hist_raw.at[idx, k] = v
                hr_points += 1
    if hr_series and hr_points == 0:
        warn_msgs.append("HRFCO API returned no in-range values; previous values retained")

    for idx in range(start_idx, len(hist_raw)):
        apply_realtime_cleaning(hist_raw, idx)
        _recompute_row_features(hist_raw, idx)

    window_raw = hist_raw.iloc[-LOOKBACK_STEPS:].copy().reset_index(drop=True)
    feat = window_raw[feat_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill()
    window_scaled = pd.DataFrame(fx.transform(feat.values), columns=feat_cols)
    window_scaled["시간"] = window_raw["시간"].values

    debug = {
        "bucket_start_kst": window_raw.iloc[-1]["시간"],
        "bucket_start_utc": to_bucket_strings(bucket)[1],
        "requested_bucket_kst": requested_bucket_kst.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_at_kst": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
        "kwater_latest_obs_kst": kw_t.strftime("%Y-%m-%d %H:%M:%S") if kw_t is not None else None,
        "hrfc_latest_obs_kst": hr_t.strftime("%Y-%m-%d %H:%M:%S") if hr_t is not None else None,
        "target_station_code": "1007639",
        "target_station_name": "여주보 상류",
        "target_variable": "수위(m)",
        "prediction_scope": "지점 수위 예측(지역 평균 아님)",
        "dam_level_usage": "댐 수위는 입력 피처로 사용, 직접 예측 대상은 아님",
        "kwater_source": "api_12h_or_prev",
        "hrfc_source": "api_12h_or_prev",
        "window_steps": LOOKBACK_STEPS,
        "window_hours": LOOKBACK_STEPS / 6,
        "kwater_points_applied": kw_points,
        "hrfc_points_applied": hr_points,
        "current_level_m": float(pd.to_numeric(window_raw.iloc[-1].get("hrfc_1007639_수위_m"), errors="coerce"))
        if "hrfc_1007639_수위_m" in window_raw.columns
        else None,
        "warnings": warn_msgs,
    }
    return window_scaled, bucket, debug


def apply_realtime_cleaning(df_raw: pd.DataFrame, new_idx: int) -> None:
    wl_cols = [c for c in df_raw.columns if "수위" in c or "댐수위" in c]
    rain_cols = [c for c in df_raw.columns if "강우" in c]

    # 수위류: IQR clip(k=1.5) 후 ffill
    for c in wl_cols:
        hist = pd.to_numeric(df_raw.loc[:new_idx, c], errors="coerce")
        q1, q3 = hist.dropna().quantile(0.25), hist.dropna().quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            v = df_raw.at[new_idx, c]
            if pd.notna(v):
                df_raw.at[new_idx, c] = min(max(float(v), float(lo)), float(hi))
        if pd.isna(df_raw.at[new_idx, c]) and new_idx > 0:
            df_raw.at[new_idx, c] = df_raw.at[new_idx - 1, c]

    # 강우: ffill only
    for c in rain_cols:
        if pd.isna(df_raw.at[new_idx, c]) and new_idx > 0:
            df_raw.at[new_idx, c] = df_raw.at[new_idx - 1, c]


def build_realtime_feature_row(
    df_scaled_hist: pd.DataFrame, env: dict[str, str], requested_bucket_kst: pd.Timestamp
) -> tuple[pd.DataFrame, pd.Timestamp, dict[str, Any]]:
    spec = get_spec()
    feat_cols: list[str] = spec["feature_columns"]
    bucket = requested_bucket_kst

    fx = get_feature_scaler()
    hist_scaled = df_scaled_hist.copy()
    hist_scaled[feat_cols] = hist_scaled[feat_cols].astype(float)
    hist_tail_scaled = hist_scaled.iloc[-TAIL_WINDOW:].copy()
    hist_raw = hist_tail_scaled.copy()
    hist_raw[feat_cols] = fx.inverse_transform(hist_tail_scaled[feat_cols].values)

    last = hist_raw.iloc[-1].copy()
    kst_str, utc_str = to_bucket_strings(bucket)
    last["시간"] = kst_str
    last["bucket_start_kst"] = kst_str
    last["bucket_start_utc"] = utc_str

    warn_msgs: list[str] = []
    kw_t: pd.Timestamp | None = None
    hr_t: pd.Timestamp | None = None
    try:
        kw, kw_t = fetch_kwater_latest(env, bucket)
    except Exception as e:
        kw = {}
        warn_msgs.append(f"K-water API fallback(prev used): {e}")
    for k, v in kw.items():
        if k in last.index and v is not None:
            last[k] = v

    wl_cols = [c for c in feat_cols if c.startswith("hrfc_") and c.endswith("_수위_m") and "평균" not in c]
    try:
        hr, hr_t = fetch_hrfc_latest(env, bucket, wl_cols)
    except Exception as e:
        hr = {}
        warn_msgs.append(f"HRFCO API fallback(prev used): {e}")
    if hr and all(v is None for v in hr.values()):
        warn_msgs.append("HRFCO API returned no usable values; previous values retained")
    for k, v in hr.items():
        if k in last.index and v is not None:
            last[k] = v

    tmp = pd.concat([hist_raw.iloc[-80:].copy(), pd.DataFrame([last])], ignore_index=True)
    new_i = len(tmp) - 1
    apply_realtime_cleaning(tmp, new_i)

    # 집계
    wl_cols2 = [c for c in tmp.columns if c.startswith("hrfc_") and c.endswith("_수위_m") and "평균" not in c]
    tmp.at[new_i, "hrfc_수위_m_평균"] = pd.to_numeric(tmp.loc[new_i, wl_cols2], errors="coerce").mean()
    tmp.at[new_i, "hrfc_관측소수"] = int(pd.to_numeric(tmp.loc[new_i, wl_cols2], errors="coerce").notna().sum())

    # 시간 주기
    ts = pd.Timestamp(tmp.at[new_i, "시간"])
    hour = ts.hour + ts.minute / 60.0
    doy = ts.dayofyear
    tmp.at[new_i, "time_sin_hour"] = np.sin(2 * np.pi * hour / 24)
    tmp.at[new_i, "time_cos_hour"] = np.cos(2 * np.pi * hour / 24)
    tmp.at[new_i, "time_sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    tmp.at[new_i, "time_cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    # diff/lag
    for col in KEY_SERIES:
        if col not in tmp.columns:
            continue
        now_v = pd.to_numeric(tmp.at[new_i, col], errors="coerce")
        prev_v = pd.to_numeric(tmp.at[new_i - 1, col], errors="coerce") if new_i > 0 else np.nan
        tmp.at[new_i, f"{col}_diff1"] = now_v - prev_v if pd.notna(now_v) and pd.notna(prev_v) else np.nan
        for lag in (6, 18, 36, 72):
            li = new_i - lag
            lv = pd.to_numeric(tmp.at[li, col], errors="coerce") if li >= 0 else np.nan
            tmp.at[new_i, f"{col}_lag_{lag}"] = lv

    row_raw = tmp.iloc[[new_i]].copy()
    row_raw = row_raw.ffill(axis=1).ffill().bfill()
    row_feat = row_raw[feat_cols].copy()
    row_scaled = pd.DataFrame(fx.transform(row_feat[feat_cols]), columns=feat_cols)
    row_scaled["시간"] = kst_str

    debug = {
        "bucket_start_kst": kst_str,
        "bucket_start_utc": utc_str,
        "requested_bucket_kst": requested_bucket_kst.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_at_kst": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
        "kwater_latest_obs_kst": kw_t.strftime("%Y-%m-%d %H:%M:%S") if kw_t is not None else None,
        "hrfc_latest_obs_kst": hr_t.strftime("%Y-%m-%d %H:%M:%S") if hr_t is not None else None,
        "target_station_code": "1007639",
        "target_station_name": "여주보 상류",
        "target_variable": "수위(m)",
        "prediction_scope": "지점 수위 예측(지역 평균 아님)",
        "dam_level_usage": "댐 수위는 입력 피처로 사용, 직접 예측 대상은 아님",
        "kwater_source": "api_or_prev",
        "hrfc_source": "api_or_prev",
        "current_level_m": float(pd.to_numeric(tmp.at[new_i, "hrfc_1007639_수위_m"], errors="coerce"))
        if "hrfc_1007639_수위_m" in tmp.columns
        else None,
        "warnings": warn_msgs,
    }
    return row_scaled, bucket, debug


def infer_once(window_df: pd.DataFrame) -> dict[str, float]:
    model, spec = get_model_bundle()
    scaler_y = get_target_scaler()
    end_idx = len(window_df)
    nodes, time_g, _ = build_node_tensor_window(window_df, end_idx, spec)
    nt = torch.from_numpy(nodes).unsqueeze(0)
    tt = torch.from_numpy(time_g).unsqueeze(0)
    with torch.no_grad():
        pred_s = model(nt, tt, spec.dam_channels).cpu().numpy().ravel()
    out: dict[str, float] = {}
    for i, h in enumerate(spec.horizons):
        out[f"h{h}_pred_m"] = float(scaler_y.inverse_transform(np.array([[pred_s[i]]])).ravel()[0])
    return out


def run(skip_api: bool = False) -> None:
    env = load_env(ENV_PATH)
    train_df = get_train_hist_sorted().copy()
    spec = get_spec()
    feat_cols = spec["feature_columns"]
    requested_bucket = floor_10m_kst()

    hist_scaled = train_df
    if len(hist_scaled) < LOOKBACK_STEPS:
        raise RuntimeError(f"학습 CSV 길이가 {LOOKBACK_STEPS} 미만이라 윈도우 구성 불가")

    if skip_api:
        print("[warn] --skip-api: API 호출 없이 학습 CSV 마지막 행으로 테스트")
        window = hist_scaled.iloc[-LOOKBACK_STEPS:][feat_cols + ["시간"]].copy().reset_index(drop=True)
        bucket = requested_bucket
        cur = pd.to_numeric(hist_scaled.iloc[-1].get("target_수위_m"), errors="coerce")
        debug = {
            "bucket_start_kst": str(bucket),
            "requested_bucket_kst": requested_bucket.strftime("%Y-%m-%d %H:%M:%S"),
            "generated_at_kst": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S"),
            "target_station_code": "1007639",
            "target_station_name": "여주보 상류",
            "target_variable": "수위(m)",
            "prediction_scope": "지점 수위 예측(지역 평균 아님)",
            "dam_level_usage": "댐 수위는 입력 피처로 사용, 직접 예측 대상은 아님",
            "mode": "skip_api",
            "current_level_m": float(cur) if pd.notna(cur) else None,
            "warnings": [],
        }
    else:
        window, bucket, debug = build_realtime_feature_window(hist_scaled, env, requested_bucket)

    preds = infer_once(window)

    out = {
        "bucket_start_kst": debug["bucket_start_kst"],
        "predictions_m": preds,
        "meta": debug,
    }
    out_path = DATA_DIR / "realtime_latest_prediction.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("Realtime inference OK")
    print(f"bucket_start_kst: {out['bucket_start_kst']}")
    for w in out["meta"].get("warnings", []):
        print(f"[warn] {w}")
    for k, v in preds.items():
        print(f"{k}: {v:.4f} m")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-api", action="store_true", help="API 없이 파이프라인/추론 구조만 점검")
    args = ap.parse_args()
    run(skip_api=args.skip_api)
