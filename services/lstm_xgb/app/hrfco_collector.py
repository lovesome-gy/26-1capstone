"""
services/lstm_xgb/app/hrfco_collector.py

한강홍수통제소 API 실시간 데이터 수집 및 AI 서버 전송
- API로초기세팅파일만들기2.py (초기 100행 CSV 생성)
- 한강홍수통제소_API로_데이터받아서_서버에보내는예시.py (실시간 수집 → 서버 전송)
두 파일을 통합하여 컨테이너 내에서 자동 실행

환경변수:
    HRFCO_SERVICE_KEY  - 한강홍수통제소 API 인증키 (필수)
    LSTM_SERVER_URL    - AI 서버 주소 (기본: http://localhost:8000)
    COLLECT_INTERVAL   - 수집 주기 초 (기본: 600 = 10분)

사용:
    python -m app.hrfco_collector           # 단발 실행
    python -m app.hrfco_collector --loop    # 10분 주기 반복
    python -m app.hrfco_collector --init    # 초기 100행 CSV 생성만
"""

import os
import sys
import time
import logging
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hrfco_collector")

SERVICE_KEY    = os.getenv("HRFCO_SERVICE_KEY", "KEY")
LSTM_SERVER_URL = os.getenv("LSTM_SERVER_URL", "http://localhost:8000/predict")
COLLECT_INTERVAL = int(os.getenv("COLLECT_INTERVAL", "600"))
OUTPUT_CSV     = os.path.join(os.path.dirname(__file__), "..", "models", "여주보_예측_필요_최근100행_데이터셋.csv")

OBS_CODES = {
    "여주보_상류":    "1007639",
    "여주보_하류":    "1007641",
    "여주대교_수위":  "1007635",
    "여주대교_강수":  "10074030",
    "주암리_강수":   "10074100",
    "문막교":       "1006690",
    "수안보_강수":   "10044020",
    "충주댐":       "1003110",
    "충주조정지댐":  "1003611",
}


# ── HRFCO API 호출 ─────────────────────────────────────────────────────────────
def fetch_range(hydro_type, obs_code, sdt, edt):
    url = f"https://api.hrfco.go.kr/{SERVICE_KEY}/{hydro_type}/list/10M/{obs_code}/{sdt}/{edt}.json"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            j = r.json()
            if "content" in j and isinstance(j["content"], list):
                return j["content"]
    except Exception as e:
        logger.warning("fetch_range 실패 [%s %s]: %s", hydro_type, obs_code, e)
    return []


def fetch_latest(hydro_type, obs_code):
    url = f"https://api.hrfco.go.kr/{SERVICE_KEY}/{hydro_type}/list/10M/{obs_code}.json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            j = r.json()
            if "content" in j and isinstance(j["content"], list) and j["content"]:
                return j["content"][0]
    except Exception as e:
        logger.warning("fetch_latest 실패 [%s %s]: %s", hydro_type, obs_code, e)
    return None


def safe_float(data, key, field_name=""):
    if not data:
        return 0.0
    clean = {str(k).lower().strip(): str(v).strip() for k, v in data.items()}
    k = str(key).lower().strip()
    if k not in clean or clean[k] in ("", "-", "none"):
        return 0.0
    try:
        return float(clean[k])
    except ValueError:
        logger.debug("수치 변환 실패: %s=%s", field_name, clean.get(k))
        return 0.0


# ── 초기 CSV 생성 (API로초기세팅파일만들기2.py) ──────────────────────────────
def build_init_csv():
    """최근 3일치 100행 CSV 생성 (초기 히스토리)"""
    now = datetime.now()
    rm = (now.minute // 10) * 10
    base_end = now.replace(minute=rm, second=0, microsecond=0)
    base_start = base_end - timedelta(days=3)
    sdt = base_start.strftime("%Y%m%d%H%M")
    edt = base_end.strftime("%Y%m%d%H%M")
    logger.info("초기 CSV 생성: %s ~ %s", sdt, edt)

    time_range = pd.date_range(start=base_start, end=base_end, freq="10min")
    final_df = pd.DataFrame({"ymdhm": time_range.strftime("%Y%m%d%H%M")})

    station_configs = [
        {"name":"여주보_상류","type":"waterlevel","mappings":{"wl":"여주보(상류)_수위_수위(m)","fw":"여주보(상류)_수위_유량(m³/s)"}},
        {"name":"여주보_하류","type":"waterlevel","mappings":{"wl":"여주보(하류)_수위_수위(m)","fw":"여주보(하류)_수위_유량(m³/s)"}},
        {"name":"여주대교_수위","type":"waterlevel","mappings":{"wl":"여주시(여주대교)_수위_수위(m)","fw":"여주시(여주대교)_수위_유량(m³/s)"}},
        {"name":"문막교","type":"waterlevel","mappings":{"wl":"원주시(문막교)_수위_수위(m)","fw":"원주시(문막교)_수위_유량(m³/s)"}},
        {"name":"여주대교_강수","type":"rainfall","mappings":{"rf":"여주시(여주대교)_강수량_강수량(mm)"}},
        {"name":"주암리_강수","type":"rainfall","mappings":{"rf":"여주시(주암리)_강수량_강수량(mm)"}},
        {"name":"수안보_강수","type":"rainfall","mappings":{"rf":"충주시(수안보면사무소)_강수량_강수량(mm)"}},
        {"name":"충주댐","type":"dam","mappings":{"swl":"충주댐_댐_현재수위(EL.m)","inf":"충주댐_댐_유입량(m³/s)","tototf":"충주댐_댐_방류량(m³/s)"}},
        {"name":"충주조정지댐","type":"dam","mappings":{"swl":"충주조정지댐_댐_현재수위(EL.m)","inf":"충주조정지댐_댐_유입량(m³/s)","tototf":"충주조정지댐_댐_방류량(m³/s)"}},
    ]
    for cfg in station_configs:
        raw = fetch_range(cfg["type"], OBS_CODES[cfg["name"]], sdt, edt)
        if raw:
            df = pd.DataFrame(raw)
            df.columns = [str(c).lower().strip() for c in df.columns]
            for src in cfg["mappings"]:
                if src in df.columns:
                    df[src] = pd.to_numeric(df[src].astype(str).str.strip(), errors="coerce")
            df_sub = df[["ymdhm"] + list(cfg["mappings"])].rename(columns=cfg["mappings"])
            final_df = pd.merge(final_df, df_sub, on="ymdhm", how="left")
            logger.info("  %s ✅", cfg["name"])
        else:
            for c in cfg["mappings"].values():
                final_df[c] = None
            logger.warning("  %s ⚠️ 데이터 없음", cfg["name"])

        # 보조 필드
        copy_map = {
            "여주보_상류":    ("여주보(상류)_수위_수위(m)",          "여주보(상류)_수위_해발수위(El.m)"),
            "여주보_하류":    ("여주보(하류)_수위_수위(m)",          "여주보(하류)_수위_해발수위(El.m)"),
            "여주대교_수위":  ("여주시(여주대교)_수위_수위(m)",       "여주시(여주대교)_수위_해발수위(El.m)"),
            "문막교":        ("원주시(문막교)_수위_수위(m)",          "원주시(문막교)_수위_해발수위(El.m)"),
            "여주대교_강수":  ("여주시(여주대교)_강수량_강수량(mm)",   "여주시(여주대교)_강수량_누적강수량(mm)"),
            "주암리_강수":   ("여주시(주암리)_강수량_강수량(mm)",      "여주시(주암리)_강수량_누적강수량(mm)"),
            "수안보_강수":   ("충주시(수안보면사무소)_강수량_강수량(mm)","충주시(수안보면사무소)_강수량_누적강수량(mm)"),
        }
        if cfg["name"] in copy_map:
            src_c, dst_c = copy_map[cfg["name"]]
            if src_c in final_df.columns:
                final_df[dst_c] = final_df[src_c]

    final_df = final_df.sort_values("ymdhm")
    final_df["ymdhm"] = pd.to_datetime(final_df["ymdhm"], format="%Y%m%d%H%M")
    final_df["시간"] = final_df["ymdhm"].dt.strftime("%Y-%m-%d %H:%M")
    final_df = final_df.tail(100)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logger.info("✅ 초기 CSV 저장: %s (%d행)", OUTPUT_CSV, len(final_df))


# ── 실시간 수집 → AI 서버 전송 (한강홍수통제소_API로_데이터받아서_서버에보내는예시.py) ──
def collect_and_send():
    """최신 1건 수집 → AI 서버 전송"""
    logger.info("실시간 10분 데이터 수집 시작...")

    raw_map = {
        "여주보_상류":   fetch_latest("waterlevel", OBS_CODES["여주보_상류"]),
        "여주보_하류":   fetch_latest("waterlevel", OBS_CODES["여주보_하류"]),
        "여주대교_수위": fetch_latest("waterlevel", OBS_CODES["여주대교_수위"]),
        "여주대교_강수": fetch_latest("rainfall",   OBS_CODES["여주대교_강수"]),
        "주암리_강수":  fetch_latest("rainfall",   OBS_CODES["주암리_강수"]),
        "문막교":      fetch_latest("waterlevel", OBS_CODES["문막교"]),
        "수안보_강수":  fetch_latest("rainfall",   OBS_CODES["수안보_강수"]),
        "충주댐":      fetch_latest("dam",        OBS_CODES["충주댐"]),
        "충주조정지댐": fetch_latest("dam",        OBS_CODES["충주조정지댐"]),
    }

    # 관측 시각 파싱
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    wl_data = raw_map.get("여주대교_수위")
    if wl_data and "ymdhm" in wl_data:
        t = str(wl_data["ymdhm"]).strip()
        if len(t) >= 12:
            current_time = f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[8:10]}:{t[10:12]}"

    payload = {
        "시간": current_time,
        "여주보_상류_수위_수위_m":              safe_float(raw_map["여주보_상류"], "wl"),
        "여주보_상류_수위_유량_m3_s":            safe_float(raw_map["여주보_상류"], "fw"),
        "여주보_상류_수위_해발수위_El_m":         safe_float(raw_map["여주보_상류"], "wl"),
        "여주보_하류_수위_수위_m":              safe_float(raw_map["여주보_하류"], "wl"),
        "여주보_하류_수위_유량_m3_s":            safe_float(raw_map["여주보_하류"], "fw"),
        "여주보_하류_수위_해발수위_El_m":         safe_float(raw_map["여주보_하류"], "wl"),
        "여주시_여주대교_강수량_강수량_mm":        safe_float(raw_map["여주대교_강수"], "rf"),
        "여주시_여주대교_강수량_누적강수량_mm":     safe_float(raw_map["여주대교_강수"], "rf"),
        "여주시_여주대교_수위_수위_m":            safe_float(raw_map["여주대교_수위"], "wl"),
        "여주시_여주대교_수위_유량_m3_s":          safe_float(raw_map["여주대교_수위"], "fw"),
        "여주시_여주대교_수위_해발수위_El_m":       safe_float(raw_map["여주대교_수위"], "wl"),
        "여주시_주암리_강수량_강수량_mm":          safe_float(raw_map["주암리_강수"], "rf"),
        "여주시_주암리_강수량_누적강수량_mm":       safe_float(raw_map["주암리_강수"], "rf"),
        "원주시_문막교_수위_수위_m":             safe_float(raw_map["문막교"], "wl"),
        "원주시_문막교_수위_유량_m3_s":           safe_float(raw_map["문막교"], "fw"),
        "원주시_문막교_수위_해발수위_El_m":        safe_float(raw_map["문막교"], "wl"),
        "충주댐_댐_현재수위_EL_m":              safe_float(raw_map["충주댐"], "swl"),
        "충주댐_댐_유입량_m3_s":                safe_float(raw_map["충주댐"], "inf"),
        "충주댐_댐_방류량_m3_s":                safe_float(raw_map["충주댐"], "tototf"),
        "충주시_수안보면사무소_강수량_강수량_mm":    safe_float(raw_map["수안보_강수"], "rf"),
        "충주시_수안보면사무소_강수량_누적강수량_mm": safe_float(raw_map["수안보_강수"], "rf"),
        "충주조정지댐_댐_현재수위_EL_m":          safe_float(raw_map["충주조정지댐"], "swl"),
        "충주조정지댐_댐_유입량_m3_s":            safe_float(raw_map["충주조정지댐"], "inf"),
        "충주조정지댐_댐_방류량_m3_s":            safe_float(raw_map["충주조정지댐"], "tototf"),
    }

    logger.info("AI 서버로 전송 중: %s", LSTM_SERVER_URL)
    try:
        r = requests.post(LSTM_SERVER_URL, json=payload, timeout=10)
        if r.status_code == 200:
            result = r.json()
            logger.info("✅ 예측 성공: %s", result.get("predictions"))
            return result
        else:
            logger.error("❌ 서버 응답 오류 %d: %s", r.status_code, r.text[:200])
    except Exception as e:
        logger.error("⚠️ 서버 연결 실패: %s", e)
    return None


# ── CLI 진입점 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRFCO 데이터 수집기")
    parser.add_argument("--init", action="store_true", help="초기 CSV 100행 생성")
    parser.add_argument("--loop", action="store_true", help="10분 주기 반복 수집")
    args = parser.parse_args()

    if SERVICE_KEY == "KEY":
        logger.error("HRFCO_SERVICE_KEY 환경변수를 설정하세요")
        sys.exit(1)

    if args.init:
        build_init_csv()
    elif args.loop:
        logger.info("10분 주기 반복 수집 시작 (Ctrl+C로 중지)")
        while True:
            collect_and_send()
            logger.info("다음 수집까지 %d초 대기...", COLLECT_INTERVAL)
            time.sleep(COLLECT_INTERVAL)
    else:
        collect_and_send()
