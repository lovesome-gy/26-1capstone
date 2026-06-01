"""
services/data_collector/app/main.py
실시간 데이터 수집 서비스 (신가연 담당)

역할:
  - 10분마다 HRFCO(한강홍수통제소) + K-water API 호출
  - 수집 데이터 → PostgreSQL water_level_raw 테이블 저장
  - LLM 보고서 통계 기반 데이터 공급

환경변수:
  HRFCO_SERVICE_KEY      한강홍수통제소 인증키
  DATA_GO_KR_SERVICE_KEY 공공데이터포털 K-water 인증키
  DATABASE_URL           PostgreSQL 연결 URL
  COLLECT_INTERVAL       수집 주기(초, 기본 600)
  SKIP_API               true면 API 호출 건너뜀 (로컬 테스트용)
"""

import os
import time
import logging
import psycopg2
import requests
from datetime import datetime, timedelta, timezone

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("data_collector")

# ── 설정 ─────────────────────────────────────────────────────────────────────
HRFCO_KEY      = os.getenv("HRFCO_SERVICE_KEY", "")
KWATER_KEY     = os.getenv("DATA_GO_KR_SERVICE_KEY", "")
DATABASE_URL   = os.getenv("DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")
COLLECT_INTERVAL = int(os.getenv("COLLECT_INTERVAL", "600"))  # 10분
SKIP_API       = os.getenv("SKIP_API", "true").lower() == "true"
STATION_ID     = "3008680"  # 여주보
KST            = timezone(timedelta(hours=9))

# 여주보 관측소 코드
OBS_CODES = {
    "여주보_상류":   "1007639",   # 수위 메인
    "여주보_하류":   "1007641",
    "여주대교_수위": "1007635",
    "여주대교_강수": "10074030",
    "주암리_강수":  "10074100",
    "문막교":      "1006690",
    "수안보_강수":  "10044020",
    "충주댐":      "1003110",
    "충주조정지댐": "1003611",
}


# ── HRFCO API 호출 ─────────────────────────────────────────────────────────────
def fetch_hrfco_latest(hydro_type: str, obs_code: str) -> dict | None:
    """한강홍수통제소 최신 10분 데이터 1건 호출"""
    url = f"https://api.hrfco.go.kr/{HRFCO_KEY}/{hydro_type}/list/10M/{obs_code}.json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        if "content" in j and j["content"]:
            return j["content"][0]
    except Exception as e:
        logger.warning("HRFCO 호출 실패 [%s %s]: %s", hydro_type, obs_code, e)
    return None


def safe_float(data: dict | None, key: str) -> float | None:
    """응답에서 float 추출, 실패 시 None"""
    if not data:
        return None
    val = str(data.get(key, "")).strip()
    if val in ("", "-", "none", "None"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


# ── 데이터 수집 ───────────────────────────────────────────────────────────────
def collect_once() -> dict | None:
    """
    HRFCO API에서 여주보 데이터 수집
    반환: {time, water_level_m, flow_rate_cms, rainfall_mm}
    """
    if SKIP_API:
        logger.info("SKIP_API=true — API 호출 건너뜀")
        return None

    if not HRFCO_KEY or HRFCO_KEY == "KEY":
        logger.warning("HRFCO_SERVICE_KEY 미설정 — 수집 건너뜀")
        return None

    # 여주보 상류 수위 (메인 관측소)
    wl_data = fetch_hrfco_latest("waterlevel", OBS_CODES["여주보_상류"])
    rf_data = fetch_hrfco_latest("rainfall",   OBS_CODES["여주대교_강수"])

    water_level = safe_float(wl_data, "wl")
    flow_rate   = safe_float(wl_data, "fw")
    rainfall    = safe_float(rf_data, "rf")

    if water_level is None:
        logger.warning("수위 데이터 없음 — 저장 건너뜀")
        return None

    # 관측 시각 파싱
    obs_time = None
    if wl_data and "ymdhm" in wl_data:
        t = str(wl_data["ymdhm"]).strip()
        try:
            obs_time = datetime.strptime(t, "%Y%m%d%H%M").replace(tzinfo=KST)
        except ValueError:
            pass
    if obs_time is None:
        now = datetime.now(KST)
        rm = (now.minute // 10) * 10
        obs_time = now.replace(minute=rm, second=0, microsecond=0)

    return {
        "time":          obs_time,
        "water_level_m": water_level,
        "flow_rate_cms": flow_rate,
        "rainfall_mm":   rainfall,
    }


# ── DB 저장 ───────────────────────────────────────────────────────────────────
def save_to_db(record: dict) -> bool:
    """water_level_raw 테이블에 INSERT (중복 시 무시)"""
    if not DATABASE_URL:
        logger.error("DATABASE_URL 미설정")
        return False
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO water_level_raw
              (time, station_id, source, water_level_m, flow_rate_cms, rainfall_mm)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (
            record["time"], STATION_ID, "hrfc",
            record["water_level_m"],
            record["flow_rate_cms"],
            record["rainfall_mm"],
        ))
        conn.commit()
        cur.close()
        conn.close()
        logger.info("저장 완료 | time=%s | wl=%.3fm | rf=%s mm",
                    record["time"].strftime("%Y-%m-%d %H:%M"),
                    record["water_level_m"],
                    record["rainfall_mm"])
        return True
    except Exception as e:
        logger.error("DB 저장 실패: %s", e)
        return False


# ── DB 연결 대기 ──────────────────────────────────────────────────────────────
def wait_for_db(max_retries: int = 10, interval: int = 5):
    """PostgreSQL 준비될 때까지 대기"""
    for i in range(max_retries):
        try:
            conn = psycopg2.connect(DATABASE_URL, connect_timeout=3)
            conn.close()
            logger.info("DB 연결 확인 완료")
            return True
        except Exception:
            logger.info("DB 대기 중... (%d/%d)", i + 1, max_retries)
            time.sleep(interval)
    logger.error("DB 연결 실패")
    return False


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
def main():
    logger.info("="*50)
    logger.info("data_collector 시작")
    logger.info("수집 주기: %d초 | SKIP_API: %s", COLLECT_INTERVAL, SKIP_API)
    logger.info("="*50)

    # DB 준비 대기
    if DATABASE_URL:
        wait_for_db()

    while True:
        start = time.time()
        logger.info("--- 수집 시작 ---")

        record = collect_once()
        if record:
            save_to_db(record)
        else:
            if SKIP_API:
                logger.info("SKIP_API 모드 — 대기 중 (API 키 설정 후 SKIP_API=false로 변경)")

        elapsed = time.time() - start
        sleep_sec = max(0, COLLECT_INTERVAL - elapsed)
        logger.info("다음 수집까지 %.0f초 대기", sleep_sec)
        time.sleep(sleep_sec)


if __name__ == "__main__":
    main()