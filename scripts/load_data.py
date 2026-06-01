"""
scripts/load_data.py
2년치 전처리 CSV 데이터를 PostgreSQL water_level_raw 테이블에 적재한다.

전달받은 데이터:
  data/merged/kwater_hrfc_10min_2024-01-01_2025-12-31_preprocessed.csv

실행:
    python scripts/load_data.py
    python scripts/load_data.py --csv data/merged/kwater_hrfc_10min_2024-01-01_2025-12-31_preprocessed.csv
    python scripts/load_data.py --dry-run   # 실제 적재 없이 파일 검증만
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ── 기본 설정 ────────────────────────────────────────────────
DEFAULT_CSV = Path("data/merged/kwater_hrfc_10min_2024-01-01_2025-12-31_preprocessed.csv")
DEFAULT_DB = "postgresql://yeoju_admin:password@localhost:5432/yeoju_water"
STATION_ID = "3008680"  # 여주보
BATCH_SIZE = 1000       # 한 번에 INSERT할 행 수


def load_env_db_url() -> str:
    """가능하면 .env에서 DATABASE_URL을 읽어온다."""
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        url = os.getenv("DATABASE_URL", "")
        # asyncpg URL → psycopg2 URL 변환
        return url.replace("postgresql+asyncpg://", "postgresql://") if url else DEFAULT_DB
    except ImportError:
        return DEFAULT_DB


def inspect_csv(path: Path) -> pd.DataFrame:
    """CSV를 읽고 컬럼 정보를 출력한다."""
    print(f"\n[1/4] CSV 파일 검사: {path}")
    if not path.exists():
        print(f"      ❌ 파일 없음: {path}")
        print(f"         data/merged/ 폴더에 전처리 CSV를 넣어주세요.")
        sys.exit(1)

    df = pd.read_csv(path, encoding="utf-8-sig", nrows=5)
    print(f"      ✅ 파일 확인 | 컬럼 수: {len(df.columns)}")
    print(f"         컬럼 목록: {list(df.columns)}")

    df_full = pd.read_csv(path, encoding="utf-8-sig")
    print(f"         전체 행 수: {len(df_full):,}")
    return df_full


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    다양한 컬럼명 형식을 통일된 내부 컬럼명으로 매핑한다.
    전달받은 CSV의 한글 컬럼명을 처리한다.
    """
    print("\n[2/4] 컬럼 정규화")

    col_map = {}
    cols_lower = {c.lower(): c for c in df.columns}

    # 시간 컬럼 탐색
    for candidate in ["bucket_start_kst", "datetime", "time", "timestamp", "일시", "관측시간", "date_time"]:
        if candidate in cols_lower:
            col_map[cols_lower[candidate]] = "time"
            break

    # 수위 컬럼 탐색 (hrfc 우선, kwater 차선)
    for candidate in ["hrfc_수위_m_평균", "hrfc_수위_m", "kwater_댐수위_m", "water_level_m", "수위"]:
        if candidate in df.columns:
            col_map[candidate] = "water_level_m"
            break

    # 강우 컬럼
    for candidate in ["kwater_강우량_mm", "rainfall_mm", "강우량_mm", "강우량"]:
        if candidate in df.columns:
            col_map[candidate] = "rainfall_mm"
            break

    # 유량 컬럼 (없으면 None)
    for candidate in ["flow_rate_cms", "유량_cms", "유량"]:
        if candidate in df.columns:
            col_map[candidate] = "flow_rate_cms"
            break

    if "time" not in col_map.values():
        print(f"      ❌ 시간 컬럼을 찾을 수 없음. 컬럼: {list(df.columns)}")
        sys.exit(1)
    if "water_level_m" not in col_map.values():
        print(f"      ❌ 수위 컬럼을 찾을 수 없음. 컬럼: {list(df.columns)}")
        sys.exit(1)

    df = df.rename(columns=col_map)
    print(f"      ✅ 매핑 완료: {col_map}")

    # 없는 컬럼은 None으로 채움
    for col in ["water_level_m", "rainfall_mm", "flow_rate_cms"]:
        if col not in df.columns:
            df[col] = None

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """시간 파싱, 결측 처리, 타입 변환."""
    print("\n[3/4] 전처리")

    # 시간 파싱 (KST)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # KST 타임존 붙이기 (이미 KST인 데이터이므로)
    if df["time"].dt.tz is None:
        import pytz
        df["time"] = df["time"].dt.tz_localize("Asia/Seoul")

    # 시간 파싱 실패 행 제거
    before = len(df)
    df = df.dropna(subset=["time"])
    after = len(df)
    if before != after:
        print(f"      ⚠️  시간 파싱 실패로 {before - after}행 제거")

    # 수치 컬럼 float 변환
    for col in ["water_level_m", "rainfall_mm", "flow_rate_cms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 중복 타임스탬프 제거
    dup = df.duplicated(subset=["time"])
    if dup.any():
        print(f"      ⚠️  중복 타임스탬프 {dup.sum()}건 제거")
        df = df.drop_duplicates(subset=["time"])

    df = df.sort_values("time").reset_index(drop=True)
    print(f"      ✅ 전처리 완료 | 최종 행 수: {len(df):,}")
    print(f"         기간: {df['time'].min()} ~ {df['time'].max()}")
    print(f"         수위 범위: {df['water_level_m'].min():.2f} ~ {df['water_level_m'].max():.2f} m")
    return df


def insert_to_db(df: pd.DataFrame, db_url: str, dry_run: bool = False) -> None:
    """배치 INSERT로 PostgreSQL에 적재한다."""
    print(f"\n[4/4] DB 적재 {'(DRY RUN - 실제 저장 안 함)' if dry_run else ''}")

    if dry_run:
        print(f"      ✅ DRY RUN 완료 | 적재 예정 행 수: {len(df):,}")
        return

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
    except Exception as e:
        print(f"      ❌ DB 연결 실패: {e}")
        print(f"         DB_URL: {db_url[:40]}...")
        sys.exit(1)

    # 기존 여주보 데이터 확인
    cur.execute("SELECT COUNT(*) FROM water_level_raw WHERE station_id = %s", (STATION_ID,))
    existing = cur.fetchone()[0]
    if existing > 0:
        print(f"      ⚠️  기존 데이터 {existing:,}건 존재 → UPSERT 방식으로 진행")

    # 배치 INSERT (ON CONFLICT DO NOTHING으로 중복 무시)
    total = len(df)
    inserted = 0

    for i in range(0, total, BATCH_SIZE):
        batch = df.iloc[i : i + BATCH_SIZE]
        rows = [
            (
                row["time"],
                STATION_ID,
                "hrfc",  # 전처리 병합 파일 기준
                row["water_level_m"] if pd.notna(row["water_level_m"]) else None,
                row["flow_rate_cms"] if pd.notna(row.get("flow_rate_cms", float("nan"))) else None,
                row["rainfall_mm"] if pd.notna(row.get("rainfall_mm", float("nan"))) else None,
                None,  # gate_open_pct
            )
            for _, row in batch.iterrows()
        ]

        execute_values(
            cur,
            """
            INSERT INTO water_level_raw
                (time, station_id, source, water_level_m, flow_rate_cms, rainfall_mm, gate_open_pct)
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            rows,
        )
        inserted += len(rows)

        # 진행률 표시
        pct = int(inserted / total * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r      [{bar}] {pct}% ({inserted:,}/{total:,})", end="", flush=True)

    conn.commit()
    print(f"\n      ✅ 적재 완료 | {inserted:,}건")

    # 적재 결과 확인
    cur.execute(
        "SELECT COUNT(*), MIN(time), MAX(time) FROM water_level_raw WHERE station_id = %s",
        (STATION_ID,),
    )
    count, min_t, max_t = cur.fetchone()
    print(f"      DB 확인: {count:,}건 | {min_t} ~ {max_t}")

    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="2년치 CSV → PostgreSQL 적재")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--dry-run", action="store_true", help="파일 검증만, DB 저장 안 함")
    args = parser.parse_args()

    db_url = args.db_url or load_env_db_url()

    print("=" * 60)
    print("  여주보 수위 데이터 적재  (2024-01-01 ~ 2025-12-31)")
    print("=" * 60)

    df = inspect_csv(args.csv)
    df = normalize_columns(df)
    df = preprocess(df)
    insert_to_db(df, db_url, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("  ✅ 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
