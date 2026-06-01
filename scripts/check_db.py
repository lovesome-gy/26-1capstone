"""
scripts/check_db.py
PostgreSQL 연결 확인 + 테이블 상태 + 적재된 데이터 요약 출력

실행:
    python scripts/check_db.py
    python scripts/check_db.py --db-url postgresql://yeoju_admin:pw@localhost:5432/yeoju_water
"""

import argparse
import sys

import psycopg2


DEFAULT_DB = "postgresql://yeoju_admin:password@localhost:5432/yeoju_water"

REQUIRED_TABLES = [
    "water_level_raw",
    "model_registry",
    "prediction_result",
    "make_report_tb",
    "decision_support_tb",
    "station_info",
]


def load_env_db_url() -> str:
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        url = os.getenv("DATABASE_URL", "")
        return url.replace("postgresql+asyncpg://", "postgresql://") if url else DEFAULT_DB
    except ImportError:
        return DEFAULT_DB


def check_connection(db_url: str):
    print("[1/4] DB 연결 확인")
    try:
        conn = psycopg2.connect(db_url, connect_timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT version()")
        ver = cur.fetchone()[0].split(",")[0]
        print(f"      ✅ 연결 성공 | {ver}")
        return conn
    except Exception as e:
        print(f"      ❌ 연결 실패: {e}")
        sys.exit(1)


def check_timescaledb(conn):
    print("\n[2/4] TimescaleDB 확장 확인")
    cur = conn.cursor()
    cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'timescaledb'")
    row = cur.fetchone()
    if row:
        print(f"      ✅ TimescaleDB {row[1]} 활성화됨")
    else:
        print("      ⚠️  TimescaleDB 미설치 — timescale/timescaledb 이미지인지 확인")


def check_tables(conn):
    print("\n[3/4] 테이블 존재 확인")
    cur = conn.cursor()
    cur.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
    )
    existing = {row[0] for row in cur.fetchall()}

    all_ok = True
    for table in REQUIRED_TABLES:
        ok = table in existing
        print(f"      {'✅' if ok else '❌'} {table}")
        if not ok:
            all_ok = False

    if not all_ok:
        print("\n      💡 init.sql이 실행되지 않았을 수 있습니다.")
        print("         docker compose up postgres 를 재실행하거나")
        print("         psql로 init.sql을 수동 실행하세요.")


def check_data(conn):
    print("\n[4/4] 데이터 현황")
    cur = conn.cursor()

    queries = {
        "water_level_raw (원시 수위)": """
            SELECT COUNT(*), MIN(time)::text, MAX(time)::text
            FROM water_level_raw
        """,
        "make_report_tb (LLM 보고서)": """
            SELECT COUNT(*), NULL, NULL FROM make_report_tb
        """,
        "decision_support_tb (의사결정)": """
            SELECT COUNT(*), NULL, NULL FROM decision_support_tb
        """,
        "model_registry (모델)": """
            SELECT COUNT(*), NULL, NULL FROM model_registry
        """,
    }

    for label, sql in queries.items():
        try:
            cur.execute(sql)
            count, min_t, max_t = cur.fetchone()
            range_str = f" | {min_t} ~ {max_t}" if min_t else ""
            print(f"      📊 {label}: {count:,}건{range_str}")
        except Exception as e:
            print(f"      ⚠️  {label}: 조회 실패 ({e})")


def main():
    parser = argparse.ArgumentParser(description="PostgreSQL 상태 확인")
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    db_url = args.db_url or load_env_db_url()

    print("=" * 60)
    print("  PostgreSQL 상태 확인")
    print(f"  {db_url[:45]}...")
    print("=" * 60)

    conn = check_connection(db_url)
    check_timescaledb(conn)
    check_tables(conn)
    check_data(conn)
    conn.close()

    print("\n" + "=" * 60)
    print("  완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
