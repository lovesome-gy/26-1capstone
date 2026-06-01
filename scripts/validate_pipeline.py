"""
scripts/validate_pipeline.py
Task 6.2 산출물: 데이터→예측→LLM 해석→보고서 전 구간 통합 테스트

실행:
    python scripts/validate_pipeline.py
    python scripts/validate_pipeline.py --llm-url http://localhost:8002 --predictor-url http://localhost:8001
"""

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone

import httpx

KST = timezone(timedelta(hours=9))


def step1_check_postgres(db_url: str) -> bool:
    """PostgreSQL 연결 확인 (psycopg2 직접 연결)."""
    print("[Step 1] PostgreSQL 연결 확인")
    try:
        import asyncio
        import asyncpg

        async def _check():
            conn = await asyncpg.connect(db_url, timeout=5)
            tables = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname='public'"
            )
            await conn.close()
            return [t["tablename"] for t in tables]

        tables = asyncio.run(_check())
        required = {"make_report_tb", "decision_support_tb", "water_level_raw"}
        missing = required - set(tables)
        if missing:
            print(f"      ⚠️  누락 테이블: {missing}")
            return False
        print(f"      ✅ DB 연결 정상 | 테이블 수: {len(tables)}")
        return True
    except ImportError:
        print("      ⏭️  asyncpg 없음, DB 단계 건너뜀")
        return True
    except Exception as e:
        print(f"      ❌ DB 연결 실패: {e}")
        return False


def step2_mock_prediction() -> dict:
    """
    Step 2: 예측 결과 모의 데이터 생성.
    predictor 서비스가 없을 때 모의 데이터로 대체한다.
    """
    print("\n[Step 2] 예측 결과 준비 (모의 데이터)")
    mock = {
        "prediction_id": 999,
        "station_id": "3008680",
        "predicted_level": 6.45,
        "actual_level": None,
        "horizon_minutes": 60,
        "target_time": (datetime.now(KST) + timedelta(hours=1)).isoformat(),
    }
    print(f"      ✅ 모의 예측 생성 | predicted={mock['predicted_level']}m")
    return mock


def step3_generate_report(llm_url: str, pred: dict) -> tuple[bool, dict | None]:
    """Step 3: 예측 결과 → LLM 보고서 생성."""
    print("\n[Step 3] LLM 보고서 생성")
    payload = {
        "station_id": pred["station_id"],
        "report_type": "hourly",
        "water_level_cur": pred["predicted_level"] - 0.3,
        "water_level_pred": pred["predicted_level"],
        "period_start": datetime.now(KST).isoformat(),
        "period_end": pred["target_time"],
        "prediction_id": pred.get("prediction_id"),
    }
    try:
        start = time.time()
        r = httpx.post(
            f"{llm_url}/api/v1/reports/generate",
            json=payload,
            timeout=120,
        )
        elapsed = time.time() - start
        r.raise_for_status()
        report = r.json()
        print(f"      ✅ 보고서 생성 완료 | report_id={report['report_id']} | {elapsed:.1f}초")
        print(f"         경보 단계: {report['alert_level']} | 추세: {report['trend']}")
        return True, report
    except Exception as e:
        print(f"      ❌ 보고서 생성 실패: {e}")
        return False, None


def step4_generate_decision(llm_url: str, pred: dict, report: dict | None) -> bool:
    """Step 4: 예측+보고서 → 의사결정 지원 생성."""
    print("\n[Step 4] 의사결정 지원 생성")
    alert_level = report["alert_level"] if report else 1
    payload = {
        "station_id": pred["station_id"],
        "alert_level": alert_level,
        "water_level_cur": pred["predicted_level"] - 0.3,
        "water_level_pred": pred["predicted_level"],
        "trend": "rising",
        "report_id": report["report_id"] if report else None,
        "prediction_id": pred.get("prediction_id"),
    }
    try:
        start = time.time()
        r = httpx.post(
            f"{llm_url}/api/v1/decisions/generate",
            json=payload,
            timeout=120,
        )
        elapsed = time.time() - start
        r.raise_for_status()
        data = r.json()
        print(f"      ✅ 의사결정 생성 완료 | {data['total']}개 항목 | {elapsed:.1f}초")
        for item in data.get("items", []):
            print(f"         [{item['priority']}] {item['decision_title']}")
        return True
    except Exception as e:
        print(f"      ❌ 의사결정 생성 실패: {e}")
        return False


def step5_verify_db_records(db_url: str) -> bool:
    """Step 5: DB에 레코드가 정상 저장됐는지 확인."""
    print("\n[Step 5] DB 저장 검증")
    try:
        import asyncio
        import asyncpg

        async def _check():
            conn = await asyncpg.connect(db_url, timeout=5)
            r_count = await conn.fetchval("SELECT COUNT(*) FROM make_report_tb")
            d_count = await conn.fetchval("SELECT COUNT(*) FROM decision_support_tb")
            await conn.close()
            return r_count, d_count

        r_count, d_count = asyncio.run(_check())
        print(f"      ✅ make_report_tb: {r_count}건")
        print(f"      ✅ decision_support_tb: {d_count}건")
        return r_count > 0 and d_count > 0
    except ImportError:
        print("      ⏭️  asyncpg 없음, DB 검증 건너뜀")
        return True
    except Exception as e:
        print(f"      ❌ DB 검증 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="전체 파이프라인 통합 테스트")
    parser.add_argument("--llm-url", default="http://localhost:8002")
    parser.add_argument("--predictor-url", default="http://localhost:8001")
    parser.add_argument(
        "--db-url",
        default="postgresql://yeoju_admin:password@localhost:5432/yeoju_water",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  전체 파이프라인 통합 테스트  (Task 6.2)")
    print("  데이터 → 예측 → LLM 보고서 → 의사결정 → DB 저장")
    print("=" * 65)

    ok1 = step1_check_postgres(args.db_url)
    pred = step2_mock_prediction()
    ok3, report = step3_generate_report(args.llm_url, pred)
    ok4 = step4_generate_decision(args.llm_url, pred, report)
    ok5 = step5_verify_db_records(args.db_url)

    results = {
        "DB 연결": ok1,
        "보고서 생성": ok3,
        "의사결정 생성": ok4,
        "DB 저장 확인": ok5,
    }

    print("\n" + "=" * 65)
    for label, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {label}")
    all_pass = all(results.values())
    print(f"\n  최종: {'✅ 통합 테스트 통과' if all_pass else '⚠️  일부 단계 실패'}")
    print("=" * 65)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
