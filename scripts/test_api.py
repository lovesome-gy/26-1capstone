"""
scripts/test_api.py
Task 5.3 산출물: FastAPI ↔ LLM 모듈 E2E 연동 테스트

실행:
    python scripts/test_api.py
    python scripts/test_api.py --base-url http://localhost:8002
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone

import httpx

KST = timezone(timedelta(hours=9))


def _now_kst() -> str:
    return datetime.now(KST).isoformat()


def _hour_later_kst() -> str:
    return (datetime.now(KST) + timedelta(hours=1)).isoformat()


def test_health(base_url: str) -> bool:
    """헬스체크 엔드포인트."""
    print("[1/6] GET /health")
    try:
        r = httpx.get(f"{base_url}/health", timeout=10)
        r.raise_for_status()
        data = r.json()
        ollama_ok = data.get("ollama_available", False)
        print(f"      ✅ 서비스 정상 | ollama_available={ollama_ok}")
        if not ollama_ok:
            print("      ⚠️  Ollama 미연결 — 보고서 생성 테스트는 실패할 수 있음")
        return True
    except Exception as e:
        print(f"      ❌ 헬스체크 실패: {e}")
        return False


def test_report_generate(base_url: str) -> tuple[bool, int | None]:
    """보고서 생성 엔드포인트 (hourly)."""
    print("\n[2/6] POST /api/v1/reports/generate (hourly)")
    payload = {
        "station_id": "3008680",
        "report_type": "hourly",
        "water_level_cur": 5.23,
        "water_level_pred": 5.87,
        "period_start": _now_kst(),
        "period_end": _hour_later_kst(),
    }
    try:
        start = time.time()
        r = httpx.post(
            f"{base_url}/api/v1/reports/generate",
            json=payload,
            timeout=120,
        )
        elapsed = time.time() - start
        r.raise_for_status()
        data = r.json()
        report_id = data.get("report_id")
        print(f"      ✅ 보고서 생성 성공 | report_id={report_id} | {elapsed:.1f}초")
        print(f"         alert_level={data.get('alert_level')} | trend={data.get('trend')}")
        print(f"         요약: {data.get('report_summary', '')[:80]}")
        return True, report_id
    except Exception as e:
        print(f"      ❌ 보고서 생성 실패: {e}")
        return False, None


def test_report_daily(base_url: str) -> bool:
    """일간 보고서 생성 테스트 (Task 3.3)."""
    print("\n[3/6] POST /api/v1/reports/generate (daily - Task 3.3)")
    payload = {
        "station_id": "3008680",
        "report_type": "daily",
        "water_level_cur": 5.50,
        "water_level_pred": 5.40,
        "period_start": (datetime.now(KST) - timedelta(days=1)).isoformat(),
        "period_end": _now_kst(),
        "avg_level": 5.35,
        "max_level": 6.10,
        "min_level": 4.90,
        "alert_count": 0,
    }
    try:
        r = httpx.post(
            f"{base_url}/api/v1/reports/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        print(f"      ✅ 일간 보고서 성공 | report_id={data.get('report_id')}")
        return True
    except Exception as e:
        print(f"      ❌ 일간 보고서 실패: {e}")
        return False


def test_report_list(base_url: str) -> bool:
    """보고서 목록 조회."""
    print("\n[4/6] GET /api/v1/reports/")
    try:
        r = httpx.get(
            f"{base_url}/api/v1/reports/",
            params={"station_id": "3008680", "limit": 5},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        print(f"      ✅ 목록 조회 성공 | total={data.get('total')}")
        return True
    except Exception as e:
        print(f"      ❌ 목록 조회 실패: {e}")
        return False


def test_decision_generate(base_url: str, report_id: int | None) -> tuple[bool, int | None]:
    """의사결정 지원 생성."""
    print("\n[5/6] POST /api/v1/decisions/generate")
    payload = {
        "station_id": "3008680",
        "alert_level": 2,
        "water_level_cur": 7.80,
        "water_level_pred": 8.50,
        "trend": "rising",
        "report_id": report_id,
    }
    try:
        start = time.time()
        r = httpx.post(
            f"{base_url}/api/v1/decisions/generate",
            json=payload,
            timeout=120,
        )
        elapsed = time.time() - start
        r.raise_for_status()
        data = r.json()
        total = data.get("total", 0)
        decision_id = data["items"][0]["decision_id"] if data.get("items") else None
        print(f"      ✅ 의사결정 생성 성공 | {total}개 항목 | {elapsed:.1f}초")
        for item in data.get("items", []):
            print(f"         [{item['priority']}] {item['decision_title']}")
        return True, decision_id
    except Exception as e:
        print(f"      ❌ 의사결정 생성 실패: {e}")
        return False, None


def test_acknowledge(base_url: str, decision_id: int | None) -> bool:
    """담당자 확인 처리."""
    print("\n[6/6] PATCH /api/v1/decisions/acknowledge")
    if decision_id is None:
        print("      ⏭️  decision_id 없음, 건너뜀")
        return True
    try:
        r = httpx.patch(
            f"{base_url}/api/v1/decisions/acknowledge",
            json={"decision_id": decision_id},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        print(f"      ✅ 확인 처리 성공 | is_acknowledged={data.get('is_acknowledged')}")
        return True
    except Exception as e:
        print(f"      ❌ 확인 처리 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="LLM Service API E2E 테스트")
    parser.add_argument("--base-url", default="http://localhost:8002")
    args = parser.parse_args()

    print("=" * 60)
    print("  FastAPI ↔ LLM 모듈 E2E 테스트  (Task 5.3)")
    print(f"  대상: {args.base_url}")
    print("=" * 60)

    ok1 = test_health(args.base_url)
    ok2, report_id = test_report_generate(args.base_url)
    ok3 = test_report_daily(args.base_url)
    ok4 = test_report_list(args.base_url)
    ok5, decision_id = test_decision_generate(args.base_url, report_id)
    ok6 = test_acknowledge(args.base_url, decision_id)

    results = [ok1, ok2, ok3, ok4, ok5, ok6]
    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 60)
    status = "✅ 전체 통과" if passed == total else f"⚠️  {passed}/{total} 통과"
    print(f"  결과: {status}")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
