"""
scripts/check_performance.py
Task 6.1 산출물: 추론 속도 및 응답 품질 검증

목표: 5초 이내 응답, 보고서/의사결정 품질 기준 충족
실행:
    python scripts/check_performance.py
    python scripts/check_performance.py --runs 5 --base-url http://localhost:8002
"""

import argparse
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone

import httpx

KST = timezone(timedelta(hours=9))
TARGET_SECONDS = 5.0  # WBS 목표: 5초 이내


def _now_kst() -> str:
    return datetime.now(KST).isoformat()


def measure_response_time(base_url: str, n: int) -> list[float]:
    """보고서 생성 응답 시간 n회 측정."""
    times = []
    print(f"\n[1/3] 응답 시간 측정 ({n}회 반복)")
    payload = {
        "station_id": "3008680",
        "report_type": "hourly",
        "water_level_cur": 5.23,
        "water_level_pred": 5.87,
        "period_start": _now_kst(),
        "period_end": (datetime.now(KST) + timedelta(hours=1)).isoformat(),
    }
    for i in range(n):
        try:
            start = time.time()
            r = httpx.post(
                f"{base_url}/api/v1/reports/generate",
                json=payload,
                timeout=120,
            )
            elapsed = time.time() - start
            r.raise_for_status()
            times.append(elapsed)
            mark = "✅" if elapsed <= TARGET_SECONDS else "⚠️ "
            print(f"      {mark} [{i+1}/{n}] {elapsed:.2f}초")
        except Exception as e:
            print(f"      ❌ [{i+1}/{n}] 실패: {e}")

    return times


def evaluate_speed(times: list[float]) -> bool:
    """속도 통계 출력 및 목표 달성 여부 반환."""
    print("\n[2/3] 속도 통계")
    if not times:
        print("      ❌ 측정 데이터 없음")
        return False

    avg = statistics.mean(times)
    med = statistics.median(times)
    mn = min(times)
    mx = max(times)
    pass_rate = sum(1 for t in times if t <= TARGET_SECONDS) / len(times) * 100

    print(f"      평균:     {avg:.2f}초")
    print(f"      중앙값:   {med:.2f}초")
    print(f"      최소:     {mn:.2f}초")
    print(f"      최대:     {mx:.2f}초")
    print(f"      목표({TARGET_SECONDS}초) 달성율: {pass_rate:.0f}%")

    passed = pass_rate >= 80
    print(f"      {'✅ 속도 기준 통과 (80% 이상)' if passed else '⚠️  속도 기준 미달'}")
    return passed


def evaluate_quality(base_url: str) -> bool:
    """응답 품질 기준 확인."""
    print("\n[3/3] 응답 품질 검증")
    payload = {
        "station_id": "3008680",
        "report_type": "alert",
        "water_level_cur": 9.5,
        "water_level_pred": 10.2,
        "period_start": _now_kst(),
        "period_end": (datetime.now(KST) + timedelta(hours=1)).isoformat(),
    }
    criteria = {
        "report_summary 존재": False,
        "report_body 200자 이상": False,
        "alert_level == 3 (경계)": False,
        "trend 필드 존재": False,
        "generation_ms 기록": False,
    }
    try:
        r = httpx.post(
            f"{base_url}/api/v1/reports/generate",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()

        criteria["report_summary 존재"] = bool(data.get("report_summary"))
        criteria["report_body 200자 이상"] = len(data.get("report_body", "")) >= 200
        criteria["alert_level == 3 (경계)"] = data.get("alert_level") == 3
        criteria["trend 필드 존재"] = data.get("trend") is not None
        criteria["generation_ms 기록"] = data.get("generation_ms") is not None

    except Exception as e:
        print(f"      ❌ 품질 테스트 요청 실패: {e}")
        return False

    passed_all = True
    for item, ok in criteria.items():
        mark = "✅" if ok else "❌"
        print(f"      {mark} {item}")
        if not ok:
            passed_all = False

    return passed_all


def main():
    parser = argparse.ArgumentParser(description="추론 속도 및 응답 품질 검증")
    parser.add_argument("--base-url", default="http://localhost:8002")
    parser.add_argument("--runs", type=int, default=3, help="반복 측정 횟수")
    args = parser.parse_args()

    print("=" * 60)
    print("  추론 속도 & 응답 품질 검증  (Task 6.1)")
    print(f"  목표: {TARGET_SECONDS}초 이내 | 대상: {args.base_url}")
    print("=" * 60)

    times = measure_response_time(args.base_url, args.runs)
    ok_speed = evaluate_speed(times)
    ok_quality = evaluate_quality(args.base_url)

    print("\n" + "=" * 60)
    results = {"속도 기준": ok_speed, "품질 기준": ok_quality}
    for label, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {label}")
    all_pass = all(results.values())
    print(f"\n  최종: {'✅ 검증 통과' if all_pass else '⚠️  개선 필요'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
