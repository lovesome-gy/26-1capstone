"""
utils/formatter.py
수위 데이터를 다양한 형식으로 포매팅하는 유틸리티 함수 모음.
Streamlit 대시보드 및 보고서 출력에서 공통 사용한다.
"""

from datetime import datetime


def format_water_level(level_m: float) -> str:
    """수위를 'X.XX m' 형식으로 포매팅한다."""
    return f"{level_m:.2f} m"


def format_alert_badge(alert_level: int) -> str:
    """경보 단계를 배지 텍스트로 변환한다."""
    badges = {
        0: "🟢 정상",
        1: "🔵 관심",
        2: "🟡 주의",
        3: "🟠 경계",
        4: "🔴 심각",
    }
    return badges.get(alert_level, f"?({alert_level})")


def format_trend_arrow(trend: str) -> str:
    """추세를 화살표 텍스트로 변환한다."""
    return {"rising": "↑ 상승", "falling": "↓ 하강", "stable": "→ 안정"}.get(
        trend, trend
    )


def format_kst(dt: datetime) -> str:
    """datetime을 'YYYY-MM-DD HH:MM (KST)' 형식으로 포매팅한다."""
    return dt.strftime("%Y-%m-%d %H:%M (KST)")


def level_to_alert(level_m: float) -> int:
    """수위(m) → 경보 단계(0~4) 변환. report_generator.py와 동기화 유지."""
    if level_m >= 10.5:
        return 4
    elif level_m >= 9.0:
        return 3
    elif level_m >= 7.5:
        return 2
    elif level_m >= 6.0:
        return 1
    return 0
