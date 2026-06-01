"""
services/report_templates.py

일간·주간·월간 보고서 유형별 프롬프트 분기 및 집계 데이터 구조 정의.
WBS Task 3.3 산출물.

보고서 유형:
  - hourly  : 1시간 단위 실시간 현황 (기본)
  - daily   : 일간 통계 요약 (평균/최고/최저 수위)
  - weekly  : 주간 추세 분석
  - monthly : 월간 종합 보고
  - alert   : 경보 발생 즉시 긴급 보고
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

ReportType = Literal["hourly", "daily", "weekly", "monthly", "alert"]


@dataclass
class AggregatedStats:
    """집계 기간의 수위 통계 요약."""
    avg_level: float
    max_level: float
    min_level: float
    max_level_time: datetime | None = None
    min_level_time: datetime | None = None
    alert_count: int = 0          # 경보 발생 횟수
    dominant_trend: str = "stable"  # 기간 중 주된 추세


@dataclass
class TemplateContext:
    """템플릿별 프롬프트 컨텍스트."""
    report_type: ReportType
    period_label: str             # "2025-07-15" / "2025년 7월 3주차" 등
    focus_instruction: str        # LLM에 전달할 작성 방향 지시
    required_sections: list[str] = field(default_factory=list)  # 본문에 포함해야 할 섹션


# ── 유형별 템플릿 컨텍스트 정의 ──────────────────────────────
TEMPLATE_CONTEXTS: dict[ReportType, TemplateContext] = {
    "hourly": TemplateContext(
        report_type="hourly",
        period_label="시간별 현황",
        focus_instruction="현재 수위와 1시간 후 예측 수위를 중심으로 즉각적인 상황을 서술하십시오.",
        required_sections=["현재 상태", "예측 수위", "조치 권고"],
    ),
    "daily": TemplateContext(
        report_type="daily",
        period_label="일간 보고",
        focus_instruction=(
            "하루 전체의 수위 변동 추세를 요약하십시오. "
            "평균·최고·최저 수위와 경보 발생 여부를 반드시 포함하십시오."
        ),
        required_sections=["일간 요약", "수위 통계", "경보 이력", "익일 전망"],
    ),
    "weekly": TemplateContext(
        report_type="weekly",
        period_label="주간 보고",
        focus_instruction=(
            "주간 수위 추세와 패턴을 분석하십시오. "
            "전주 대비 변화, 강우 이벤트와의 연관성, 다음 주 예상 동향을 포함하십시오."
        ),
        required_sections=["주간 요약", "주간 추세", "주요 이벤트", "다음 주 전망"],
    ),
    "monthly": TemplateContext(
        report_type="monthly",
        period_label="월간 보고",
        focus_instruction=(
            "월간 수위 현황을 종합적으로 분석하십시오. "
            "월별 평균 수위, 경보 발생 빈도, 계절적 패턴, 시설 운영 현황을 포함하십시오."
        ),
        required_sections=["월간 요약", "통계 분석", "경보 이력", "운영 평가", "다음 달 전망"],
    ),
    "alert": TemplateContext(
        report_type="alert",
        period_label="긴급 경보",
        focus_instruction=(
            "경보 발생 원인과 현재 위험 상황을 즉각적으로 전달하십시오. "
            "대피·수문 조작 등 즉각 조치가 필요한 사항을 최우선으로 서술하십시오."
        ),
        required_sections=["긴급 상황 요약", "위험 수위 현황", "즉각 조치 사항"],
    ),
}


def get_template_context(report_type: ReportType) -> TemplateContext:
    """보고서 유형에 맞는 템플릿 컨텍스트를 반환한다."""
    return TEMPLATE_CONTEXTS.get(report_type, TEMPLATE_CONTEXTS["hourly"])


def build_stats_section(stats: AggregatedStats | None) -> str:
    """
    집계 통계가 있을 때 프롬프트에 추가할 통계 섹션을 생성한다.
    hourly는 stats가 None이므로 빈 문자열을 반환한다.
    """
    if stats is None:
        return ""

    lines = [
        "\n[기간 통계]",
        f"- 평균 수위: {stats.avg_level:.2f} m",
        f"- 최고 수위: {stats.max_level:.2f} m"
        + (f" ({stats.max_level_time.strftime('%Y-%m-%d %H:%M')})" if stats.max_level_time else ""),
        f"- 최저 수위: {stats.min_level:.2f} m"
        + (f" ({stats.min_level_time.strftime('%Y-%m-%d %H:%M')})" if stats.min_level_time else ""),
        f"- 기간 중 주요 추세: {stats.dominant_trend}",
        f"- 경보 발생 횟수: {stats.alert_count}회",
    ]
    return "\n".join(lines)


def build_sections_instruction(ctx: TemplateContext) -> str:
    """보고서 본문에 포함해야 할 섹션 지시를 생성한다."""
    if not ctx.required_sections:
        return ""
    sections = ", ".join(f"[{s}]" for s in ctx.required_sections)
    return f"\n[필수 포함 섹션]\n{sections}"
