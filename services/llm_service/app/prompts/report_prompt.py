"""
prompts/report_prompt.py

보고서 생성용 프롬프트 템플릿 모음.
버전 관리: PROMPT_VERSION 상수로 추적.

설계 원칙:
  - 시스템 프롬프트: LLM의 역할과 출력 형식을 명확히 정의한다.
  - 유저 프롬프트: 수치 데이터를 구조적으로 전달한다.
  - Qwen3 특성상 한국어 지시에 충실하므로 한국어 프롬프트를 기본으로 한다.
"""

from dataclasses import dataclass
from datetime import datetime

PROMPT_VERSION = "v1.0"

# ── 경보 수준 텍스트 매핑 ────────────────────────────────────
ALERT_LABELS = {
    0: "정상",
    1: "관심",
    2: "주의",
    3: "경계",
    4: "심각",
}

TREND_LABELS = {
    "rising": "상승 중",
    "falling": "하강 중",
    "stable": "안정",
}


@dataclass
class ReportPromptInput:
    """보고서 프롬프트에 전달할 입력 데이터."""
    station_name: str
    report_type: str
    period_start: datetime
    period_end: datetime
    water_level_cur: float
    water_level_pred: float
    trend: str
    alert_level: int
    alert_label: str
    trend_label: str


REPORT_SYSTEM_PROMPT = """당신은 수자원 관리 전문 AI 리포터입니다.
주어진 수위 데이터를 바탕으로 **한국어** 수위 현황 보고서를 작성하십시오.

[출력 형식 규칙]
1. 응답은 반드시 아래 XML 태그 구조를 따를 것.
2. <summary>: 1~2 문장 핵심 요약 (50자 이내)
3. <body>: 전체 보고서 본문 (200~400자)
   - 현재 수위와 예측 수위를 구체적 수치로 언급할 것
   - 수위 추세(상승/하강/안정)와 그 의미를 설명할 것
   - 경보 단계에 따른 상황 해석을 포함할 것
   - 불필요한 추측이나 과장 없이 사실 기반으로 작성할 것
4. XML 태그 외 다른 텍스트는 출력하지 말 것.

[경보 단계 기준 - 여주보]
- 0 정상: 수위 6.0m 미만
- 1 관심: 6.0m ~ 7.5m 미만
- 2 주의: 7.5m ~ 9.0m 미만
- 3 경계: 9.0m ~ 10.5m 미만
- 4 심각: 10.5m 이상

[출력 예시]
<summary>여주보 수위가 현재 5.23m로 정상 범위를 유지하고 있으며, 향후 1시간 내 5.87m로 소폭 상승이 예상됩니다.</summary>
<body>2025년 7월 15일 14시 기준 여주보 수위는 5.23m로 정상(6.0m 미만) 범위에 있습니다. 최근 1시간 동안 수위는 완만한 상승 추세를 보이고 있으며, 향후 1시간 내 5.87m까지 상승할 것으로 예측됩니다. 현재 경보 단계는 0단계(정상)로, 즉각적인 조치는 불필요하나 상류 강우 상황에 따라 지속적인 모니터링이 권장됩니다.</body>"""


def build_report_user_prompt(inp: ReportPromptInput) -> str:
    """유저 프롬프트를 구성하여 반환한다."""
    report_type_kr = {
        "hourly": "시간별",
        "daily": "일별",
        "weekly": "주간",
        "monthly": "월간",
        "alert": "긴급",
    }.get(inp.report_type, inp.report_type)
    return f"""다음 데이터를 기반으로 {report_type_kr} 수위 현황 보고서를 작성하십시오.

[관측 정보]
- 관측소: {inp.station_name}
- 보고 기간: {inp.period_start.strftime('%Y-%m-%d %H:%M')} ~ {inp.period_end.strftime('%Y-%m-%d %H:%M')} (KST)

[수위 데이터]
- 현재 수위: {inp.water_level_cur:.2f} m
- 예측 수위 (1시간 후): {inp.water_level_pred:.2f} m
- 수위 변화 방향: {inp.trend_label}
- 현재 경보 단계: {inp.alert_level}단계 ({inp.alert_label})"""
