"""
prompts/decision_prompt.py

의사결정 지원 프롬프트 템플릿.
경보 단계별로 다른 조치 카테고리와 우선순위 맥락을 주입한다.
"""

from dataclasses import dataclass

PROMPT_VERSION = "v1.0"

TREND_LABELS = {
    "rising": "상승 중",
    "falling": "하강 중",
    "stable": "안정",
}

# ── 경보 단계별 조치 맥락 ────────────────────────────────────
ALERT_CONTEXT = {
    0: {
        "label": "정상",
        "categories": ["monitoring"],
        "context": "수위가 정상 범위에 있습니다. 정기 모니터링 지침을 안내하십시오.",
    },
    1: {
        "label": "관심",
        "categories": ["monitoring", "standby"],
        "context": "수위 상승 추세 주시가 필요합니다. 대기 및 모니터링 강화 조치를 안내하십시오.",
    },
    2: {
        "label": "주의",
        "categories": ["gate_control", "monitoring", "standby"],
        "context": "수문 조작 검토 및 유관 기관 통보가 필요한 단계입니다.",
    },
    3: {
        "label": "경계",
        "categories": ["gate_control", "evacuation", "monitoring"],
        "context": "수문 조작과 대피 준비가 필요한 단계입니다. 즉각적인 조치 지침을 안내하십시오.",
    },
    4: {
        "label": "심각",
        "categories": ["evacuation", "gate_control"],
        "context": "즉각적인 대피 조치와 긴급 수문 제어가 필요한 최고 경보 단계입니다.",
    },
}

CATEGORY_LABELS = {
    "gate_control": "수문 제어",
    "evacuation": "대피 조치",
    "monitoring": "모니터링",
    "standby": "대기 준비",
}


@dataclass
class DecisionPromptInput:
    """의사결정 프롬프트에 전달할 입력 데이터."""
    station_name: str
    alert_level: int
    alert_label: str
    water_level_cur: float
    water_level_pred: float
    trend: str
    trend_label: str
    categories: list[str]
    alert_context: str


DECISION_SYSTEM_PROMPT = """당신은 수자원 관리 의사결정 지원 AI입니다.
수위 경보 단계에 따른 구체적인 조치 지침을 **한국어**로 제공하십시오.

[출력 형식 규칙]
1. 조치 항목마다 아래 XML 블록을 하나씩 출력할 것. (1~3개 항목)
2. 각 블록 구조:
   <decision>
     <category>gate_control|evacuation|monitoring|standby 중 하나</category>
     <priority>1(긴급)|2(일반)|3(참고) 중 하나</priority>
     <title>조치 제목 (30자 이내)</title>
     <body>조치 상세 내용 (100~200자)</body>
     <rationale>이 조치를 권고하는 근거 (50~100자)</rationale>
   </decision>
3. XML 블록 외 다른 텍스트는 출력하지 말 것.
4. 조치는 구체적이고 실행 가능한 내용으로 작성할 것.
5. 경보 단계와 관계없는 조치는 포함하지 말 것.

[조치 카테고리 설명]
- gate_control: 수문 개방/폐쇄, 방류량 조절 등 구조물 제어
- evacuation: 하류 주민 대피, 차량 통제, 대피소 운영
- monitoring: 수위 관측 주기 조정, 상류 상황 확인
- standby: 장비/인력 대기, 유관 기관 연락망 확인"""


def build_decision_user_prompt(inp: DecisionPromptInput) -> str:
    """유저 프롬프트를 구성하여 반환한다."""
    categories_str = ", ".join(
        CATEGORY_LABELS.get(c, c) for c in inp.categories
    )
    return f"""다음 상황에 대한 의사결정 지원 항목을 작성하십시오.

[현황]
- 관측소: {inp.station_name}
- 현재 수위: {inp.water_level_cur:.2f} m
- 예측 수위 (1시간 후): {inp.water_level_pred:.2f} m
- 수위 추세: {inp.trend_label}
- 경보 단계: {inp.alert_level}단계 ({inp.alert_label})

[상황 맥락]
{inp.alert_context}

[권장 조치 카테고리]
{categories_str}

위 카테고리를 중심으로 경보 단계에 적합한 조치 항목을 작성하십시오."""
