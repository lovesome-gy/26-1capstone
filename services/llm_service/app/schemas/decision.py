"""
schemas/decision.py
의사결정 지원 API의 요청/응답 Pydantic 스키마
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ── 요청 ─────────────────────────────────────────────────────
class DecisionRequest(BaseModel):
    """의사결정 지원 생성 요청 본문."""

    station_id: str = Field(default="3008680")
    alert_level: int = Field(..., ge=0, le=4, description="0:정상~4:심각")
    water_level_cur: float = Field(..., ge=0)
    water_level_pred: float = Field(..., ge=0)
    trend: Literal["rising", "falling", "stable"] = "stable"
    report_id: int | None = Field(default=None, description="연계 보고서 ID")
    prediction_id: int | None = None

    model_config = {"json_schema_extra": {
        "example": {
            "station_id": "3008680",
            "alert_level": 2,
            "water_level_cur": 7.8,
            "water_level_pred": 8.5,
            "trend": "rising",
            "report_id": 10,
        }
    }}


# ── 응답 ─────────────────────────────────────────────────────
class DecisionResponse(BaseModel):
    """개별 의사결정 항목 응답."""

    decision_id: int
    created_at: datetime
    station_id: str
    alert_level: int
    action_category: str
    priority: int = Field(description="1:긴급 2:일반 3:참고")
    decision_title: str
    decision_body: str
    rationale: str | None
    llm_model: str
    generation_ms: int | None
    is_acknowledged: bool

    model_config = {"from_attributes": True}


class DecisionListResponse(BaseModel):
    """의사결정 목록 응답 (단일 요청에서 복수 항목 생성 가능)."""

    total: int
    items: list[DecisionResponse]


class AcknowledgeRequest(BaseModel):
    """담당자 확인 처리 요청."""

    decision_id: int
