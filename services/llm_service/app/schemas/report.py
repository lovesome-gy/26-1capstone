"""
schemas/report.py
보고서 생성 API의 요청/응답 Pydantic 스키마
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ReportRequest(BaseModel):
    station_id: str = Field(default="3008680")
    report_type: Literal["hourly", "daily", "weekly", "monthly", "alert"] = "hourly"
    water_level_cur: float = Field(..., ge=0, le=30)
    water_level_pred: float = Field(..., ge=0, le=30)
    period_start: datetime
    period_end: datetime
    prediction_id: int | None = None
    # 일간/주간/월간용 집계 통계 (선택)
    avg_level: float | None = None
    max_level: float | None = None
    min_level: float | None = None
    alert_count: int | None = None


class ReportResponse(BaseModel):
    report_id: int
    created_at: datetime
    station_id: str
    report_type: str
    alert_level: int
    trend: str | None
    water_level_cur: float | None
    water_level_pred: float | None
    report_summary: str
    report_body: str
    llm_model: str
    prompt_version: str | None
    generation_ms: int | None

    model_config = {"from_attributes": True}


class ReportListResponse(BaseModel):
    total: int
    items: list[ReportResponse]
