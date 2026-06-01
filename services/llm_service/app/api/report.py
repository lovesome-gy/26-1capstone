"""
api/report.py
보고서 생성 FastAPI 라우터
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.report import ReportListResponse, ReportRequest, ReportResponse
from app.services.report_generator import ReportGeneratorService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reports", tags=["보고서 생성"])


@router.post(
    "/generate",
    response_model=ReportResponse,
    summary="수위 현황 보고서 생성",
    description="수위 예측 데이터를 입력받아 LLM 기반 자연어 보고서를 생성하고 DB에 저장합니다.",
)
async def generate_report(
    req: ReportRequest,
    db: AsyncSession = Depends(get_db),
) -> ReportResponse:
    svc = ReportGeneratorService(db)
    try:
        return await svc.generate(req)
    except Exception as e:
        logger.error("보고서 생성 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"보고서 생성 중 오류 발생: {e}")


@router.get(
    "/",
    response_model=ReportListResponse,
    summary="보고서 목록 조회",
)
async def list_reports(
    station_id: str = Query(default="3008680"),
    limit: int = Query(default=10, ge=1, le=100),
    alert_level: int | None = Query(default=None, ge=0, le=4),
    db: AsyncSession = Depends(get_db),
) -> ReportListResponse:
    svc = ReportGeneratorService(db)
    items = await svc.get_recent(station_id, limit, alert_level)
    return ReportListResponse(total=len(items), items=items)


@router.get(
    "/{report_id}",
    response_model=ReportResponse,
    summary="보고서 단건 조회",
)
async def get_report(
    report_id: int,
    db: AsyncSession = Depends(get_db),
) -> ReportResponse:
    from sqlalchemy import select
    from app.models.report import MakeReportTB

    result = await db.execute(
        select(MakeReportTB).where(MakeReportTB.report_id == report_id)
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail=f"보고서 ID {report_id}를 찾을 수 없습니다.")
    return ReportResponse.model_validate(record)
