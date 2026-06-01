"""
api/decision.py
의사결정 지원 FastAPI 라우터
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.decision import (
    AcknowledgeRequest,
    DecisionListResponse,
    DecisionRequest,
    DecisionResponse,
)
from app.services.decision_support import DecisionSupportService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/decisions", tags=["의사결정 지원"])


@router.post(
    "/generate",
    response_model=DecisionListResponse,
    summary="의사결정 지원 항목 생성",
    description="경보 단계와 수위 데이터를 입력받아 LLM 기반 의사결정 지원 항목을 생성합니다.",
)
async def generate_decisions(
    req: DecisionRequest,
    db: AsyncSession = Depends(get_db),
) -> DecisionListResponse:
    svc = DecisionSupportService(db)
    try:
        items = await svc.generate(req)
        return DecisionListResponse(total=len(items), items=items)
    except Exception as e:
        logger.error("의사결정 생성 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"의사결정 생성 중 오류 발생: {e}")


@router.get(
    "/",
    response_model=DecisionListResponse,
    summary="의사결정 목록 조회",
)
async def list_decisions(
    station_id: str = Query(default="3008680"),
    limit: int = Query(default=10, ge=1, le=100),
    alert_level: int | None = Query(default=None, ge=0, le=4),
    unacknowledged_only: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
) -> DecisionListResponse:
    from sqlalchemy import select, desc
    from app.models.decision import DecisionSupportTB

    stmt = (
        select(DecisionSupportTB)
        .where(DecisionSupportTB.station_id == station_id)
        .order_by(desc(DecisionSupportTB.created_at))
        .limit(limit)
    )
    if alert_level is not None:
        stmt = stmt.where(DecisionSupportTB.alert_level == alert_level)
    if unacknowledged_only:
        stmt = stmt.where(DecisionSupportTB.is_acknowledged.is_(False))

    result = await db.execute(stmt)
    rows = result.scalars().all()
    items = [DecisionResponse.model_validate(r) for r in rows]
    return DecisionListResponse(total=len(items), items=items)


@router.patch(
    "/acknowledge",
    response_model=DecisionResponse,
    summary="담당자 확인 처리",
)
async def acknowledge_decision(
    req: AcknowledgeRequest,
    db: AsyncSession = Depends(get_db),
) -> DecisionResponse:
    svc = DecisionSupportService(db)
    result = await svc.acknowledge(req.decision_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"의사결정 ID {req.decision_id}를 찾을 수 없습니다.",
        )
    return result
