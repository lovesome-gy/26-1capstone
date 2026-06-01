"""
services/report_generator.py

수위 예측 데이터 → LLM 호출 → DB 저장 파이프라인.
핵심 비즈니스 로직을 담당한다.
"""

import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.report import MakeReportTB
from app.prompts.report_prompt import (
    ALERT_LABELS,
    PROMPT_VERSION,
    TREND_LABELS,
    ReportPromptInput,
    REPORT_SYSTEM_PROMPT,
    build_report_user_prompt,
)
from app.schemas.report import ReportRequest, ReportResponse
from app.services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)
settings = get_settings()


def _compute_alert_level(water_level_m: float) -> int:
    """
    여주보 수위(m)를 경보 단계(0~4)로 변환한다.
    기준값은 station_info 테이블과 동기화되어야 한다.
    """
    if water_level_m >= 10.5:
        return 4
    elif water_level_m >= 9.0:
        return 3
    elif water_level_m >= 7.5:
        return 2
    elif water_level_m >= 6.0:
        return 1
    return 0


def _compute_trend(cur: float, pred: float) -> str:
    """현재 수위와 예측 수위를 비교해 추세를 결정한다."""
    diff = pred - cur
    if diff > 0.1:
        return "rising"
    elif diff < -0.1:
        return "falling"
    return "stable"


def _parse_llm_report(raw: str) -> tuple[str, str]:
    """
    LLM 출력에서 <summary>와 <body> 태그를 추출한다.

    Returns:
        (summary, body) 튜플. 파싱 실패 시 원문을 body에 넣고 summary를 잘라 반환.
    """
    summary_match = re.search(r"<summary>(.*?)</summary>", raw, re.DOTALL)
    body_match = re.search(r"<body>(.*?)</body>", raw, re.DOTALL)

    summary = summary_match.group(1).strip() if summary_match else ""
    body = body_match.group(1).strip() if body_match else raw.strip()

    # 파싱 실패 폴백
    if not summary and body:
        summary = body[:80] + ("..." if len(body) > 80 else "")

    return summary, body


class ReportGeneratorService:
    """
    수위 보고서 생성 서비스.

    흐름:
        1. 입력 데이터로 경보 단계/추세 계산
        2. 프롬프트 구성
        3. Ollama LLM 호출
        4. XML 파싱
        5. DB 저장
        6. 응답 반환
    """

    def __init__(self, db: AsyncSession):
        self._db = db
        self._llm = OllamaClient()

    async def generate(self, req: ReportRequest) -> ReportResponse:
        """보고서를 생성하고 DB에 저장한 뒤 응답을 반환한다."""

        # ── 1. 경보 단계 & 추세 계산 ──────────────────────
        alert_level = _compute_alert_level(
            max(req.water_level_cur, req.water_level_pred)
        )
        trend = _compute_trend(req.water_level_cur, req.water_level_pred)

        # ── 2. 프롬프트 구성 ──────────────────────────────
        prompt_input = ReportPromptInput(
            station_name="여주보",
            report_type=req.report_type,
            period_start=req.period_start,
            period_end=req.period_end,
            water_level_cur=req.water_level_cur,
            water_level_pred=req.water_level_pred,
            trend=trend,
            alert_level=alert_level,
            alert_label=ALERT_LABELS[alert_level],
            trend_label=TREND_LABELS[trend],
        )
        user_prompt = build_report_user_prompt(prompt_input)

        # ── 3. LLM 호출 ───────────────────────────────────
        logger.info(
            "보고서 생성 시작 | type=%s | level=%d | cur=%.2fm",
            req.report_type,
            alert_level,
            req.water_level_cur,
        )
        llm_resp = await self._llm.chat(
            system_prompt=REPORT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
        )

        # ── 4. XML 파싱 ───────────────────────────────────
        summary, body = _parse_llm_report(llm_resp.content)

        # ── 5. DB 저장 ────────────────────────────────────
        record = MakeReportTB(
            station_id=req.station_id,
            period_start=req.period_start,
            period_end=req.period_end,
            report_type=req.report_type,
            water_level_cur=req.water_level_cur,
            water_level_pred=req.water_level_pred,
            trend=trend,
            alert_level=alert_level,
            report_summary=summary,
            report_body=body,
            llm_model=llm_resp.model,
            prompt_version=PROMPT_VERSION,
            generation_ms=llm_resp.generation_ms,
            prediction_id=req.prediction_id,
        )
        self._db.add(record)
        await self._db.flush()   # ID 획득
        await self._db.refresh(record)

        logger.info(
            "보고서 생성 완료 | report_id=%d | %dms",
            record.report_id,
            llm_resp.generation_ms,
        )

        return ReportResponse.model_validate(record)

    async def get_recent(
        self,
        station_id: str,
        limit: int = 10,
        alert_level: int | None = None,
    ) -> list[ReportResponse]:
        """최근 보고서 목록을 조회한다."""
        from sqlalchemy import select, desc

        stmt = (
            select(MakeReportTB)
            .where(MakeReportTB.station_id == station_id)
            .order_by(desc(MakeReportTB.created_at))
            .limit(limit)
        )
        if alert_level is not None:
            stmt = stmt.where(MakeReportTB.alert_level == alert_level)

        result = await self._db.execute(stmt)
        rows = result.scalars().all()
        return [ReportResponse.model_validate(r) for r in rows]


# ── 템플릿 기반 보고서 생성 헬퍼 (Task 3.3) ─────────────────
from app.services.report_templates import (
    AggregatedStats,
    get_template_context,
    build_stats_section,
    build_sections_instruction,
)


def build_full_user_prompt(
    inp,
    stats: AggregatedStats | None = None,
) -> str:
    """
    템플릿 컨텍스트 + 집계 통계를 합쳐 최종 유저 프롬프트를 생성한다.
    hourly/alert는 stats=None, daily/weekly/monthly는 stats를 전달한다.
    """
    base = build_report_user_prompt(inp)
    ctx = get_template_context(inp.report_type)
    stats_section = build_stats_section(stats)
    sections_instruction = build_sections_instruction(ctx)
    focus = f"\n[작성 방향]\n{ctx.focus_instruction}"
    return base + stats_section + sections_instruction + focus
