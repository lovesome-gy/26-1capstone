"""
services/decision_support.py

경보 단계별 의사결정 지원 항목 생성 서비스.
LLM이 생성한 XML 블록을 파싱해 복수 항목을 DB에 저장한다.
"""

import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.decision import DecisionSupportTB
from app.prompts.decision_prompt import (
    ALERT_CONTEXT,
    CATEGORY_LABELS,
    DECISION_SYSTEM_PROMPT,
    PROMPT_VERSION,
    TREND_LABELS,
    DecisionPromptInput,
    build_decision_user_prompt,
)
from app.schemas.decision import DecisionRequest, DecisionResponse
from app.services.ollama_client import OllamaClient

logger = logging.getLogger(__name__)
settings = get_settings()


def _parse_decision_blocks(raw: str) -> list[dict]:
    """
    LLM 출력에서 <decision> 블록을 모두 파싱한다.

    Returns:
        파싱된 딕셔너리 목록. 각 항목: category, priority, title, body, rationale
    """
    blocks = re.findall(r"<decision>(.*?)</decision>", raw, re.DOTALL)
    results = []

    for block in blocks:
        def _extract(tag: str) -> str:
            m = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.DOTALL)
            return m.group(1).strip() if m else ""

        category = _extract("category")
        priority_str = _extract("priority")
        title = _extract("title")
        body = _extract("body")
        rationale = _extract("rationale")

        # 유효성 기본 검사
        if not title or not body:
            logger.warning("의사결정 블록 파싱 불완전, 건너뜀: %s", block[:80])
            continue

        try:
            priority = int(priority_str)
        except (ValueError, TypeError):
            priority = 2  # 기본: 일반

        # category 검증
        if category not in CATEGORY_LABELS:
            category = "monitoring"

        results.append({
            "action_category": category,
            "priority": priority,
            "decision_title": title,
            "decision_body": body,
            "rationale": rationale or None,
        })

    return results


class DecisionSupportService:
    """
    의사결정 지원 생성 서비스.

    흐름:
        1. 경보 단계에 맞는 프롬프트 컨텍스트 선택
        2. LLM 호출
        3. XML 블록 파싱 (1~3개 항목)
        4. DB 저장 (복수 레코드)
        5. 응답 반환
    """

    def __init__(self, db: AsyncSession):
        self._db = db
        self._llm = OllamaClient()

    async def generate(self, req: DecisionRequest) -> list[DecisionResponse]:
        """의사결정 지원 항목을 생성하고 DB에 저장한 뒤 목록을 반환한다."""

        ctx = ALERT_CONTEXT.get(req.alert_level, ALERT_CONTEXT[0])
        trend_label = TREND_LABELS.get(req.trend, req.trend)

        # ── 프롬프트 구성 ────────────────────────────────
        prompt_input = DecisionPromptInput(
            station_name="여주보",
            alert_level=req.alert_level,
            alert_label=ctx["label"],
            water_level_cur=req.water_level_cur,
            water_level_pred=req.water_level_pred,
            trend=req.trend,
            trend_label=trend_label,
            categories=ctx["categories"],
            alert_context=ctx["context"],
        )
        user_prompt = build_decision_user_prompt(prompt_input)

        # ── LLM 호출 ─────────────────────────────────────
        logger.info(
            "의사결정 생성 시작 | alert_level=%d | cur=%.2fm",
            req.alert_level,
            req.water_level_cur,
        )
        llm_resp = await self._llm.chat(
            system_prompt=DECISION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,   # 의사결정은 더 결정론적으로
        )

        # ── XML 파싱 ──────────────────────────────────────
        parsed_items = _parse_decision_blocks(llm_resp.content)

        if not parsed_items:
            logger.error("의사결정 XML 파싱 실패, 원문: %s", llm_resp.content[:200])
            # 폴백: 원문을 단일 항목으로 저장
            parsed_items = [{
                "action_category": "monitoring",
                "priority": 2,
                "decision_title": f"{ctx['label']} 단계 상황 모니터링",
                "decision_body": llm_resp.content.strip(),
                "rationale": None,
            }]

        # ── DB 저장 ───────────────────────────────────────
        saved_records: list[DecisionSupportTB] = []
        for item in parsed_items:
            record = DecisionSupportTB(
                station_id=req.station_id,
                alert_level=req.alert_level,
                action_category=item["action_category"],
                priority=item["priority"],
                decision_title=item["decision_title"],
                decision_body=item["decision_body"],
                rationale=item["rationale"],
                report_id=req.report_id,
                prediction_id=req.prediction_id,
                llm_model=llm_resp.model,
                prompt_version=PROMPT_VERSION,
                generation_ms=llm_resp.generation_ms,
            )
            self._db.add(record)
            saved_records.append(record)

        await self._db.flush()
        for r in saved_records:
            await self._db.refresh(r)

        logger.info(
            "의사결정 생성 완료 | %d개 항목 | %dms",
            len(saved_records),
            llm_resp.generation_ms,
        )

        return [DecisionResponse.model_validate(r) for r in saved_records]

    async def acknowledge(self, decision_id: int) -> DecisionResponse | None:
        """담당자 확인(acknowledge) 처리."""
        from datetime import datetime, timezone
        from sqlalchemy import select

        result = await self._db.execute(
            select(DecisionSupportTB).where(
                DecisionSupportTB.decision_id == decision_id
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            return None

        record.is_acknowledged = True
        record.acknowledged_at = datetime.now(timezone.utc)
        await self._db.flush()
        await self._db.refresh(record)
        return DecisionResponse.model_validate(record)
