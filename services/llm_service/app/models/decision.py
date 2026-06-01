"""
models/decision.py
decision_support_tb ORM 매핑 (Part④ 의사결정 지원)
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class DecisionSupportTB(Base):
    __tablename__ = "decision_support_tb"

    decision_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    station_id: Mapped[str] = mapped_column(String(20), nullable=False, default="3008680")
    alert_level: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    # 의사결정 분류
    action_category: Mapped[str] = mapped_column(String(30), nullable=False)
    # gate_control | evacuation | monitoring | standby
    priority: Mapped[int] = mapped_column(SmallInteger, default=2)
    # 1:긴급 2:일반 3:참고

    # LLM 생성 텍스트
    decision_title: Mapped[str] = mapped_column(String(200), nullable=False)
    decision_body: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text)

    # 연계 키
    report_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("make_report_tb.report_id"), nullable=True
    )
    prediction_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # 메타
    llm_model: Mapped[str] = mapped_column(String(50), default="qwen3:8b")
    prompt_version: Mapped[str | None] = mapped_column(String(20))
    generation_ms: Mapped[int | None] = mapped_column(Integer)
    is_acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 역방향 관계
    report: Mapped["MakeReportTB | None"] = relationship(  # noqa: F821
        "MakeReportTB", back_populates="decisions"
    )