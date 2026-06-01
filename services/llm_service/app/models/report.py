"""
models/report.py
make_report_tb ORM 매핑 (Part④ 보고서 생성)
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Double,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class MakeReportTB(Base):
    __tablename__ = "make_report_tb"

    report_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    station_id: Mapped[str] = mapped_column(String(20), nullable=False, default="3008680")

    # 보고서 기간 (시작/종료)
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    report_type: Mapped[str] = mapped_column(String(20), nullable=False)   # hourly|daily|alert
    water_level_cur: Mapped[float | None] = mapped_column(Double)
    water_level_pred: Mapped[float | None] = mapped_column(Double)
    trend: Mapped[str | None] = mapped_column(String(10))                  # rising|falling|stable
    alert_level: Mapped[int] = mapped_column(SmallInteger, default=0)      # 0~4

    # LLM 생성 텍스트
    report_summary: Mapped[str] = mapped_column(Text, nullable=False)
    report_body: Mapped[str] = mapped_column(Text, nullable=False)

    # 메타
    llm_model: Mapped[str] = mapped_column(String(50), default="qwen3:8b")
    prompt_version: Mapped[str | None] = mapped_column(String(20))
    generation_ms: Mapped[int | None] = mapped_column(Integer)
    prediction_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # 역방향 관계
    decisions: Mapped[list["DecisionSupportTB"]] = relationship(
        "DecisionSupportTB", back_populates="report", lazy="select"
    )