"""
core/database.py
비동기 SQLAlchemy 엔진과 세션 팩토리를 설정한다.
FastAPI lifespan 이벤트에서 초기화/종료를 처리한다.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

settings = get_settings()

# ── 엔진 ────────────────────────────────────────────────────
engine = create_async_engine(
    settings.database_url,
    echo=(settings.log_level == "DEBUG"),
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,             # 연결 유효성 사전 확인
)

# ── 세션 팩토리 ──────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,         # 커밋 후 lazy-load 방지
    autoflush=False,
)


# ── ORM 베이스 ───────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── 의존성 주입용 세션 제너레이터 ─────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI Depends에서 사용하는 DB 세션 의존성."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
