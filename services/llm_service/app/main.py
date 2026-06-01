"""
main.py
LLM Service FastAPI 애플리케이션 진입점.

담당: 신가연 (Part④ - Generative AI / LLM Integration)
"""

import logging
import logging.config
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.database import engine, Base
from app.api.report import router as report_router
from app.api.decision import router as decision_router
from app.services.ollama_client import OllamaClient

settings = get_settings()

# ── 로깅 설정 ────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행되는 이벤트 핸들러."""

    # ── Startup ───────────────────────────────────────────
    logger.info("LLM Service 시작 중...")

    # DB 테이블 생성 (이미 init.sql로 생성됐다면 충돌 없음)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("DB 연결 확인 완료")

    # Ollama 가용성 확인 + 모델 준비
    ollama = OllamaClient()
    if await ollama.is_available():
        logger.info("Ollama 서버 연결 확인 완료: %s", settings.ollama_base_url)
        await ollama.ensure_model_pulled()
    else:
        logger.warning(
            "Ollama 서버에 연결할 수 없습니다: %s — 보고서 생성 기능이 비활성화됩니다.",
            settings.ollama_base_url,
        )

    logger.info(
        "LLM Service 준비 완료 | host=%s:%d | model=%s",
        settings.llm_service_host,
        settings.llm_service_port,
        settings.ollama_model,
    )

    yield  # ── 서비스 실행 중 ──────────────────────────

    # ── Shutdown ──────────────────────────────────────────
    await engine.dispose()
    logger.info("LLM Service 종료")


# ── FastAPI 앱 ───────────────────────────────────────────────
app = FastAPI(
    title="여주보 수위 예측 LLM 서비스",
    description=(
        "실시간 수위 예측 결과를 자연어 보고서와 의사결정 지원 항목으로 변환하는 API.\n\n"
        "- **Part④ 담당**: 신가연\n"
        "- **LLM**: Ollama / Qwen3-8b (온프레미스)\n"
        "- **주요 테이블**: `make_report_tb`, `decision_support_tb`"
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 개발 환경: 전체 허용 / 운영 시 제한
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 라우터 등록 ──────────────────────────────────────────────
app.include_router(report_router,   prefix="/api/v1")
app.include_router(decision_router, prefix="/api/v1")


# ── 헬스체크 ─────────────────────────────────────────────────
@app.get("/health", tags=["헬스체크"])
async def health() -> dict:
    ollama = OllamaClient()
    return {
        "status": "ok",
        "service": "llm_service",
        "ollama_available": await ollama.is_available(),
        "ollama_model": settings.ollama_model,
    }


# ── 직접 실행 ─────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.llm_service_host,
        port=settings.llm_service_port,
        workers=settings.llm_service_workers,
        reload=False,
        log_level=settings.log_level.lower(),
    )
