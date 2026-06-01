"""
core/config.py
환경변수를 타입-세이프하게 관리한다.
.env 파일 또는 Docker env_file에서 자동으로 로드된다.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── PostgreSQL ────────────────────────────────────────
    database_url: str = (
        "postgresql+asyncpg://yeoju_admin:password@postgres:5432/yeoju_water"
    )

    # ── Ollama ────────────────────────────────────────────
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "qwen3:8b"
    ollama_timeout: int = 120       # 초
    ollama_max_tokens: int = 2048

    # ── LLM Service ───────────────────────────────────────
    llm_service_host: str = "0.0.0.0"
    llm_service_port: int = 8002
    llm_service_workers: int = 2

    # ── 연동 서비스 ───────────────────────────────────────
    predictor_base_url: str = "http://predictor:8001"

    # ── 여주보 기본값 ─────────────────────────────────────
    yeoju_station_id: str = "3008680"

    # ── 로깅 ─────────────────────────────────────────────
    log_level: str = "INFO"

    # ── 프롬프트 버전 ──────────────────────────────────────
    prompt_version: str = "v1.0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """싱글턴으로 설정 객체를 반환한다. FastAPI Depends에서 사용."""
    return Settings()
