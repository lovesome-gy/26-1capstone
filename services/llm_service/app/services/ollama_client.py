"""
services/ollama_client.py

Ollama /api/chat 엔드포인트를 감싸는 비동기 클라이언트.
연결 실패, 타임아웃, 빈 응답에 대한 예외 처리를 포함한다.
"""

import logging
import time
from dataclasses import dataclass

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class LLMResponse:
    """LLM 응답 결과 컨테이너."""
    content: str
    model: str
    generation_ms: int


class OllamaClient:
    """
    Ollama HTTP API 비동기 클라이언트.

    사용 예시:
        client = OllamaClient()
        response = await client.chat(system_prompt="...", user_prompt="...")
        print(response.content)
    """

    def __init__(self):
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_model
        self._timeout = settings.ollama_timeout
        self._max_tokens = settings.ollama_max_tokens

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """
        Ollama /api/chat 를 호출하고 LLMResponse를 반환한다.

        Args:
            system_prompt: 역할과 출력 형식을 정의하는 시스템 메시지.
            user_prompt: 실제 데이터와 요청을 담은 유저 메시지.
            temperature: 생성 다양성 (보고서는 0.3, 창의적 내용은 높게).

        Raises:
            OllamaConnectionError: Ollama 서버에 연결할 수 없을 때.
            OllamaTimeoutError: 응답 시간이 초과됐을 때.
            OllamaEmptyResponseError: 빈 응답이 반환됐을 때.
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": self._max_tokens,
            },
        }

        start_ms = int(time.time() * 1000)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()

        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"Ollama 서버에 연결할 수 없습니다: {self._base_url}"
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(
                f"Ollama 응답 타임아웃 ({self._timeout}초 초과)"
            ) from e
        except httpx.HTTPStatusError as e:
            raise OllamaConnectionError(
                f"Ollama HTTP 오류: {e.response.status_code}"
            ) from e

        generation_ms = int(time.time() * 1000) - start_ms

        data = resp.json()
        content: str = data.get("message", {}).get("content", "").strip()

        if not content:
            raise OllamaEmptyResponseError("Ollama가 빈 응답을 반환했습니다.")

        logger.debug(
            "Ollama 응답 완료 | model=%s | %dms | %d chars",
            self._model,
            generation_ms,
            len(content),
        )

        return LLMResponse(
            content=content,
            model=self._model,
            generation_ms=generation_ms,
        )

    async def is_available(self) -> bool:
        """Ollama 서버 가용성을 체크한다."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def ensure_model_pulled(self) -> None:
        """모델이 로컬에 없으면 pull 요청을 보낸다 (최초 1회)."""
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                logger.info("Ollama 모델 확인 중: %s", self._model)
                resp = await client.post(
                    f"{self._base_url}/api/pull",
                    json={"name": self._model, "stream": False},
                )
                resp.raise_for_status()
                logger.info("Ollama 모델 준비 완료: %s", self._model)
        except Exception as e:
            logger.warning("모델 pull 실패 (이미 존재할 수 있음): %s", e)


# ── 예외 클래스 ───────────────────────────────────────────────
class OllamaConnectionError(RuntimeError):
    """Ollama 서버 연결 실패."""


class OllamaTimeoutError(RuntimeError):
    """Ollama 응답 타임아웃."""


class OllamaEmptyResponseError(RuntimeError):
    """Ollama 빈 응답."""
