import logging
from typing import Any, Dict

import httpx
from groq import (
    AsyncGroq,
    APIConnectionError,
    APIStatusError,
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
    NotFoundError,
)

from utils.constants import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
from utils.llm import BaseLLM

logger = logging.getLogger(__name__)

class GroqClient(BaseLLM):
    """
    Groq implementation of BaseLLM. Only implements provider logic.

    Responsibilities:
    - Own Groq SDK client lifecycle
    - Translate normalized BaseLLM.generate(...) arguments
    - Extract token usage
    """
    def __init__(
            self,
            *,
            timeout: float | None = None,
            max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """
        Construct a new async Groq client instance.
        Args:
            timeout: Request timeout in seconds.
            max_retries: Maximum automatic retries by SDK.
        """
        self._client = AsyncGroq(
            max_retries=max_retries,
            timeout=httpx.Timeout(timeout, connect=10.0) if timeout and timeout > 0 else DEFAULT_TIMEOUT,
        )

        super().__init__(provider="groq")

    async def _generate_impl(self, model: str, **kwargs) -> Dict[str, Any]:
        """
        Provider-specific implementation for text/chat generation.

        Expected normalized kwargs from BaseLLM.generate():
            - messages: list[dict]
            - temperature: float (optional)
            - max_tokens: int (optional)
            - top_p: float (optional)
            - stream: bool (optional)

        Returns:
            Raw Groq SDK response object (converted to dict if needed).
        """
        try:
            response = await self._client.chat.completions.create(
                model=model,
                **kwargs,
            )
            return response

        except NotFoundError as e:
            raise ValueError(f"Model '{model}' does not exist or is not accessible in Groq") from e
        except BadRequestError as e:
            raise ValueError("Invalid request parameters sent to Groq") from e
        except AuthenticationError as e:
            raise RuntimeError("Invalid or missing GROQ API key") from e
        except PermissionDeniedError as e:
            raise RuntimeError("API key does not have permission for this Groq model") from e
        except RateLimitError as e:
            raise RuntimeError("Groq rate limit exceeded") from e
        except APIConnectionError as e:
            raise RuntimeError("Network / connection error while calling Groq") from e
        except APIStatusError as e:
            raise RuntimeError(f"Groq API returned error status {e.status_code}") from e
        except Exception as e:
            raise RuntimeError("Unexpected Groq API error") from e  # Future-proof catch-all

    def extract_usage(self, response: Any) -> Dict[str, int]:
        """
        Extract token usage from Groq response.
        BaseLLM calls this after _generate_impl to update metrics.
        Returns:
            {
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int
            }
        """
        usage = getattr(response, "usage", None)

        if not usage:
            logger.warning(f"Groq API returned empty usage response: {response}")
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
