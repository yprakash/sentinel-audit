from typing import Any, Dict

import httpx
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIStatusError,
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
    NotFoundError,
    OpenAIError,
)

from utils.constants import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
from utils.llm import BaseLLM


class OpenAIClient(BaseLLM):
    """
    OpenAI implementation of BaseLLM. Only implements provider logic.

    Responsibilities:
    - Own OpenAI SDK client lifecycle
    - Translate normalized BaseLLM.generate(...) arguments
    - Extract token usage

    All shared logic (metrics, timing, orchestration) is handled in BaseLLM.
    """

    def __init__(
            self,
            *,
            api_key: str | None = None,
            timeout: float | None = None,
            max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """
        Construct a new async OpenAI client instance.

        If api_key is not provided, it will be inferred from
        the OPENAI_API_KEY environment variable by the SDK.

        Args:
            api_key: Optional explicit API key.
            timeout: Request timeout in seconds.
            max_retries: Maximum automatic retries by SDK.
        """
        self._client = AsyncOpenAI(
            api_key=api_key,
            max_retries=max_retries,
            timeout=httpx.Timeout(timeout, connect=10.0) if timeout and timeout > 0 else DEFAULT_TIMEOUT,
        )

        super().__init__(provider="openai")

    async def _generate_impl(self, model: str, **kwargs) -> Any:
        """
        Provider-specific implementation for chat/text generation.

        Expected normalized kwargs from BaseLLM.generate():
            - messages: list[dict]
            - temperature: float (optional)
            - max_tokens: int (optional)
            - top_p: float (optional)
            - stream: bool (optional)

        Returns:
            Raw OpenAI SDK response object.
        """
        try:
            response = await self._client.chat.completions.create(
                model=model,
                **kwargs,
            )
            return response

        except NotFoundError as e:
            raise ValueError(f"Model '{model}' does not exist or is not accessible") from e
        except BadRequestError as e:
            raise ValueError("Invalid request parameters sent to OpenAI") from e
        except AuthenticationError as e:
            raise RuntimeError("Invalid or missing OpenAI API key") from e
        except PermissionDeniedError as e:
            raise RuntimeError("API key does not have permission for this model") from e
        except RateLimitError as e:
            raise RuntimeError("OpenAI rate limit exceeded") from e
        except APIConnectionError as e:
            raise RuntimeError("Network / connection error while calling OpenAI") from e
        except APIStatusError as e:
            raise RuntimeError(f"OpenAI API returned error status {e.status_code}") from e
        except OpenAIError as e:
            # Future-proof catch-all for SDK-specific errors
            raise RuntimeError("Unexpected OpenAI API error") from e
        except Exception as e:
            raise RuntimeError("Unknown error occurred while calling OpenAI") from e  # Final safety net

    def extract_usage(self, response: Any) -> Dict[str, int]:
        """
        Extract token usage from OpenAI response.
        Called by BaseLLM after _generate_impl to update metrics.
        Returns:
            {
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int
            }
        """
        usage = getattr(response, "usage", None)

        if not usage:
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
