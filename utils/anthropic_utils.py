from typing import Any, Dict

import httpx
from anthropic import (
    AsyncAnthropic,
    APIConnectionError,
    APIStatusError,
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
    NotFoundError,
    AnthropicError,
)

from utils.constants import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
from utils.llm import BaseLLM


class AnthropicClient(BaseLLM):
    """
    Anthropic implementation of BaseLLM.

    Responsibilities:
    - Own Anthropic SDK client lifecycle
    - Translate normalized BaseLLM.generate(...) arguments
    - Extract token usage
    - Map Anthropic-specific exceptions to normalized runtime errors

    Shared logic (metrics, timing, orchestration) lives in BaseLLM.
    """

    def __init__(
            self,
            *,
            api_key: str | None = None,
            timeout: float | None = None,
            max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """
        Construct a new async Anthropic client instance.

        If api_key is not provided, it will be inferred from
        the ANTHROPIC_API_KEY environment variable by the SDK.

        Args:
            api_key: Optional explicit API key.
            timeout: Request timeout in seconds.
            max_retries: Maximum automatic retries by SDK.
        """
        self._client = AsyncAnthropic(
            api_key=api_key,
            max_retries=max_retries,
            timeout=httpx.Timeout(timeout, connect=10.0) if timeout and timeout > 0 else DEFAULT_TIMEOUT,
        )

        super().__init__(provider="anthropic")

    async def _generate_impl(self, model: str, **kwargs) -> Any:
        """
        Provider-specific implementation using Anthropic Messages API.

        Expected normalized kwargs from BaseLLM.generate():
            - messages: list[dict]
            - max_tokens: int (required by Anthropic)
            - temperature: float (optional)
            - top_p: float (optional)
            - stream: bool (optional)
            - system: str (optional; Anthropic separates system prompt)

        NOTE:
        Anthropic differs from OpenAI/Groq:
        - Uses messages API (not chat.completions)
        - Requires max_tokens
        - System prompt is separate (system=...)
        """
        try:
            response = await self._client.messages.create(
                model=model,
                **kwargs,
            )
            return response
        except NotFoundError as e:
            raise ValueError(f"Model '{model}' does not exist or is not accessible in Anthropic") from e
        except BadRequestError as e:
            raise ValueError("Invalid request parameters sent to Anthropic") from e
        except AuthenticationError as e:
            raise RuntimeError("Invalid or missing Anthropic API key") from e
        except PermissionDeniedError as e:
            raise RuntimeError("API key does not have permission for this model") from e
        except RateLimitError as e:
            raise RuntimeError("Anthropic rate limit exceeded") from e
        except APIConnectionError as e:
            raise RuntimeError("Network / connection error while calling Anthropic") from e
        except APIStatusError as e:
            raise RuntimeError(f"Anthropic API returned error status {e.status_code}") from e
        except AnthropicError as e:
            raise RuntimeError("Unexpected Anthropic API error") from e
        except Exception as e:
            raise RuntimeError("Unknown error occurred while calling Anthropic") from e

    def extract_usage(self, response: Any) -> Dict[str, int]:
        """
        Extract token usage from Anthropic response.

        Anthropic usage format:
            response.usage.input_tokens
            response.usage.output_tokens
        """
        usage = getattr(response, "usage", None)

        if not usage:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        prompt_tokens = getattr(usage, "input_tokens", 0)
        completion_tokens = getattr(usage, "output_tokens", 0)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
