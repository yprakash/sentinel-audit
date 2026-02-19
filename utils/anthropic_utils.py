import asyncio
import logging
import os
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

logger = logging.getLogger(__name__)

anthropic_params = {
    "max_retries": os.environ["ANTHROPIC_MAX_RETRIES"] if "ANTHROPIC_MAX_RETRIES" in os.environ else DEFAULT_MAX_RETRIES
}
if "ANTHROPIC_BASE_URL" in os.environ:
    anthropic_params["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
if "ANTHROPIC_TIMEOUT" in os.environ:
    anthropic_params["timeout"] = httpx.Timeout(float(os.environ["ANTHROPIC_TIMEOUT"]), connect=10.0)
else:
    anthropic_params["timeout"] = DEFAULT_TIMEOUT

anthropic_client = AsyncAnthropic(**anthropic_params)
active_tasks: set[asyncio.Task] = set()


async def shutdown():
    if active_tasks:  # wait for in-flight LLM calls
        logger.info(f"Waiting {len(active_tasks)} anthropic active tasks")
        await asyncio.gather(*active_tasks, return_exceptions=True)

    if anthropic_client:
        logger.info("Closing AsyncAnthropic client...")
        await anthropic_client.close()


class AnthropicClient(BaseLLM):
    def __init__(
            self,
            model,
            agent_role,
    ) -> None:
        super().__init__("anthropic", model, agent_role)

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
        task = asyncio.current_task()
        active_tasks.add(task)
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
        finally:
            active_tasks.discard(task)

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
