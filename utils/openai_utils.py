import asyncio
import logging
import os
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

logger = logging.getLogger(__name__)

openai_params = {
    "max_retries": os.environ["OPENAI_MAX_RETRIES"] if "OPENAI_MAX_RETRIES" in os.environ else DEFAULT_MAX_RETRIES
}
if "OPENAI_BASE_URL" in os.environ:
    openai_params["base_url"] = os.environ["OPENAI_BASE_URL"]
if "OPENAI_TIMEOUT" in os.environ:
    openai_params["timeout"] = httpx.Timeout(float(os.environ["OPENAI_TIMEOUT"]), connect=10.0)
else:
    openai_params["timeout"] = DEFAULT_TIMEOUT

openai_client = AsyncOpenAI(**openai_params)
active_tasks: set[asyncio.Task] = set()


async def shutdown():
    if active_tasks:  # wait for in-flight LLM calls
        logger.info(f"Waiting {len(active_tasks)} openai active tasks")
        await asyncio.gather(*active_tasks, return_exceptions=True)

    if openai_client:
        logger.info("Closing AsyncOpenAI client...")
        await openai_client.close()


class OpenAIClient(BaseLLM):
    def __init__(
            self,
            model,
            agent_role,
    ) -> None:
        super().__init__("openai", model, agent_role)

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
        task = asyncio.current_task()
        active_tasks.add(task)
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
        finally:
            active_tasks.discard(task)

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
