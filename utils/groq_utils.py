import asyncio
import logging
import os
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

groq_params = {
    "max_retries": os.environ["GROQ_MAX_RETRIES"] if "GROQ_MAX_RETRIES" in os.environ else DEFAULT_MAX_RETRIES
}
if "GROQ_BASE_URL" in os.environ:
    groq_params["base_url"] = os.environ["GROQ_BASE_URL"]
if "GROQ_TIMEOUT" in os.environ:
    groq_params["timeout"] = httpx.Timeout(float(os.environ["GROQ_TIMEOUT"]), connect=10.0)
else:
    groq_params["timeout"] = DEFAULT_TIMEOUT

groq_client = AsyncGroq(**groq_params)
active_tasks: set[asyncio.Task] = set()


async def shutdown():
    if active_tasks:  # wait for in-flight LLM calls
        logger.info(f"Waiting {len(active_tasks)} groq active tasks")
        await asyncio.gather(*active_tasks, return_exceptions=True)

    if groq_client:
        logger.info("Closing AsyncGroq client...")
        await groq_client.close()


class GroqClient(BaseLLM):
    def __init__(
            self,
            model,
            agent_role,
    ) -> None:
        super().__init__("groq", model, agent_role)

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
        task = asyncio.current_task()
        active_tasks.add(task)
        try:
            response = await self._client.chat.completions.create(
                model=model,
                **kwargs,
            )
            return response.model_dump()

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
        finally:
            active_tasks.discard(task)

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
