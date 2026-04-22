import asyncio
import logging
import os
from typing import Any

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

from llm_registry import LLMRegistry
from utils.constants import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
from utils.llm import shutdown_llm_client, BaseLLM

logger = logging.getLogger(__name__)

_groq_client = None


def get_groq_client():
    global _groq_client
    if not _groq_client:
        groq_params = {
            "max_retries": os.environ["GROQ_MAX_RETRIES"] if "GROQ_MAX_RETRIES" in os.environ else DEFAULT_MAX_RETRIES
        }
        if "GROQ_BASE_URL" in os.environ:
            groq_params["base_url"] = os.environ["GROQ_BASE_URL"]
        if "GROQ_TIMEOUT" in os.environ:
            groq_params["timeout"] = httpx.Timeout(float(os.environ["GROQ_TIMEOUT"]), connect=10.0)
        else:
            groq_params["timeout"] = DEFAULT_TIMEOUT
        _groq_client = AsyncGroq(**groq_params)
    return _groq_client


class GroqClient(BaseLLM):
    def __init__(
            self,
            model,
            agent_role,
    ) -> None:
        super().__init__("groq", model, agent_role)
        self._client = get_groq_client()  # ensure shared client usage

    async def shutdown(self) -> None:
        await shutdown_llm_client(self._client, self.active_tasks)

    async def _generate_impl(self, model: str, **kwargs) -> Any:
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
            if response:
                response = response.model_dump(mode="json")
                self.write_llm_output(response)
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


LLMRegistry.register("groq", GroqClient)


# Below main methods are just to test connection. Can't be used anywhere
async def main() -> None:
    model_name = "llama-3.1-8b-instant"  # input("Model: ")
    llm = GroqClient(model=model_name, agent_role="healthcheck")
    response = await llm.generate(
        messages=[
            {"role": "user", "content": "Say OK"}
        ],
        temperature=0.0,
        max_tokens=16,
    )
    print(response)


if __name__ == "__main__":
    from getpass import getpass

    os.environ["GROQ_API_KEY"] = getpass(f"Enter GROQ_API_KEY: ")
    asyncio.run(main())
