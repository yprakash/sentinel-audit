import logging
import os
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
)
from openai import AsyncOpenAI

from llm_registry import LLMRegistry
from utils.constants import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
from utils.llm import shutdown_llm_client, BaseLLM

logger = logging.getLogger(__name__)
_nvidia_client = None


def get_nvidia_client():
    global _nvidia_client
    if _nvidia_client:
        return _nvidia_client

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("Missing NVIDIA_API_KEY")

    timeout = (
        httpx.Timeout(float(os.environ["NVIDIA_TIMEOUT"]), connect=60.0)
        if "NVIDIA_TIMEOUT" in os.environ
        else DEFAULT_TIMEOUT
    )
    max_retries = int(os.environ.get("NVIDIA_MAX_RETRIES", DEFAULT_MAX_RETRIES))
    base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

    _nvidia_client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    return _nvidia_client


class NvidiaClient(BaseLLM):
    def __init__(self, model: str, agent_role: str) -> None:
        super().__init__("nvidia", model, agent_role)
        self._client = get_nvidia_client()

    async def shutdown(self) -> None:
        await shutdown_llm_client(self._client, self.active_tasks)

    async def _generate_impl(self, model: str, **kwargs) -> Any:
        try:
            model = model or self.model
            response = await self._client.chat.completions.create(model=model, **kwargs)

            if response:
                response_dict = response.model_dump(mode="json")
                self.write_llm_output(response_dict)
                return response_dict

            return response

        except NotFoundError as e:
            raise ValueError(f"NVIDIA model '{model}' not found") from e
        except BadRequestError as e:
            raise ValueError("Invalid request sent to NVIDIA API") from e
        except AuthenticationError as e:
            raise RuntimeError("Invalid or missing NVIDIA API key") from e
        except PermissionDeniedError as e:
            raise RuntimeError("Permission denied for this NVIDIA model") from e
        except RateLimitError as e:
            raise RuntimeError("NVIDIA API rate limit exceeded") from e
        except APIConnectionError as e:
            raise RuntimeError("Network error while calling NVIDIA API") from e
        except APIStatusError as e:
            raise RuntimeError(f"NVIDIA API error: {e.status_code}") from e
        except Exception as e:
            raise RuntimeError("Unexpected NVIDIA API error") from e


LLMRegistry.register("nvidia", NvidiaClient)


async def main() -> None:
    model_name = os.environ.get("NVIDIA_MODEL_NAME", "meta/llama-3.1-8b-instruct")

    llm = NvidiaClient(model=model_name, agent_role="healthcheck")

    response = await llm.generate(
        messages=[{"role": "user", "content": "Say OK"}],
        temperature=0.0,
        max_tokens=16,
    )
    print(response)


if __name__ == "__main__":
    import asyncio
    from getpass import getpass

    if "NVIDIA_API_KEY" not in os.environ:
        os.environ["NVIDIA_API_KEY"] = getpass("Enter NVIDIA_API_KEY: ")

    asyncio.run(main())
