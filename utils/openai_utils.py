import asyncio
import logging
import os
from typing import Any

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

from llm_registry import LLMRegistry
from utils.constants import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
from utils.llm import shutdown_llm_client, BaseLLM

logger = logging.getLogger(__name__)

_openai_client = None


def get_openai_client():
    global _openai_client
    if not _openai_client:
        openai_params = {
            "max_retries": os.environ["OPENAI_MAX_RETRIES"] if \
                "OPENAI_MAX_RETRIES" in os.environ else DEFAULT_MAX_RETRIES
        }
        if "OPENAI_BASE_URL" in os.environ:
            openai_params["base_url"] = os.environ["OPENAI_BASE_URL"]
        if "OPENAI_TIMEOUT" in os.environ:
            openai_params["timeout"] = httpx.Timeout(
                float(os.environ["OPENAI_TIMEOUT"]),
                connect=10.0
            )
        else:
            openai_params["timeout"] = DEFAULT_TIMEOUT

        _openai_client = AsyncOpenAI(**openai_params)
        logger.info(f"Openai client created")
    return _openai_client


class OpenAIClient(BaseLLM):
    def __init__(self, model, agent_role) -> None:
        super().__init__("openai", model, agent_role)
        self._client = get_openai_client()  # ensure shared client usage

    async def shutdown(self) -> None:
        await shutdown_llm_client(self._client, self.active_tasks)

    async def _generate_impl(self, model: str, **kwargs) -> Any:
        """
        Generate response using OpenAI Responses API.

        Expected normalized kwargs from BaseLLM.generate():
            - input: list[dict]
            - temperature: float (optional)
            - max_tokens: int (optional)
            - top_p: float (optional)
            - stream: bool (optional)

        Returns:
            Raw OpenAI SDK response object.
        """
        try:
            if "messages" in kwargs:
                kwargs["input"] = kwargs.pop("messages")
            if "max_tokens" in kwargs:
                kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

            response = await self._client.responses.create(
                model=model,
                **kwargs,
            )
            if response:
                response = response.model_dump(mode="json")
                self.write_llm_output(response)
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


LLMRegistry.register("openai", OpenAIClient)


# Below main methods are just to test connection. Can't be used anywhere
async def main() -> None:
    model_name = "gpt-4o-mini"  # input("Model: ")
    llm = OpenAIClient(model=model_name, agent_role="healthcheck")

    try:
        response = await llm.generate(
            messages=[
                {"role": "user", "content": "Say OK"}
            ],
            temperature=0.0,
            max_tokens=16,
        )
        print(response["choices"][0]["message"]["content"])
    except RuntimeError as re:
        models = await llm.get_available_models()
        print(f"Available models: {models}")
        if isinstance(re.__cause__, RateLimitError):
            print("RateLimitError (but connection is OK)")
        else:
            print("Other runtime error:", re)


if __name__ == "__main__":
    from getpass import getpass

    os.environ["OPENAI_API_KEY"] = getpass(f"Enter OPENAI_API_KEY: ")
    asyncio.run(main())
