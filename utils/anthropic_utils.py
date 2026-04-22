import asyncio
import logging
import os
from typing import Any

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

from llm_registry import LLMRegistry
from utils.constants import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
from utils.llm import shutdown_llm_client, BaseLLM

logger = logging.getLogger(__name__)
_anthropic_client = None


def get_anthropic_client():
    global _anthropic_client
    if not _anthropic_client:
        anthropic_params = {
            "max_retries": os.environ["ANTHROPIC_MAX_RETRIES"] if \
                "ANTHROPIC_MAX_RETRIES" in os.environ else DEFAULT_MAX_RETRIES
        }
        if "ANTHROPIC_BASE_URL" in os.environ:
            anthropic_params["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
        if "ANTHROPIC_TIMEOUT" in os.environ:
            anthropic_params["timeout"] = httpx.Timeout(float(os.environ["ANTHROPIC_TIMEOUT"]), connect=10.0)
        else:
            anthropic_params["timeout"] = DEFAULT_TIMEOUT

        _anthropic_client = AsyncAnthropic(**anthropic_params)
    return _anthropic_client


class AnthropicClient(BaseLLM):
    def __init__(
            self,
            model,
            agent_role,
    ) -> None:
        super().__init__("anthropic", model, agent_role)
        self._client = get_anthropic_client()  # ensure shared client usage

    async def shutdown(self) -> None:
        await shutdown_llm_client(self._client, self.active_tasks)

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
            if response:
                self.write_llm_output(response.content[0].text)  # ToDo Correct the input param
            return response.content[0].text
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


LLMRegistry.register("anthropic", AnthropicClient)


# Below main methods are just to test connection. Can't be used anywhere
async def main() -> None:
    model_name = input("Model: ")  # "claude-haiku-4-5"
    llm = AnthropicClient(model=model_name, agent_role="healthcheck")

    response = await llm.generate(
        messages=[
            {"role": "user", "content": "Say OK"}
        ],
        temperature=0.0,
        max_tokens=10,
    )
    print(response)


if __name__ == "__main__":
    from getpass import getpass

    os.environ["ANTHROPIC_API_KEY"] = getpass(f"Enter ANTHROPIC_API_KEY: ")
    asyncio.run(main())
