import asyncio
import logging
import os
from typing import Any, Dict

import httpx

from llm_registry import LLMRegistry
from utils.constants import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
from utils.llm import shutdown_llm_client, BaseLLM

logger = logging.getLogger(__name__)
_cf_client: httpx.AsyncClient | None = None


def get_cloudflare_client() -> httpx.AsyncClient:
    global _cf_client
    if _cf_client:
        return _cf_client

    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not api_token:
        raise RuntimeError("Missing CLOUDFLARE_API_TOKEN")

    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("Missing CLOUDFLARE_ACCOUNT_ID")

    timeout = (
        httpx.Timeout(float(os.environ["CLOUDFLARE_TIMEOUT"]), connect=20.0)
        if "CLOUDFLARE_TIMEOUT" in os.environ
        else DEFAULT_TIMEOUT
    )

    _cf_client = httpx.AsyncClient(
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/",
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    return _cf_client


class CloudflareClient(BaseLLM):
    def __init__(self, model: str, agent_role: str) -> None:
        super().__init__("cloudflare", model, agent_role)
        self._client = get_cloudflare_client()
        self._max_retries = int(os.environ.get("CLOUDFLARE_MAX_RETRIES", DEFAULT_MAX_RETRIES))

    async def shutdown(self) -> None:
        await shutdown_llm_client(self._client, self.active_tasks)

    async def _generate_impl(self, model: str, **kwargs) -> Any:
        model = model or self.model
        messages = kwargs.get("messages")
        if not messages:
            raise ValueError("`messages` is required for Cloudflare requests")

        payload = {
            "messages": messages,
            **self._filter_supported_params(kwargs),
        }
        last_exception = None

        for attempt in range(self._max_retries):
            try:
                resp = await self._client.post(model, json=payload)

                if resp.status_code == 200:
                    data = resp.json()

                    normalized = self._normalize_response(data)
                    self.write_llm_output(normalized)
                    return normalized

                if resp.status_code == 400:
                    raise ValueError(f"Invalid request to Cloudflare: {resp.text}")
                if resp.status_code == 401:
                    raise RuntimeError("Invalid CLOUDFLARE_API_TOKEN")
                if resp.status_code == 403:
                    raise RuntimeError("Permission denied for this Cloudflare model")
                if resp.status_code == 404:
                    raise ValueError(f"Cloudflare model '{model}' not found")
                if resp.status_code == 429:
                    raise RuntimeError("Cloudflare rate limit exceeded")
                if resp.status_code >= 500:
                    raise RuntimeError(f"Cloudflare server error: {resp.status_code}")
                raise RuntimeError(f"Unexpected Cloudflare response: {resp.text}")

            except Exception as e:
                last_exception = e
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # exponential backoff
                else:
                    break

        raise RuntimeError("Cloudflare API request failed after retries") from last_exception

    def _filter_supported_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cloudflare does not support full OpenAI parameter surface.
        Filter only safe parameters.
        """
        allowed = {
            "temperature",
            "max_tokens",
            "top_p",
        }
        return {k: v for k, v in kwargs.items() if k in allowed}

    def _normalize_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = data.get("result", {})

        # Extract text (varies by model)
        text = ""
        if "response" in result:
            text = result["response"]
        elif "output_text" in result:
            text = result["output_text"]

        normalized = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": text,
                    }
                }
            ],
            "usage": {
                # Cloudflare usually does not return token usage
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        return normalized


LLMRegistry.register("cloudflare", CloudflareClient)


async def main() -> None:
    model_name = os.environ.get(
        "CLOUDFLARE_MODEL_NAME",
        "@cf/meta/llama-3-8b-instruct",
    )

    llm = CloudflareClient(model=model_name, agent_role="healthcheck")

    response = await llm.generate(
        messages=[{"role": "user", "content": "Say OK"}],
        temperature=0.0,
        max_tokens=16,
    )

    print(response)


if __name__ == "__main__":
    from getpass import getpass

    if "CLOUDFLARE_API_TOKEN" not in os.environ:
        os.environ["CLOUDFLARE_API_TOKEN"] = getpass("Enter CLOUDFLARE_API_TOKEN: ")
    if "CLOUDFLARE_ACCOUNT_ID" not in os.environ:
        os.environ["CLOUDFLARE_ACCOUNT_ID"] = input("Enter CLOUDFLARE_ACCOUNT_ID: ")

    asyncio.run(main())
