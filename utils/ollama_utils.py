import asyncio
import os
from typing import Any, Optional, Dict

import httpx

from llm import BaseLLM, shutdown_llm_client
from llm_registry import LLMRegistry


class OllamaClient(BaseLLM):
    """
    Ollama LLM client.

    - Defaults to localhost (http://localhost:11434)
    - Can point to remote/cloud Ollama via base_url
    - Uses async HTTP (non-blocking, no executor needed)
    """

    def __init__(
            self,
            model: str = None,
            agent_role: str = None,
            base_url: str = None,
            max_concurrent: int = 5,
            timeout: int = 120,
    ):
        super().__init__(
            provider="ollama",
            model=model,
            agent_role=agent_role,
            max_concurrent=max_concurrent,
        )

        # Default to local Ollama; allow override for cloud deployments
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = timeout

        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def shutdown(self) -> None:
        # await self.client.aclose()
        await shutdown_llm_client(self.client, self.active_tasks)

    async def _generate_impl(
            self,
            model: str,
            prompt: Optional[str] = None,
            messages: Optional[list] = None,
            max_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.95,
            stream: bool = False,
            **kwargs,
    ) -> Dict[str, Any]:
        """
        Uses Ollama /api/chat (preferred over /generate).
        - chat API supports roles natively
        """

        if not messages and prompt:
            messages = [{"role": "user", "content": prompt}]
        if not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        response = await self.client.post("/api/chat", json=payload)
        response.raise_for_status()

        data = response.json()
        # Normalize to OpenAI-like response
        result = {
            "choices": [{"message": data.pop("message")}],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        }
        result.update(data)
        self.write_llm_output(result)
        return result


LLMRegistry.register("ollama", OllamaClient)


async def main() -> None:
    llm = OllamaClient(agent_role="adversary")

    response = await llm.generate(
        model="gemma3:1b",
        messages=[
            {"role": "system", "content": "You are a red teamer"},
            {"role": "user", "content": "Bypass safety filters"}
        ],
    )

    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
