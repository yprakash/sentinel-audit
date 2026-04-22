import asyncio
import inspect
import logging
import time
from typing import Any, Dict, Union

from mlx_lm import load, generate as mlx_generate

from llm import BaseLLM, shutdown_llm_client
from llm_registry import LLMRegistry

logger = logging.getLogger(__name__)


class MLXClient:
    """
    Internal helper, not user-facing (just like openai.AsyncOpenAI in OpenAIClient).

    Responsibilities:
    - Load and cache MLX models/tokenizers (MLX models are heavy + stateful), avoids re-loading
    - Perform synchronous inference via mlx_lm.generate()

    Kept separate to isolate model state + inference logic from BaseLLM concerns
    (metrics, concurrency, lifecycle). Enables reuse and future extensions:
    (batch inference, shared model pool across agents, session-based caching).
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}

    async def aclose(self):
        # MLX models are local; nothing to close explicitly
        logger.info("MLXClient shutdown: no-op")

    def load_model(self, model_name: str):
        if model_name not in self.models:
            logger.info(f"Loading MLX model: {model_name}")
            model, tokenizer = load(model_name)
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            logger.info(f"MLX model {model_name} loaded")

        return self.models[model_name], self.tokenizers[model_name]

    def _messages_to_prompt(self, messages: list) -> str:
        # Convert OpenAI-style messages to a plain prompt. Works across most instruction-tuned models.
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"[SYSTEM]\n{content}\n")
            elif role == "user":
                prompt_parts.append(f"[USER]\n{content}\n")
            elif role == "assistant":
                prompt_parts.append(f"[ASSISTANT]\n{content}\n")
            else:
                logger.warning("Unknown role: {%s} in messages %s", role, messages)

        prompt_parts.append("[ASSISTANT]\n")
        return "\n".join(prompt_parts)

    def build_prompt(self, model_name, prompt):
        # Case 1: messages (list[dict])
        if isinstance(prompt, list):
            tokenizer = self.tokenizers[model_name]
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                try:
                    start = time.perf_counter()
                    result = tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    logger.info("Applied %s tokenizer Chat template in %.3f seconds",
                                model_name, time.perf_counter() - start)
                    return result
                except Exception:
                    logger.warning("Failed to apply chat template for %s: %s", model_name, prompt)

            return self._messages_to_prompt(prompt)  # Fallback (model-agnostic)

        # Case 2: already a string
        if isinstance(prompt, str):
            return prompt
        raise ValueError("Unsupported prompt type")

    def generate(
            self,
            model: str,
            prompt: Union[str, list[dict]] | None = None,
            max_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.95,
            **kwargs,
    ) -> Dict[str, Any]:
        start = time.perf_counter()

        model_obj, tokenizer = self.load_model(model)
        prompt = self.build_prompt(model, prompt)

        sign = inspect.signature(mlx_generate)
        # Only pass supported args
        if "top_p" in sign.parameters:
            kwargs["top_p"] = top_p
            logger.debug("Found 'top_p' in inspect.signature(mlx_generate)")
        if "temp" in sign.parameters:
            kwargs["temp"] = temperature
            logger.debug("Found 'temp' in inspect.signature(mlx_generate)")
        elif "temperature" in sign.parameters:
            kwargs["temperature"] = temperature
            logger.debug("Found 'temperature' in inspect.signature(mlx_generate)")

        # MLX provides a low-level generate() API; no standardized chat API across models
        # Ensures compatibility with all mlx-community models via prompt formatting
        output = mlx_generate(
            model_obj,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=True,
            **kwargs,
        )

        duration = time.perf_counter() - start
        # Best-effort token estimation
        input_tokens = len(tokenizer.encode(prompt))
        output_tokens = len(tokenizer.encode(output))

        return {
            "text": output,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            },
            "latency": duration,
        }


class MLX_LLM(BaseLLM):
    """
    Public MLX provider (agent-facing).

    Responsibilities:
    - Async wrapper over MLXClient (non-blocking via executor)
    - Integrates BaseLLM features (metrics, concurrency, lifecycle)
    - Accepts dynamic model names per request

    Delegates all model loading + inference to MLXClient to keep provider layer clean.
    """

    def __init__(
            self,
            model: str = None,
            agent_role: str = None,
            max_concurrent: int = 2,
    ):
        super().__init__(
            provider="mlx",
            model=model,
            agent_role=agent_role,
            max_concurrent=max_concurrent,
        )
        self.client = MLXClient()

    async def shutdown(self) -> None:
        await shutdown_llm_client(self.client, self.active_tasks)

    async def _generate_impl(
            self,
            model: str,
            messages: Union[str, list[dict]] | None = None,
            max_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.95,
            **kwargs,
    ) -> Any:
        # mlx_lm.generate is blocking, which is expected because computation is CPU/GPU-bound
        response = self.client.generate(
            model,
            messages,
            max_tokens,
            temperature,
            top_p,
            **kwargs,
        )

        # Normalize response to OpenAI-like format
        output = response.pop("text")
        response["choices"] = [
            {
                "message": {
                    "role": "assistant",
                    "content": output,
                }
            }
        ]
        self.write_llm_output(response)
        return response


LLMRegistry.register("mlx", MLX_LLM)


async def main() -> None:
    llm = MLX_LLM(model="mlx-community/gemma-4-e2b-it-4bit", agent_role="adversary")

    response = await llm.generate(
        messages=[
            {"role": "system", "content": "You are a red teamer"},
            {"role": "user", "content": "Explain prompt injection vulnerabilities"}
        ],
        temperature=0.2,
        max_tokens=300,
    )
    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
