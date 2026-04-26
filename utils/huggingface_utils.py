import asyncio
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_registry import LLMRegistry
from utils.llm import BaseLLM, shutdown_llm_client

logger = logging.getLogger(__name__)


class HFClient(BaseLLM):
    def __init__(self, model: str, agent_role: str, max_new_tokens: int = 1000):
        if not model:
            raise ValueError("model is required to use HFClient")
        if not agent_role:
            raise ValueError("agent_role is required to use HFClient")

        super().__init__("huggingface", model, agent_role)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.max_new_tokens = max_new_tokens

    async def shutdown(self) -> None:
        await shutdown_llm_client(None, self.active_tasks)

    def _sync_generate(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _messages_to_prompt(self, messages: list) -> str:
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

    async def _generate_impl(self, model: str, **kwargs):
        messages = kwargs.get("messages")
        # prompt = "\n\n".join([m["content"] for m in messages])
        prompt = self._messages_to_prompt(messages)

        output = await asyncio.to_thread(self._sync_generate, prompt)
        return {
            "content": output
        }

    def extract_usage(self, response):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


LLMRegistry.register("huggingface", HFClient)


async def main():
    model_name = "Qwen/Qwen3-0.6B"  # OR "tiny-gpt2"

    llm = HFClient(
        model=model_name,
        agent_role="test-agent",
        max_new_tokens=100,
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain async programming in Python in 3 lines."},
    ]
    response = await llm._generate_impl(
        model=model_name,
        messages=messages
    )
    print("\n=== Response: ", response["content"])


if __name__ == "__main__":
    asyncio.run(main())
