import asyncio
import logging
import os
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from pydantic_core import ValidationError

from llm_registry import LLMRegistry
from utils.llm import shutdown_llm_client, BaseLLM

logger = logging.getLogger(__name__)

_gemini_client = None
_gemini_supported_keys = {
    'temperature', 'top_p', 'top_k', 'max_output_tokens',
    'stop_sequences', 'safety_settings', 'response_mime_type',
    'response_schema', 'thinking_config', 'tools', 'candidate_count'
}
_retry_after_seconds = 3


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if not _gemini_client:
        timeout_ms = int(os.environ.get("GEMINI_TIMEOUT_MS", 120000))
        # The new SDK is more centralized. We create one client for all calls.
        # You can also pass vertexai=True here if using Google Cloud Vertex AI.

        if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" not in os.environ:
            import getpass
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter GOOGLE_API_KEY: ")

        _gemini_client = genai.Client(
            api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"),
            http_options=types.HttpOptions(
                timeout=timeout_ms,
            ),
        )
        logger.info("Initialized google.genai.Client instance")

    return _gemini_client


class GeminiClient(BaseLLM):
    def __init__(self, model: str = None) -> None:
        super().__init__("gemini", model)
        self._client = get_gemini_client()
        logger.info("Initialized GeminiClient with model: %s", self.model)

    def get_ai_message_from_response(self, response):
        if not response or "output" not in response:
            return None
        return response["output"]

    async def shutdown(self) -> None:
        """
        The google.genai SDK doesn't require a manual .close() for most async operations
        as it manages the connection pool, but we handle active tasks.
        """
        await shutdown_llm_client(None, self.active_tasks)

    def print_gemini_models(self):
        """Helper to list and print models on 404."""
        print(f"{'Model Name':<50} | {'Supported Actions'}")
        print("-" * 100)
        try:
            for model in self._client.models.list():
                actions = ", ".join(model.supported_actions)
                print(f"{model.name:<50} | {actions}")
        except Exception as e:
            print(f"Error fetching models: {e}")

    def print_token_limits(self, model: str):
        """Helper to print specific model info on 429."""
        try:
            m_info = self._client.models.get(model=model)
            print(f"Details for {model}:")
            print(f" - Input Limit: {m_info.input_token_limit}")
            print(f" - Output Limit: {m_info.output_token_limit}")
        except Exception:
            print(f"Model {model}: 0 (Hard Quota/Blocked)")

    async def _generate_impl(self, agent_role: str, model: str, **kwargs) -> Any:
        try:
            if "retries" not in kwargs:
                kwargs["retries"] = 3
            model = model or self.model
            messages = kwargs.get("messages", [])
            contents = []

            for m in messages:
                # 'assistant' is mapped to 'model' for Gemini
                role = "model" if m["role"] in ["assistant", "model"] else "user"  # m["role"]
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))

            gemini_config_dict = {k: v for k, v in kwargs.items() if k in _gemini_supported_keys}
            if "candidate_count" not in gemini_config_dict:
                gemini_config_dict["candidate_count"] = 1  # multiple candidates increase cost/tokens

            response_model = kwargs.get("response_model")
            if response_model:
                gemini_config_dict["response_schema"] = kwargs["response_model"]
                gemini_config_dict["response_mime_type"] = "application/json"
            else:
                logger.info("response_model is NOT present in %s kwargs for %s agent", self.provider, agent_role)

            config = types.GenerateContentConfig(**gemini_config_dict)

            response = await self._client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            if not response:
                raise Exception(f"Empty response from generate_content() with model: {model}")

            if response_model:
                try:
                    # parsed_output = response.parsed
                    parsed_output = response_model.model_validate_json(response.text.strip())
                except ValidationError as e:
                    logger.exception(
                        "Structured output validation failed. provider=%s model=%s schema=%s errors=%s",
                        self.provider,
                        model,
                        response_model.__name__,
                        e.errors(),
                    )
                    logger.error("Raw LLM output:\n%s", response.text)
                    raise

                assert isinstance(parsed_output, response_model)
            else:
                parsed_output = response.text

            logger.info("[Structured Output] provider=%s model=%s parsed=%s",
                        self.provider, model, type(parsed_output).__name__)

            # Convert Pydantic response to Dict for BaseLLM.extract_usage
            # This ensures compatibility with your existing logging/metrics logic
            standardized_res = {
                "provider": self.provider,
                "model": model,
                "agent_role": agent_role,
                "config": gemini_config_dict,
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "thinking_tokens": response.usage_metadata.thoughts_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                },
                "output": parsed_output,
                "response": response.model_dump(mode='json', exclude_unset=True),
            }

            self.write_llm_output(agent_role, standardized_res)
            return standardized_res

        # except ValidationError as e:
        #     kwargs["retries"] -= 1
        #     if kwargs["retries"] <= 0:
        #         raise
        #     repair_prompt = f"""
        #     Your previous response returned invalid/truncated JSON.
        #     Validation error:
        #     {str(e)}
        #     Return ONLY valid JSON.
        #     """
        #     # ReTry logic
        except ClientError as e:
            if e.code == 404:
                print(f"[{e.code}] Model '{model}' NOT FOUND. Fetching available models...")
                self.print_gemini_models()
            elif e.code == 429:
                logger.error("Error: %s", e)
                print(f"[{e.code}] RESOURCE_EXHAUSTED hit for {model}: {e.message}")
                self.print_token_limits(model)
            else:
                error_type = type(e).__name__
                logger.exception("Gemini API Error [%s]: ", error_type)
                raise e
        except ServerError as e:
            if e.code == 503:
                kwargs["retries"] -= 1
                if kwargs["retries"] > 0:
                    logger.error("Error with model %s: %s", model, e)
                    model = os.getenv('GEMINI_FALLBACK_MODEL', 'gemini-3.1-flash-lite')

                    await asyncio.sleep(_retry_after_seconds)
                    logger.warning("Retrying %s after %d seconds for %s agent with model %s",
                                   self.provider, _retry_after_seconds, agent_role, model)
                    return await self._generate_impl(agent_role, model, **kwargs)

            raise e
        except Exception as e:
            error_type = type(e).__name__
            logger.exception("Gemini API Error [%s]: ", error_type)
            raise
            # raise RuntimeError(f"Gemini call failed: {error_type}") from e


LLMRegistry.register("gemini", GeminiClient)


async def main() -> None:
    model_name = "gemini-2.5-flash"
    agent_role = "healthcheck"
    llm = GeminiClient(model=model_name)

    response = await llm.generate(
        agent_role,
        messages=[
            {"role": "user", "content": "Confirm system operational. Answer with 'OK'."}
        ],
        temperature=0.0
    )
    print(f"Response: {response['output']}")
    print(f"Usage: {response['usage']}")


if __name__ == "__main__":
    import getpass

    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter GOOGLE_API_KEY: ")
    asyncio.run(main())
