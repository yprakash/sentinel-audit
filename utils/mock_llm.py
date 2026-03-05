import json
import logging
from pathlib import Path
from typing import Any, Dict

from llm_registry import LLMRegistry
from utils.llm import BaseLLM

logger = logging.getLogger(__name__)


class MockClient(BaseLLM):
    """
    Deterministic LLM stub for development/testing.

    Loads a pre-recorded JSON response from disk and returns it
    exactly as a real LLM client would (parsed domain dict).
    """

    def __init__(
            self,
            model: str,
            agent_role: str,
            # **kwargs: Any,
    ) -> None:
        mock_file = f"mocked_outputs/{agent_role}"
        self._mock_path = Path(mock_file + ".json")
        if self._mock_path.exists():
            self._mock_file = mock_file + ".json"
        else:
            self._mock_file = mock_file + ".txt"
            self._mock_path = Path(self.mock_file)

        if not self._mock_path.exists():
            raise FileNotFoundError(
                f"Mock file not found: {self._mock_path}"
            )

        logger.info(f"Using mock file: {self._mock_path}")
        super().__init__("mock", model, agent_role)
        self.model = model

    async def _generate_impl(self, model: str, **kwargs) -> Any:
        """
        Ignores prompt content. Returns pre-recorded JSON.
        """
        with self._mock_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def extract_usage(self, response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        print(f"type={type(response)}: {response}")

        if not usage:
            logger.warning(f"Mock Data returned empty usage response: {response}")
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


LLMRegistry.register("mock", MockClient)
