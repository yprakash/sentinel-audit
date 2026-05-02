import json
import logging
from pathlib import Path
from typing import Any

from llm_registry import LLMRegistry
from utils.llm import BaseLLM, shutdown_llm_client

logger = logging.getLogger(__name__)


class MockClient(BaseLLM):
    """
    Deterministic LLM stub for development/testing.

    Loads a pre-recorded JSON response from disk and returns it
    exactly as a real LLM client would (parsed domain dict).
    """
    def __init__(self, model: str, agent_role: str) -> None:
        mock_file = f"mocked_outputs/{agent_role}.json"
        self._mock_path = Path(mock_file)
        if self._mock_path.exists():
            logger.info("MockClient reads LLM output from %s", self._mock_path)
        else:
            raise FileNotFoundError(f"Mock file not found: {self._mock_path}")

        super().__init__("mock", model, agent_role)
        self.model = model

    async def shutdown(self) -> None:
        # await self.client.aclose()
        await shutdown_llm_client(None, self.active_tasks)

    async def _generate_impl(self, model: str, **kwargs) -> Any:
        # Ignores prompt content. Returns pre-recorded JSON.
        with self._mock_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info("Got Mock data from %s", self._mock_path)
        return data


LLMRegistry.register("mock", MockClient)
