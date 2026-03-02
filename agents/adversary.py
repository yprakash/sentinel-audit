import logging

from agents.prompts import ADVERSARY_AGENT_PROMPT
from state import AuditState
from utils.llm import BaseLLM
from utils.prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


async def adversary_agent(state: AuditState, llm: BaseLLM):
    prompt = render_prompt(
        ADVERSARY_AGENT_PROMPT,
        invariants=state.invariants,
        raw_code=state.raw_code,
    )

    logger.info("Adversary agent LLM call initiated")
    response = await llm.generate(prompt)

    return {
        "findings": [response]
    }
