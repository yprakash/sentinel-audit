import logging

from sentinel_audit.prompts import VALIDATOR_AGENT_PROMPT
from sentinel_audit.state import AuditState
from utils.llm import BaseLLM
from utils.prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


async def validator_agent(state: AuditState, llm: BaseLLM):
    prompt = render_prompt(
        VALIDATOR_AGENT_PROMPT,
        raw_code=state.raw_code,
        test_cases=state.test_cases,
    )

    logger.info("Validator agent LLM call initiated")
    # Validator should behave almost like a compiler/interpreter.
    response = await llm.generate(
        prompt,
        top_p=0.7,
        temperature=0.0,  # no creative interpretation
        max_output_tokens=4096,
    )

    return {
        "report": response
    }
