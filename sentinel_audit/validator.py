import logging
import os

from sentinel_audit.llm_outputs import ValidatorOutput
from sentinel_audit.prompts import VALIDATOR_AGENT_PROMPT
from sentinel_audit.state import AuditState
from utils.llm import BaseLLM
from utils.prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


async def validator_agent(state: AuditState, llm: BaseLLM):
    prompt = render_prompt(
        VALIDATOR_AGENT_PROMPT,
        raw_code=state.raw_code,
        test_cases=state.adversary_output.model_dump_json(indent=2),
    )

    logger.info("Validator agent LLM call initiated")
    # Validator should behave almost like a compiler/interpreter.
    response = await llm.generate(
        "validator",
        messages=[{"role": "system", "content": prompt}],
        top_p=0.7,
        temperature=0.0,  # no creative interpretation
        max_output_tokens=os.getenv("VALIDATOR_MAX_TOKENS", 1024),
        response_model=ValidatorOutput,
    )

    return {
        "validator_output": llm.get_ai_message_from_response(response),
        "current_agent": "reporter",
    }
