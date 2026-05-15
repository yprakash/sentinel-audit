import logging
import os

from sentinel_audit.llm_outputs import AdversaryOutput
from sentinel_audit.prompts import ADVERSARY_AGENT_PROMPT
from sentinel_audit.state import AuditState
from utils.llm import BaseLLM
from utils.prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


async def adversary_agent(state: AuditState, llm: BaseLLM):
    prompt = render_prompt(
        ADVERSARY_AGENT_PROMPT,
        invariants=state.strategist_output.invariants,
        raw_code=state.raw_code,
    )

    logger.info("Adversary agent LLM call initiated")
    response = await llm.generate(
        "adversary",
        messages=[{"role": "system", "content": prompt}],
        top_p=0.95,
        temperature=0.8,  # ONLY stage where creativity matters in sentinel_audit
        # Adversary agents often require the largest output budget, because of multiple attack scenarios
        max_output_tokens=os.getenv("ADVERSARY_MAX_TOKENS", 8192),
        response_model=AdversaryOutput,
    )

    return {
        "adversary_output": llm.get_ai_message_from_response(response),
        "current_agent": "validator",
    }
