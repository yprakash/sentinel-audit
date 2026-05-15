import logging
import os

from sentinel_audit.llm_outputs import StrategistOutput
from sentinel_audit.prompts import STRATEGIST_AGENT_PROMPT
from sentinel_audit.state import AuditState
from utils.llm import BaseLLM
from utils.prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


async def strategist_agent(state: AuditState, llm: BaseLLM):
    prompt = render_prompt(
        STRATEGIST_AGENT_PROMPT,
        raw_code=state.raw_code,
        # previous_findings=state.findings if state.findings else [],
    )

    logger.info("Strategist agent LLM call initiated")
    response = await llm.generate(
        "strategist",
        messages=[{"role": "system", "content": prompt}],
        top_p=0.8,
        temperature=0.1,  # Low reduces hallucinated invariants
        max_output_tokens=os.getenv("STRATEGIST_MAX_TOKENS", 1024),
        response_model=StrategistOutput,
    )

    return {
        "strategist_output": llm.get_ai_message_from_response(response),
        "current_agent": "adversary",
    }
