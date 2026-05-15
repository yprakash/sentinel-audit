import logging
import os

from sentinel_audit.llm_outputs import AuditReport
from sentinel_audit.prompts import REPORTER_AGENT_PROMPT
from sentinel_audit.state import AuditState
from utils.llm import BaseLLM
from utils.prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


async def reporter_agent(state: AuditState, llm: BaseLLM):
    prompt = render_prompt(
        REPORTER_AGENT_PROMPT,
        validated_findings=state.validator_output.model_dump_json(indent=2),
    )

    logger.info("Reporter agent LLM call initiated")

    # Reporter is a polished professional writer
    response = await llm.generate(
        "reporter",
        messages=[{"role": "system", "content": prompt}],
        top_p=0.9,
        temperature=0.4,
        max_output_tokens=os.getenv("REPORTER_MAX_TOKENS", 1024),
        response_model=AuditReport,
    )

    return {
        "final_report": llm.get_ai_message_from_response(response),
        "current_agent": "None",
    }
