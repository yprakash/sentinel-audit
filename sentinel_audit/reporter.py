import logging

from sentinel_audit.prompts import REPORTER_AGENT_PROMPT
from sentinel_audit.state import AuditState
from utils.llm import BaseLLM
from utils.prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


async def reporter_agent(state: AuditState, llm: BaseLLM):
    prompt = render_prompt(
        REPORTER_AGENT_PROMPT,
        validated_findings=state.validated_findings,
    )

    logger.info("Reporter agent LLM call initiated")

    # Reporter is a polished professional writer
    response = await llm.generate(
        prompt,
        top_p=0.9,
        temperature=0.4,
        max_output_tokens=8192,
    )

    return {
        "report": response
    }
