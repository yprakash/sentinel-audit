import logging

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
        messages=[{"role": "system", "content": prompt}],
        top_p=0.8,
        temperature=0.1,  # Low reduces hallucinated invariants
        max_output_tokens=4096,
    )
    content = llm.get_ai_message_from_response(response)

    return {
        "business_logic_summary": content.get("business_logic_summary"),
        "invariants": content.get("invariants", []),
        "assumptions": content.get("assumptions", []),
        "trust_boundaries": content.get("trust_boundaries", []),
    }
