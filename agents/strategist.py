from agents.prompts import STRATEGIST_AGENT_PROMPT
from state import AuditState
from utils.llm import BaseLLM
from utils.prompt_renderer import render_prompt


async def strategist_agent(state: AuditState, llm: BaseLLM):
    prompt = render_prompt(
        STRATEGIST_AGENT_PROMPT,
        raw_code=state.get("raw_code"),
        # previous_findings=state.get("findings", []),
    )

    response = await llm.generate(prompt)

    return {
        "invariants": [response]
    }
