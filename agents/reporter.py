from agents.prompts import REPORTER_AGENT_PROMPT
from state import AuditState
from utils.llm import BaseLLM
from utils.prompt_renderer import render_prompt


async def reporter_agent(state: AuditState, llm: BaseLLM):
    prompt = render_prompt(
        REPORTER_AGENT_PROMPT,
        validated_findings=state.validated_findings,
    )

    response = await llm.generate(prompt)

    return {
        "report": response
    }
