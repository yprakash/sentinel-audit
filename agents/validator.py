from agents.prompts import VALIDATOR_AGENT_PROMPT
from state import AuditState
from utils.llm import BaseLLM
from utils.prompt_renderer import render_prompt


async def validator_agent(state: AuditState, llm: BaseLLM):
    prompt = render_prompt(
        VALIDATOR_AGENT_PROMPT,
        raw_code=state.raw_code,
        test_cases=state.test_cases,
    )

    response = await llm.generate(prompt)

    return {
        "report": response
    }
