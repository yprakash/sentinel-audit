def build_async_agent_node(agent_fn, llm):
    async def node(state):
        return await agent_fn(state, llm)

    return node
