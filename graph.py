import logging
import os

from langgraph.checkpoint.memory import MemorySaver
# from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import StateGraph, START, END

from agents.adversary import adversary_agent
from agents.reporter import reporter_agent
from agents.strategist import strategist_agent
from agents.validator import validator_agent
from config import settings
from llm_factory import create_llm
from state import AuditState

logger = logging.getLogger(__name__)

"""
Graph wiring layer.

Responsibilities:
- Instantiate LLMs (via factory)
- Inject dependencies into agents
- Configure checkpointing
- Compile graph
"""

REDIS_URL = os.getenv("REDIS_URL", "localhost:6379")
_graph_instance = None


def get_graph():
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = _create_graph()
    return _graph_instance


def build_async_agent_node(agent_fn, llm):
    async def node(state: AuditState):
        return await agent_fn(state, llm)
    return node


def _create_graph():  # -> CompiledGraph:
    # 1. Create LLM Instances (once, nowhere else in this app)
    strategist_node_name = "strategist"
    strategist_llm = create_llm(
        provider=settings.STRATEGIST_PROVIDER,
        model=settings.STRATEGIST_MODEL,
        agent_role=strategist_node_name,
    )

    adversary_node_name = "adversary"
    adversary_llm = create_llm(
        provider=settings.ADVERSARY_PROVIDER,
        model=settings.ADVERSARY_MODEL,
        agent_role=adversary_node_name,
    )

    validator_node_name = "validator"
    validator_llm = create_llm(
        provider=settings.VALIDATOR_PROVIDER,
        model=settings.VALIDATOR_MODEL,
        agent_role=validator_node_name,
    )

    reporter_node_name = "reporter"
    reporter_llm = create_llm(
        provider=settings.REPORTER_PROVIDER,
        model=settings.REPORTER_MODEL,
        agent_role=reporter_node_name,
    )

    # 2. Build Graph
    builder = StateGraph(AuditState)

    # Dependency injection via lambda
    builder.add_node(strategist_node_name, build_async_agent_node(strategist_agent, strategist_llm))
    builder.add_node(adversary_node_name, build_async_agent_node(adversary_agent, adversary_llm))
    builder.add_node(validator_node_name, build_async_agent_node(validator_agent, validator_llm))
    builder.add_node(reporter_node_name, build_async_agent_node(reporter_agent, reporter_llm))

    builder.add_edge(START, strategist_node_name)
    builder.add_edge(strategist_node_name, adversary_node_name)
    builder.add_edge(adversary_node_name, validator_node_name)
    builder.add_edge(validator_node_name, reporter_node_name)
    builder.add_edge(reporter_node_name, END)

    graph = builder.compile(
        checkpointer=MemorySaver(),
        # checkpointer=redis_checkpointer,
        # interrupt_before=[strategist_node_name, adversary_node_name, validator_node_name, reporter_node_name],
    )
    logger.info(
        "Sentinel graph compiled successfully.",
        extra={
            "node_count": len(builder.nodes),
            "edge_count": len(builder.edges),
        },
    )

    return graph
