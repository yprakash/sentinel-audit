import logging
import os

from langgraph.checkpoint.redis import RedisSaver
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


def _create_graph():  # -> CompiledGraph:
    # 1. Create LLM Instances (once, nowhere else in this app)
    strategist_llm = create_llm(
        provider=settings.STRATEGIST_PROVIDER,
        model=settings.STRATEGIST_MODEL,
        temperature=settings.STRATEGIST_TEMPERATURE,
        max_tokens=settings.STRATEGIST_MAX_TOKENS,
    )
    adversary_llm = create_llm(
        provider=settings.ADVERSARY_PROVIDER,
        model=settings.ADVERSARY_MODEL,
        temperature=settings.ADVERSARY_TEMPERATURE,
        max_tokens=settings.ADVERSARY_MAX_TOKENS,
    )
    validator_llm = create_llm(
        provider=settings.VALIDATOR_PROVIDER,
        model=settings.VALIDATOR_MODEL,
        temperature=settings.VALIDATOR_TEMPERATURE,
        max_tokens=settings.VALIDATOR_MAX_TOKENS,
    )
    reporter_llm = create_llm(
        provider=settings.REPORTER_PROVIDER,
        model=settings.REPORTER_MODEL,
        temperature=settings.REPORTER_TEMPERATURE,
        max_tokens=settings.REPORTER_MAX_TOKENS,
    )

    strategist_node_name = "strategist"
    adversary_node_name = "adversary"
    validator_node_name = "validator"
    reporter_node_name = "reporter"

    # 2. Build Graph
    builder = StateGraph(AuditState)

    # Dependency injection via lambda
    builder.add_node(strategist_node_name, lambda state: strategist_agent(state, llm=strategist_llm))
    builder.add_node(adversary_node_name, lambda state: adversary_agent(state, llm=adversary_llm))
    builder.add_node(validator_node_name, lambda state: validator_agent(state, llm=validator_llm))
    builder.add_node(reporter_node_name, lambda state: reporter_agent(state, llm=reporter_llm))

    builder.add_edge(START, strategist_node_name)
    builder.add_edge(strategist_node_name, adversary_node_name)
    builder.add_edge(adversary_node_name, validator_node_name)
    builder.add_edge(validator_node_name, reporter_node_name)
    builder.add_edge(reporter_node_name, END)

    # Redis Stack must have been started by now in terminal `redis-stack-server`
    with RedisSaver.from_conn_string(REDIS_URL) as _checkpointer:
        _checkpointer.setup()  # performs one-time initialization in Redis:
        # Creates RediSearch index (via FT.CREATE), Ensures schema exists, Prepares storage structures
        # Without calling setup(), Redis won’t have required indices and queries will fail.
        redis_checkpointer = _checkpointer

    graph = builder.compile(
        checkpointer=redis_checkpointer,
        interrupt_before=[strategist_node_name, adversary_node_name, validator_node_name, reporter_node_name],
    )
    logger.info(
        "Sentinel graph compiled",
        extra={
            "node_count": len(builder.nodes),
            "edge_count": len(builder.edges),
        },
    )

    return graph
