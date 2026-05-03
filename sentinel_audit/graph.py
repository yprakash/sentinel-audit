import logging

from langgraph.graph import StateGraph, START, END

from llm_factory import create_llm
from sentinel_audit.adversary import adversary_agent
from sentinel_audit.config import settings
from sentinel_audit.reporter import reporter_agent
from sentinel_audit.state import AuditState
from sentinel_audit.strategist import strategist_agent
from sentinel_audit.validator import validator_agent
from utils.agent_utils import build_async_agent_node
from utils.checkpointing_utils import get_checkpointers_async
from utils.composite_checkpoint_saver import CompositeCheckpointSaver

logger = logging.getLogger(__name__)

"""
Graph wiring layer.

Responsibilities:
- Instantiate LLMs (via factory)
- Inject dependencies into agents
- Configure checkpointing
- Compile graph
"""

_graph_instance = None


async def get_graph():
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = await _create_graph()
    return _graph_instance


async def _create_graph():  # -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:
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

    builder.add_node(strategist_node_name, build_async_agent_node(strategist_agent, strategist_llm))
    builder.add_node(adversary_node_name, build_async_agent_node(adversary_agent, adversary_llm))
    builder.add_node(validator_node_name, build_async_agent_node(validator_agent, validator_llm))
    builder.add_node(reporter_node_name, build_async_agent_node(reporter_agent, reporter_llm))

    builder.add_edge(START, strategist_node_name)
    builder.add_edge(strategist_node_name, adversary_node_name)
    builder.add_edge(adversary_node_name, validator_node_name)
    builder.add_edge(validator_node_name, reporter_node_name)
    builder.add_edge(reporter_node_name, END)

    ckpts: list = await get_checkpointers_async(settings.GRAPH_CHECKPOINTS.split(","))
    checkpointer = CompositeCheckpointSaver(ckpts)
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=[strategist_node_name, adversary_node_name, validator_node_name, reporter_node_name],
    )
    logger.info("Sentinel graph compiled successfully with %d nodes and %d edges.",
                len(builder.nodes), len(builder.edges)
    )

    return graph
