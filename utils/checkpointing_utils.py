import asyncio
import logging
import os
from typing import List, Any, Callable, Awaitable, Dict

import psycopg
import redis.asyncio as redis
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.redis import RedisSaver
from pgvector.psycopg import register_vector_async

from utils.kafka_checkpointing import AsyncKafkaSaver

logger = logging.getLogger(__name__)


async def create_redis_checkpointer() -> RedisSaver:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    client = redis.from_url(redis_url, decode_responses=False)

    try:
        await client.ping()  # health check
    except Exception as e:
        logger.error("Redis connection failed: %s", e)
        raise

    cp = RedisSaver(client)
    logger.info("RedisSaver created from %s", redis_url)
    return cp


async def create_postgres_checkpointer():
    pg_url = os.getenv(
        "POSTGRES_URL",
        "postgresql://postgres:postgres@localhost:5432/postgres",
    )
    # if not asyncpg:
    #     raise RuntimeError("asyncpg not installed")
    # conn = await asyncpg.connect(pg_url)

    cp = await AsyncPostgresSaver.from_conn_string(pg_url)
    logger.info("PostgreSQL checkpointer created from %s", pg_url)
    return cp


async def create_pgvector_checkpointer():
    pg_url = os.getenv(
        "POSTGRES_VECTOR_URL",
        os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/postgres"),
    )

    conn = await psycopg.AsyncConnection.connect(pg_url)
    await register_vector_async(conn)  # REQUIRED for pgvector

    cp = await AsyncPostgresSaver.from_conn(conn)
    logger.info("PGVector checkpointer created from %s", pg_url)
    return cp


async def create_memory_checkpointer() -> MemorySaver:
    logger.warning("Using MemorySaver fallback")
    return MemorySaver()


async def create_kafka_checkpointer() -> AsyncKafkaSaver:
    logger.info("Using AsyncKafkaSaver")
    return AsyncKafkaSaver()


CHECKPOINTER_CREATORS: Dict[str, Callable[[], Awaitable[Any]]] = {
    "RedisSaver": create_redis_checkpointer,
    "PostgreSQL": create_postgres_checkpointer,
    "PostgresWithPGvector": create_pgvector_checkpointer,
    "MemorySaver": create_memory_checkpointer,
    "KafkaSaver": create_kafka_checkpointer,
}


async def get_checkpointers_async(cps: List[str]) -> List[Any]:
    tasks = []
    seen = set()

    for cp in cps:
        creator = CHECKPOINTER_CREATORS.get(cp)

        if not creator:
            logger.warning("Unknown checkpointer: %s, using MemorySaver", cp)
            cp = "MemorySaver"
            creator = CHECKPOINTER_CREATORS.get(cp)

        if cp not in seen:
            seen.add(cp)
            tasks.append(creator())

    results = await asyncio.gather(*tasks, return_exceptions=True)

    final = []
    for res in results:
        if isinstance(res, Exception):
            logger.error("Checkpointer init failed: %s", res)
            final.append(MemorySaver())
        else:
            final.append(res)

    if not final:
        logger.error("NO Checkpointers found from: %s", cps)
        final.append(MemorySaver())
    return final


def get_redis_checkpointer():
    """
    Sync-friendly version (no health check)
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    client = redis.from_url(redis_url, decode_responses=False)
    cp = RedisSaver(client)

    logger.info("RedisSaver created from %s", redis_url)
    return cp
