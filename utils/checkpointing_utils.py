import asyncio
import logging
import os
import time
from typing import List, Any, Callable, Awaitable, Dict
from urllib.parse import urlparse

import psycopg
import redis.asyncio as redis
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from utils.kafka_checkpointing import AsyncKafkaSaver

logger = logging.getLogger(__name__)


async def create_redis_checkpointer() -> AsyncRedisSaver:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    try:
        client = redis.from_url(redis_url)
        await client.ping()  # health check
    except Exception as e:
        logger.error("Redis connection failed: %s", e)
        raise

    cp = AsyncRedisSaver(redis_url)
    logger.info("AsyncRedisSaver created from %s", redis_url)
    return cp


async def create_memory_checkpointer() -> MemorySaver:
    return MemorySaver()


async def create_kafka_checkpointer() -> AsyncKafkaSaver:
    return AsyncKafkaSaver()


async def get_postgres_checkpointer():
    start = time.perf_counter()
    pg_url = os.getenv(
        "POSTGRES_URL",
        "postgresql://postgres:postgres@localhost:5432/postgres",
    )

    conn = await psycopg.AsyncConnection.connect(pg_url)
    await conn.set_autocommit(True)  # REQUIRED for setup()

    cp = AsyncPostgresSaver(conn)
    await cp.setup()  # create tables

    parsed = urlparse(pg_url)
    # Reconstruct without password
    safe_uri = f"{parsed.scheme}://{parsed.username}:****@{parsed.hostname}:{parsed.port}{parsed.path}"

    logger.info("AsyncPostgresSaver created in %.3f seconds from %s", time.perf_counter() - start, safe_uri)
    return cp


CHECKPOINTER_CREATORS: Dict[str, Callable[[], Awaitable[Any]]] = {
    "AsyncRedisSaver": create_redis_checkpointer,
    "PostgreSQL": get_postgres_checkpointer,
    # "PostgresWithPGvector": get_pgvector_checkpointer,
    "MemorySaver": create_memory_checkpointer,
    "AsyncKafkaSaver": create_kafka_checkpointer,
}


async def get_checkpointers_async(cps: List[str]) -> List[Any]:
    tasks = []
    seen = set()

    for cp in cps:
        creator = CHECKPOINTER_CREATORS.get(cp)

        if not creator:
            logger.warning("Unknown checkpointer: %s", cp)
            continue

        if cp not in seen:
            seen.add(cp)
            tasks.append(creator())

    results = await asyncio.gather(*tasks, return_exceptions=False)

    final = []
    for res in results:
        if isinstance(res, Exception):
            logger.error("Checkpointer init failed: %s", res)
        else:
            final.append(res)

    if final:
        logger.info("%d Checkpointers created from %s", len(final), cps)
    else:
        logger.error("NO Checkpointers found from: %s", cps)
    return final


def get_redis_checkpointer():
    # Sync-friendly version (no health check)
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    client = redis.from_url(redis_url, decode_responses=False)
    cp = RedisSaver(client)

    logger.info("RedisSaver created from %s", redis_url)
    return cp
