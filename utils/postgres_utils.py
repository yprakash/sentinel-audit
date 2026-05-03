import logging
import os
import time

import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, ChannelVersions
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pgvector.psycopg import register_vector_async

logger = logging.getLogger(__name__)

postgres_checkpointer = None
pgvector_checkpointer = None


class _PostgresCheckpointer(BaseCheckpointSaver):
    def __init__(self):
        self.pg_url = os.getenv(
            "POSTGRES_URL",
            "postgresql://postgres:postgres@localhost:5432/postgres",
        )
        self._client = None

    async def setup(self):
        start = time.perf_counter()
        conn = await psycopg.AsyncConnection.connect(self.pg_url)
        await conn.set_autocommit(True)  # REQUIRED for setup()

        cp = AsyncPostgresSaver(conn)
        await cp.setup()  # create tables

        logger.info("_PostgresCheckpointer created in %.3f seconds from %s",
                    time.perf_counter() - start, self.pg_url)
        self._client = cp

    async def aget_tuple(self, config):
        result = await self._client.aget_tuple(config)
        return result

    async def alist(self, config, **kwargs):
        result = await self._client.alist(config, **kwargs)
        return result

    async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
    ) -> RunnableConfig:
        print("CUSTOM aput CALLED")
        if "channel_values" not in checkpoint:
            checkpoint["channel_values"] = {}
        if "channel_versions" not in checkpoint:
            checkpoint["channel_versions"] = {}
        if "versions_seen" not in checkpoint:
            checkpoint["versions_seen"] = {}

        result = await self._client.aput(config, checkpoint, metadata, new_versions)
        return result

    def put_writes(self, config: RunnableConfig, writes, task_id: str, task_path: str = ""):
        print("CUSTOM put_writes CALLED")
        return self._client.put_writes(
            config,
            writes,
            task_id,
            task_path,
        )

    async def aput_writes(self, config: RunnableConfig, writes, task_id: str, task_path: str = ""):
        print("CUSTOM aput_writes CALLED")
        return await self._client.aput_writes(
            config,
            writes,
            task_id,
            task_path,
        )


class _PGvectorCheckpointer(BaseCheckpointSaver):
    # ToDo: Incomplete
    def __init__(self):
        self.pg_url = os.getenv(
            "POSTGRES_URL",
            "postgresql://postgres:postgres@localhost:5432/postgres",
        )
        self._client = None

    async def setup(self):
        start = time.perf_counter()
        conn = await psycopg.AsyncConnection.connect(self.pg_url)
        await register_vector_async(conn)  # REQUIRED for pgvector
        # await conn.set_autocommit(True)  # REQUIRED for setup()

        cp = AsyncPostgresSaver(conn)
        await cp.setup()

        logger.info("_PGvectorCheckpointer created in %.3f seconds from %s",
                    time.perf_counter() - start, self.pg_url)
        self._client = cp


async def get_postgres_checkpointer():
    global postgres_checkpointer
    if postgres_checkpointer is None:
        postgres_checkpointer = _PostgresCheckpointer()
        await postgres_checkpointer.setup()

    return postgres_checkpointer


async def get_pgvector_checkpointer():
    global pgvector_checkpointer
    if not pgvector_checkpointer:
        pgvector_checkpointer = _PGvectorCheckpointer()
        await pgvector_checkpointer.setup()

    return pgvector_checkpointer
