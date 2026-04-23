import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Optional, Dict, Any

from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from kafka_utils import KafkaClientFactory

logger = logging.getLogger(__name__)


def get_runnable_config(config: RunnableConfig | Dict[str, Any] | None) -> RunnableConfig:
    if config is None:
        raise ValueError("config(Dict[str, Any]) is required")
    if isinstance(config, dict):
        config = ensure_config(config)
    return config


class AsyncKafkaSaver(BaseCheckpointSaver):
    """
    Async Kafka-based checkpoint saver using a COMPACTED topic.

    Design:
    - Stores checkpoints with `thread_id` as the Kafka message key.
    - Uses Kafka log compaction to retain only the latest checkpoint per thread_id.
    - Provides a durable and fault-tolerant store for restoring latest agent state.
    - Retrieval scans the entire topic and returns the latest checkpoint for the given thread_id.

    Intended Use:
    - Restoring latest state in LangGraph workflows
    - Stateless agent recovery
    - Kafka is being used as log-backed state store, but NOT stream processor. NO need for groups, commits, offsets.

    Limitations:
    - No full checkpoint history (older states are removed by compaction)
    - Kafka does not support direct key-based lookup; scans are required.
    - Read performance depends on topic size (bounded via compaction)

    Topic Requirements:
    - cleanup.policy=compact
    - Producers MUST use thread_id as key
    - Kafka topic must use default partitioner
    - Partition count should remain stable

    Kafka Topic Creation:
        kafka-topics.sh \
          --create \
          --topic agent_checkpoints \
          --bootstrap-server localhost:9092 \
          --partitions 3 \
          --replication-factor 1 \
          --config cleanup.policy=compact \
          --config min.cleanable.dirty.ratio=0.01 \
          --config segment.ms=10000

    Operational Notes:
    - Compaction is asynchronous; multiple versions may temporarily exist
    - Ordering is guaranteed per partition (ensured via thread_id key)
    - Consumers scan from beginning due to lack of key-based lookup
    """

    def __init__(
            self,
            topic: str = "agent_checkpoints",
            # serde: Optional[SerializerProtocol] = None
    ):
        # super().__init__(serde=serde or langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer())
        self.topic = topic
        super().__init__(serde=None)

    def _create_tuple(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            parent_config: Optional[RunnableConfig] = None  # Agent can "fork" into two different paths
    ) -> CheckpointTuple:
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config
        )

    async def aget_tuple(self, config: RunnableConfig | Dict[str, Any]) -> Optional[CheckpointTuple]:
        """
        Retrieves a specific checkpoint. If checkpoint_id is None, it finds the latest
        message in Kafka for the given thread_id.

        Since LangGraph needs to "get" the latest state for a thread_id, and Kafka doesn't
        support random-access lookups by key natively, we will use a short-lived consumer that
        seeks to the end of the partition to find the most recent message for that specific thread.
        """
        start_time = time.perf_counter()
        config = get_runnable_config(config)
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        consumer = await KafkaClientFactory.get_consumer(
            self.topic,
            # group_id=None,  Not needed as it is short-lived, No offsets stored on broker
            enable_auto_commit=False,  # Auto commit is meaningless without a group_id
            consumer_timeout_ms=5000,
        )
        found_tuple = None
        counter = 0
        try:
            # We seek to the beginning to find the specific ID or the latest
            await consumer.seek_to_beginning()  # O(N) scan for every read

            # bounded wait (prevents infinite loop)
            for _ in range(10):  # ~10 polling rounds
                batch = await consumer.getmany(timeout_ms=500)
                if not batch:
                    break

                for _, messages in batch.items():
                    for msg in messages:
                        counter += 1
                        # Kafka keys are bytes; LangGraph thread_ids are usually strings
                        msg_key = msg.key.decode("utf-8") if msg.key else None

                        if msg_key == thread_id:
                            data = json.loads(msg.value.decode("utf-8"))
                            checkpoint = data["checkpoint"]
                            metadata = data.get("metadata", {})
                            parent_config = data.get("parent_config", None)

                            if checkpoint_id and checkpoint["id"] == checkpoint_id:
                                found_tuple = self._create_tuple(config, checkpoint, metadata, parent_config)
                                logger.debug(f"Found checkpoint {checkpoint_id} for thread {thread_id}")
                                return found_tuple
                            elif not checkpoint_id:
                                found_tuple = self._create_tuple(config, checkpoint, metadata, parent_config)

        except Exception as e:
            logger.exception("Failed to retrieve checkpoint")
        finally:
            await consumer.stop()
            elapsed = time.perf_counter() - start_time

        logger.info("Checkpoint retrieved in %.6f seconds from %d kafka messages for thread %s.",
                    elapsed, counter, thread_id)
        return found_tuple

    async def alist(
            self,
            config: RunnableConfig | Dict[str, Any] | None,
            *,
            filter: dict[str, Any] | None = None,
            before: RunnableConfig | Dict[str, Any] | None = None,
            limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """
        Lists checkpoints for a given thread_id.
        """
        start_time = time.perf_counter()
        config = get_runnable_config(config)
        thread_id = config["configurable"]["thread_id"]

        consumer = await KafkaClientFactory.get_consumer(
            self.topic,
            enable_auto_commit=False,
            consumer_timeout_ms=5000,
        )

        counter = 0

        try:
            await consumer.seek_to_beginning()

            # Use bounded polling instead of infinite async iteration
            # This prevents hanging when no more messages are available
            for _ in range(10):  # fixed number of polling rounds
                batch = await consumer.getmany(timeout_ms=500)  # bounded wait per poll

                # If no messages returned, assume topic exhausted → exit
                if not batch:
                    break

                # batch is {TopicPartition: [messages]}
                for _, messages in batch.items():
                    for msg in messages:
                        if msg.key.decode("utf-8") == thread_id:
                            data = json.loads(msg.value.decode("utf-8"))

                            checkpoint = data["checkpoint"]
                            metadata = data.get("metadata", {})
                            parent_config = data.get("parent_config", None)

                            yield self._create_tuple(config, checkpoint, metadata, parent_config)

                            counter += 1

                            # Respect user-provided limit
                            if limit and counter >= limit:
                                return  # stop iteration early

        except Exception:
            logger.exception("Failed to list checkpoint")

        finally:
            await consumer.stop()
            elapsed = time.perf_counter() - start_time

        logger.info(
            "Checkpoint retrieved in %.6f seconds from %d kafka messages for thread %s.",
            elapsed, counter, thread_id
        )

    async def aput(
            self,
            config: RunnableConfig | Dict[str, Any] | None,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Asynchronously store a checkpoint with its configuration and metadata.

        Args:
            config: Configuration for the checkpoint.
            checkpoint: The checkpoint to store.
            metadata: Additional metadata for the checkpoint.
            new_versions: New channel versions as of this write.
        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.

        NOTE: self.topic must have already been created before calling this method.
        kafka-topics.sh \
          --create \
          --topic agent_checkpoints \
          --bootstrap-server localhost:9092 \
          --partitions 3 \
          --replication-factor 1 \
          --config cleanup.policy=compact,delete \
          --config min.insync.replicas=1
        """
        start_time = time.perf_counter()
        config = get_runnable_config(config)
        thread_id = config["configurable"]["thread_id"]
        producer = await KafkaClientFactory.get_producer()
        payload = json.dumps({
            "checkpoint": checkpoint,
            "metadata": metadata,
            "versions": new_versions,
        }).encode("utf-8")

        try:
            await producer.send_and_wait(
                self.topic,
                key=thread_id.encode("utf-8"),
                value=payload
            )
            elapsed = time.perf_counter() - start_time
            logger.info("Checkpoint saved to Kafka in %.6f seconds for thread %s", elapsed, thread_id)
        except Exception:
            logger.exception("Kafka produce failed for thread %s", thread_id)
            raise

        return config


async def main():
    import uuid
    from datetime import datetime, UTC
    # Optional: configure Kafka
    # KafkaClientFactory.configure(bootstrap_servers="localhost:9092")

    saver = AsyncKafkaSaver(topic="agent_checkpoints")

    # Minimal RunnableConfig structure expected by saver
    thread_id = f"test-thread-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    # Dummy checkpoint + metadata
    checkpoint = {
        "id": str(uuid.uuid4()),
        "ts": datetime.now(UTC).isoformat(),
        "state": {"step": "test", "status": "ok"},
    }
    metadata = {"source": "connectivity-test"}
    new_versions = {}  # minimal

    try:
        # --- Test WRITE (aput) ---
        await saver.aput(config, checkpoint, metadata, new_versions)
        print("Checkpoint write successful")

        # --- Test READ (aget_tuple) ---
        result = await saver.aget_tuple(config)
        if result:
            print("Checkpoint read successful:", result.checkpoint)
        else:
            print("ERROR: No checkpoint found for thread_id=", thread_id)

        # --- Test LIST (alist) ---
        print("Listing checkpoints:")
        async for item in saver.alist(config, limit=5):
            print(" - %s", item.checkpoint)

    except Exception as e:
        print("ERROR: Kafka checkpointing test failed: ", e)
    finally:
        await KafkaClientFactory.close_all()


if __name__ == "__main__":
    asyncio.run(main())
