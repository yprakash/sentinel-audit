import asyncio
import logging
import os

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from utils.shutdown_manager import shutdown_manager

logger = logging.getLogger(__name__)


class KafkaClientFactory:
    _producer_lock = asyncio.Lock()
    _producer: AIOKafkaProducer | None = None
    _config = {
        "bootstrap_servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    }

    @classmethod
    def register_shutdown(cls, shutdown_mgr):
        shutdown_mgr.register_callback(cls.close_all)

    @classmethod
    def configure(cls, **kwargs):
        # Clients can set their own configs: KafkaClientFactory.configure(bootstrap_servers="kafka-prod:9092", ...)
        cls._config.update(kwargs)

    @classmethod
    async def is_healthy(cls) -> bool:
        try:
            producer = await cls.get_producer()
            return producer._closed is False
        except Exception:
            return False

    @classmethod
    async def close_all(cls):
        if cls._producer:
            try:
                async with asyncio.timeout(10):
                    await cls._producer.flush()
                    logger.info("Kafka Producer flush() completed")
            except TimeoutError:
                logger.warning("Kafka producer flush timed out")

            try:
                await cls._producer.stop()
            finally:
                cls._producer = None
                logger.info("Closed Kafka Producer")
        else:
            logger.warning("Kafka producer not initialized or already closed")

    @classmethod
    async def flush_producer(cls):
        if cls._producer:
            await cls._producer.flush()

    @classmethod
    async def get_producer(
            cls,
            **kwargs
    ) -> AIOKafkaProducer:
        if cls._producer is None:
            async with cls._producer_lock:
                # Double-check pattern after acquiring lock
                if cls._producer is None:
                    # In Kafka, Producers are thread-safe and expensive to set up, so we almost always use a Singleton
                    # We allow **kwargs so you can override defaults like auto_offset_reset
                    options = {
                        "bootstrap_servers": cls._config["bootstrap_servers"],
                        "acks": "all",  # durability: ensures data is committed to all replicas
                        "enable_idempotence": True,  # prevent duplicate writes during retries
                        "linger_ms": 5,  # batching
                        # some of low-level network behaviors are NOT supported by aiokafka like below
                        # "max_in_flight_requests_per_connection": 5,
                        # "retries": 5,
                        # ToDo: Add Security Support hooks
                        # "security_protocol": "SASL_SSL",
                        # "sasl_mechanism": "PLAIN",
                        # "sasl_plain_username": ...,
                        # "sasl_plain_password": ...,
                    }
                    options.update(kwargs)

                    cls._producer = AIOKafkaProducer(**options)
                    async with asyncio.timeout(10):
                        await cls._producer.start()
                    logger.info("Created Kafka Producer to %s", cls._config["bootstrap_servers"])

        return cls._producer

    @classmethod
    async def get_consumer(
            cls,
            topic: str,
            **kwargs
    ) -> AIOKafkaConsumer:
        """
        Creates and starts a new Kafka consumer instance.

        Requirements:
        - The caller is responsible for stopping the consumer.
        - This consumer defaults to manual commit. Caller must handle offset commits.
        - This utility does NOT create topics automatically. The topic must already exist in Kafka.
        - Topic creation is usually infra responsibility. Ensure topics are provisioned.
        """
        options = {
            # "group_id": group_id,  # part of kwargs when needed
            "bootstrap_servers": cls._config["bootstrap_servers"],
            "auto_offset_reset": "earliest",
            "enable_auto_commit": False,
        }
        options.update(kwargs)

        consumer: AIOKafkaConsumer = AIOKafkaConsumer(topic, **options)
        try:
            async with asyncio.timeout(10):
                await consumer.start()
            logger.info("Started new consumer on topic: %s with configs: %s", topic, options)
            return consumer
        except Exception:
            logger.exception("Failed to start Kafka consumer")
            await consumer.stop()  # If start fails, ensure we don't leave a half-baked object
            raise


KafkaClientFactory.register_shutdown(shutdown_manager)


async def main():
    # Optional: override bootstrap servers if needed
    # KafkaClientFactory.configure(bootstrap_servers="localhost:9092")
    try:
        # --- Test Producer Connectivity ---
        producer = await KafkaClientFactory.get_producer()
        print("Producer started successfully")

        # --- Fetch cluster metadata (no produce/consume) ---
        client = producer.client  # underlying aiokafka client
        await client.bootstrap()  # ensure metadata is loaded

        topics = list(client.cluster.topics())
        print(f"Available topics: {topics}")

        # --- Health Check ---
        is_healthy = await KafkaClientFactory.is_healthy()
        print("Kafka health check:", is_healthy)

    except Exception as e:
        logging.exception("Kafka connectivity test failed: %s", e)
    finally:
        await KafkaClientFactory.close_all()


if __name__ == "__main__":
    asyncio.run(main())
