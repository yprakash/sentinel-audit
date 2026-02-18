"""
Prometheus Metrics Utility
---------------------------
This module exposes system-level metrics over HTTP using prometheus_client.

By default, metrics are exposed at:
    http://<host>:<port>/metrics

Each microservice using this utility MUST expose a unique port.

----------------------------------------------------------------------
How to Configure Prometheus (prometheus.yml)
----------------------------------------------------------------------

Add a scrape job per microservice:

scrape_configs:
  - job_name: 'my-service-name'
    static_configs:
      - targets: ['localhost:9518']

If running in Docker:

scrape_configs:
  - job_name: 'my-service-name'
    static_configs:
      - targets: ['container-name:9518']

If running in Kubernetes:

scrape_configs:
  - job_name: 'my-service-name'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: my-service-name

----------------------------------------------------------------------
Best Practices
----------------------------------------------------------------------
1. Each service must expose metrics on a dedicated port.
2. Do NOT dynamically create metrics with unbounded label values.
3. Keep label cardinality low (service_name, hostname are acceptable).
4. Avoid per-request labels (request_id, user_id, etc.).
5. Do not expose metrics endpoints publicly. Protect via:
   - internal network
   - reverse proxy
   - service mesh
6. Prefer counters and histograms for application metrics.
7. System metrics should be scraped at 5–15 second intervals.

| Update Interval | Scrape Interval | Result                   |
| --------------- | --------------- | ------------------------ |
| 1s              | 15s             | Waste CPU                |
| 10s             | 5s              | Same value scraped twice |
| 5s              | 5s              | Perfect alignment        |
| 5s              | 10s             | Normal and safe          |
"""

import asyncio
import logging
import os
import socket

import psutil
from prometheus_client import start_http_server, Gauge

logger = logging.getLogger(__name__)
SystemMetricsTask = None

CPU_PERCENT = Gauge(
    "cpu_usage_percentage",
    "CPU Usage in percentage",
    ["service_name", "hostname"],
)

MEMORY_USED = Gauge(
    "memory_usage_bytes",
    "Memory Usage in bytes",
    ["service_name", "hostname"],
)

MEMORY_PERCENT = Gauge(
    "memory_usage_percentage",
    "Memory Usage in percentage",
    ["service_name", "hostname"],
)

ASYNC_TASKS_RUNNING = Gauge(
    "asyncio_running_tasks",
    "Number of concurrently running asyncio tasks",
    ["service_name", "hostname"],
)


class SystemMetrics:
    def __init__(self, service_name: str, shutdown_event: asyncio.Event, interval: int = 5):
        self.hostname = os.getenv("HOSTNAME") or socket.gethostname()
        self.service_name = service_name
        self.shutdown_event = shutdown_event
        self.interval = max(interval, 1)

    async def collect_metrics(self):
        logger.info(
            "Started collecting metrics for %s on host %s",
            self.service_name,
            self.hostname,
        )

        while not self.shutdown_event.is_set():
            CPU_PERCENT.labels(self.service_name, self.hostname).set(
                psutil.cpu_percent(interval=None)
            )

            mem = psutil.virtual_memory()

            MEMORY_USED.labels(self.service_name, self.hostname).set(mem.used)
            MEMORY_PERCENT.labels(self.service_name, self.hostname).set(mem.percent)

            ASYNC_TASKS_RUNNING.labels(self.service_name, self.hostname).set(
                len(asyncio.all_tasks(asyncio.get_running_loop()))
            )

            await asyncio.sleep(self.interval)

        logger.info("Stopped metrics collection for %s", self.service_name)


async def start_metrics_server(port, app_name, shutdown_event, interval):
    global SystemMetricsTask
    if SystemMetricsTask is None:
        start_http_server(port)
        logger.info("Metrics server started on port %d", port)

        metrics = SystemMetrics(app_name, shutdown_event, interval)
        SystemMetricsTask = asyncio.create_task(metrics.collect_metrics())

    return SystemMetricsTask


async def main():
    shutdown_event = asyncio.Event()

    task = await start_metrics_server(9518, "test", shutdown_event, 1)

    await shutdown_event.wait()
    await task


if __name__ == '__main__':
    asyncio.run(main())
