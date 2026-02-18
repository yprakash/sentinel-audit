import httpx

DEFAULT_TIMEOUT = httpx.Timeout(timeout=10, connect=10.0)
DEFAULT_MAX_RETRIES = 2
DEFAULT_CONNECTION_LIMITS = httpx.Limits(max_connections=10, max_keepalive_connections=5)

INITIAL_RETRY_DELAY = 0.5
MAX_RETRY_DELAY = 8.0
