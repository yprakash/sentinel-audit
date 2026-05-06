from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration for sentinel audit settings.
    Loads values from:
    - Environment variables
    - .env file
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",  # mandatory
    )

    # Graph Checkpointing
    GRAPH_CHECKPOINTS: str = Field(default="MemorySaver")

    # ---- Strategist Agent LLM ----
    STRATEGIST_PROVIDER: str = Field(default="mlx")
    STRATEGIST_MODEL: Optional[str] = Field(default=None)

    # ---- Adversary Agent LLM ----
    ADVERSARY_PROVIDER: str = Field(default="mlx")
    ADVERSARY_MODEL: Optional[str] = Field(default=None)

    # ---- Validator Agent LLM ----
    VALIDATOR_PROVIDER: str = Field(default="mlx")
    VALIDATOR_MODEL: Optional[str] = Field(default=None)

    # ---- Reporter Agent LLM ----
    REPORTER_PROVIDER: str = Field(default="mlx")
    REPORTER_MODEL: Optional[str] = Field(default=None)


settings = Settings()
