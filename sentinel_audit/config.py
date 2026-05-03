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
    STRATEGIST_MODEL: str = Field(default="mlx-community/gemma-4-e2b-it-4bit")

    # ---- Adversary Agent LLM ----
    ADVERSARY_PROVIDER: str = Field(default="mlx")
    ADVERSARY_MODEL: str = Field(default="mlx-community/gemma-4-e2b-it-4bit")

    # ---- Validator Agent LLM ----
    VALIDATOR_PROVIDER: str = Field(default="mlx")
    VALIDATOR_MODEL: str = Field(default="mlx-community/gemma-4-e2b-it-4bit")

    # ---- Reporter Agent LLM ----
    REPORTER_PROVIDER: str = Field(default="mlx")
    REPORTER_MODEL: str = Field(default="mlx-community/gemma-4-e2b-it-4bit")


settings = Settings()
