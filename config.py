from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration.
    Loads values from:
    - Environment variables
    - .env file
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",  # mandatory
    )

    # ---- Strategist Agent LLM ----
    STRATEGIST_PROVIDER: str = Field(default="groq")
    STRATEGIST_MODEL: str = Field(default="llama-3.1-8b-instant")

    # ---- Adversary Agent LLM ----
    ADVERSARY_PROVIDER: str = Field(default="groq")
    ADVERSARY_MODEL: str = Field(default="llama-3.1-8b-instant")

    # ---- Validator Agent LLM ----
    VALIDATOR_PROVIDER: str = Field(default="groq")
    VALIDATOR_MODEL: str = Field(default="llama-3.1-8b-instant")

    # ---- Reporter Agent LLM ----
    REPORTER_PROVIDER: str = Field(default="groq")
    REPORTER_MODEL: str = Field(default="llama-3.1-8b-instant")


settings = Settings()
