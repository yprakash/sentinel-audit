from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Centralized configuration.
    Loads values from:
    - Environment variables
    - .env file
    """

    # ---- Strategist Agent LLM ----
    STRATEGIST_PROVIDER: str = Field(default="groq")
    STRATEGIST_MODEL: str = Field(default="llama-3.1-8b-instant")
    STRATEGIST_TEMPERATURE: float = Field(default=0.0)
    STRATEGIST_MAX_TOKENS: int = Field(default=2048)

    # ---- Adversary Agent LLM ----
    ADVERSARY_PROVIDER: str = Field(default="groq")
    ADVERSARY_MODEL: str = Field(default="llama-3.1-8b-instant")
    ADVERSARY_TEMPERATURE: float = Field(default=0.0)
    ADVERSARY_MAX_TOKENS: int = Field(default=2048)

    # ---- Validator Agent LLM ----
    VALIDATOR_PROVIDER: str = Field(default="groq")
    VALIDATOR_MODEL: str = Field(default="llama-3.1-8b-instant")
    VALIDATOR_TEMPERATURE: float = Field(default=0.0)
    VALIDATOR_MAX_TOKENS: int = Field(default=2048)

    # ---- Reporter Agent LLM ----
    REPORTER_PROVIDER: str = Field(default="groq")
    REPORTER_MODEL: str = Field(default="llama-3.1-8b-instant")
    REPORTER_TEMPERATURE: float = Field(default=0.0)
    REPORTER_MAX_TOKENS: int = Field(default=2048)

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
