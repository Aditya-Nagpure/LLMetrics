from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: Literal["groq", "anthropic"] = "groq"
    llm_model: str = "llama3-8b-8192"
    groq_api_key: str = ""
    anthropic_api_key: str = ""

    # Storage
    storage_backend: Literal["jsonl", "sqlite"] = "jsonl"
    jsonl_path: str = "data/traces.jsonl"
    sqlite_path: str = "data/llmetrics.db"

    # Evaluation
    enable_llm_judge: bool = False
    llm_judge_model: str = "llama3-8b-8192"
    faithfulness_threshold: float = 0.6

    # Cost overrides (USD per 1M tokens)
    cost_input_per_1m: float | None = None
    cost_output_per_1m: float | None = None

    # ChromaDB
    chroma_collection: str = "documents"
    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # Retrieval
    retriever_top_k: int = Field(default=10, ge=1)
    reranker_top_n: int = Field(default=5, ge=1)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


settings = Settings()