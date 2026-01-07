from pathlib import Path
from typing import Optional

import yaml
from config.settings_loader import load_settings
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    provider: str = "ollama"
    model: str = "mistral:7b-instruct"
    base_url: str = "http://ollama:11434"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: int = 60


class QueryAnsweringSettings(BaseSettings):
    strategy: str = "rag"
    top_k: int = 5
    similarity_threshold: float = 0.3
    llm: LLMSettings = Field(default_factory=LLMSettings)


class ChunkingSettings(BaseSettings):
    strategy: str = "hierarchical"
    chunk_size: int = 512
    chunk_overlap: int = 50
    preserve_structure: bool = True


class PDFParserSettings(BaseSettings):
    type: str = "pdfplumber"


class EmbeddingSettings(BaseSettings):
    type: str = "nomic_embed"
    model_name: str = "nomic-embed-text-v1"
    device: str = "cpu"
    batch_size: int = 32
    dimension: int = 768
    
    model_config = SettingsConfigDict(
        protected_namespaces=('settings_',)
    )


class VectorDBSettings(BaseSettings):
    type: str = "chroma"
    persist_directory: str = "./data/vector_db"
    collection_name: str = "financial_documents"
    distance_metric: str = "cosine"


class GatekeepingSettings(BaseSettings):
    enabled: bool = True
    threshold: float = 0.3
    rejection_message: str = "This query does not appear to be related to the document."


class QueueSettings(BaseSettings):
    broker: str = "redis://redis:6379/0"
    result_backend: str = "redis://redis:6379/0"
    worker_concurrency: int = 4
    task_time_limit: int = 300


class StorageSettings(BaseSettings):
    type: str = "local"
    upload_directory: str = "./data/uploads"
    max_file_size_mb: int = 50


class DatabaseSettings(BaseSettings):
    url: str = "sqlite:///./data/metadata.db"
    echo: bool = False


class LoggingSettings(BaseSettings):
    level: str = "INFO"
    format: str = "json"
    file: str = "./logs/app.log"


class AppSettings(BaseSettings):
    name: str = "Document Query Answerer"
    version: str = "1.0.0"
    mode: str = "async"
    max_concurrent_requests: int = 10
    request_timeout: int = 300


class Settings(BaseSettings):
    app: AppSettings = Field(default_factory=AppSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    query_answering: QueryAnsweringSettings = Field(default_factory=QueryAnsweringSettings)
    gatekeeping: GatekeepingSettings = Field(default_factory=GatekeepingSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "Settings":
        raw_config = load_settings(config_path)
        
        # Extract llm from query_answering to avoid duplicate argument
        query_answering_config = raw_config.get("query_answering", {}).copy()
        llm_config = query_answering_config.pop("llm", {})
        
        return cls(
            app=AppSettings(**raw_config.get("app", {})),
            pdf_parser=PDFParserSettings(**raw_config.get("pdf_parser", {})),
            chunking=ChunkingSettings(**raw_config.get("chunking", {})),
            embeddings=EmbeddingSettings(**raw_config.get("embeddings", {})),
            vector_db=VectorDBSettings(**raw_config.get("vector_db", {})),
            query_answering=QueryAnsweringSettings(
                **query_answering_config,
                llm=LLMSettings(**llm_config)
            ),
            gatekeeping=GatekeepingSettings(**raw_config.get("gatekeeping", {})),
            queue=QueueSettings(**raw_config.get("queue", {})),
            storage=StorageSettings(**raw_config.get("storage", {})),
            database=DatabaseSettings(**raw_config.get("database", {})),
            logging=LoggingSettings(**raw_config.get("logging", {}))
        )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__"
    )
