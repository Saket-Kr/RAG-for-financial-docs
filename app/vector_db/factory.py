from typing import Dict, Type

from app.core.exceptions import VectorDBError
from app.vector_db.base import BaseVectorDB
from app.vector_db.chroma_db import ChromaVectorDB
from app.vector_db.faiss_db import FAISSVectorDB
from app.vector_db.qdrant_db import QdrantVectorDB


class VectorDBFactory:
    _dbs: Dict[str, Type[BaseVectorDB]] = {
        "chroma": ChromaVectorDB,
        "faiss": FAISSVectorDB,
        "qdrant": QdrantVectorDB,
    }

    @classmethod
    def create_db(
        cls,
        db_type: str,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "financial_documents",
        distance_metric: str = "cosine",
        host: str = "localhost",
        port: int = 6333,
        use_global_collection: bool = True,
    ) -> BaseVectorDB:
        if db_type not in cls._dbs:
            raise VectorDBError(f"Unknown vector DB type: {db_type}")

        db_class = cls._dbs[db_type]
        if db_type == "qdrant":
            return db_class(
                persist_directory,
                collection_name,
                distance_metric,
                host,
                port,
                use_global_collection,
            )
        return db_class(persist_directory, collection_name, distance_metric)

    @classmethod
    def register_db(cls, name: str, db_class: Type[BaseVectorDB]) -> None:
        cls._dbs[name] = db_class
