from typing import Dict, Type
from app.vector_db.base import BaseVectorDB
from app.vector_db.chroma_db import ChromaVectorDB
from app.vector_db.faiss_db import FAISSVectorDB
from app.core.exceptions import VectorDBError


class VectorDBFactory:
    _dbs: Dict[str, Type[BaseVectorDB]] = {
        "chroma": ChromaVectorDB,
        "faiss": FAISSVectorDB,
    }

    @classmethod
    def create_db(
        cls,
        db_type: str,
        persist_directory: str = "./data/vector_db",
        collection_name: str = "financial_documents",
        distance_metric: str = "cosine"
    ) -> BaseVectorDB:
        if db_type not in cls._dbs:
            raise VectorDBError(f"Unknown vector DB type: {db_type}")
        
        db_class = cls._dbs[db_type]
        return db_class(persist_directory, collection_name, distance_metric)

    @classmethod
    def register_db(cls, name: str, db_class: Type[BaseVectorDB]) -> None:
        cls._dbs[name] = db_class
