from fastapi import Depends
from app.core.database import Database
from app.core.config import Settings
from app.services.metadata_service import MetadataService
from app.services.storage_service import LocalStorage
from app.services.document_service import DocumentService


def get_settings() -> Settings:
    return Settings.from_yaml()


def get_database(settings: Settings = Depends(get_settings)) -> Database:
    return Database(settings)


def get_metadata_service(db: Database = Depends(get_database)) -> MetadataService:
    return MetadataService(db)


def get_storage(settings: Settings = Depends(get_settings)) -> LocalStorage:
    return LocalStorage(settings.storage.upload_directory)


def get_document_service(
    settings: Settings = Depends(get_settings),
    metadata_service: MetadataService = Depends(get_metadata_service),
    storage: LocalStorage = Depends(get_storage)
) -> DocumentService:
    return DocumentService(settings, metadata_service, storage)
