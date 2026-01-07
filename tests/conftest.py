import pytest
import tempfile
import shutil
from pathlib import Path
from app.core.config import Settings
from app.core.database import Database
from app.services.metadata_service import MetadataService
from app.services.storage_service import LocalStorage


@pytest.fixture
def temp_dir():
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_settings(temp_dir):
    settings = Settings.from_yaml()
    settings.database.url = f"sqlite:///{temp_dir}/test.db"
    settings.storage.upload_directory = f"{temp_dir}/uploads"
    settings.vector_db.persist_directory = f"{temp_dir}/vector_db"
    return settings


@pytest.fixture
def test_database(test_settings):
    db = Database(test_settings)
    db.init_db()
    return db


@pytest.fixture
def test_metadata_service(test_database):
    return MetadataService(test_database)


@pytest.fixture
def test_storage(test_settings):
    return LocalStorage(test_settings.storage.upload_directory)


@pytest.fixture
def sample_pdf_content():
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 0\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF"


@pytest.fixture
def sample_text():
    return "This is a sample document text for testing purposes. It contains multiple sentences. Each sentence provides context for testing chunking strategies."
