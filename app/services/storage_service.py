from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

from app.core.exceptions import StorageError


class BaseStorage(ABC):
    @abstractmethod
    def save_file(self, file_content: bytes, filename: str) -> str:
        pass

    @abstractmethod
    def get_file(self, file_path: str) -> bytes:
        pass

    @abstractmethod
    def delete_file(self, file_path: str) -> None:
        pass

    @abstractmethod
    def file_exists(self, file_path: str) -> bool:
        pass


class LocalStorage(BaseStorage):
    def __init__(self, upload_directory: str = "./data/uploads"):
        self.upload_directory = Path(upload_directory)
        self.upload_directory.mkdir(parents=True, exist_ok=True)

    def save_file(self, file_content: bytes, filename: str) -> str:
        try:
            safe_filename = self._sanitize_filename(filename)
            file_path = self.upload_directory / safe_filename

            if file_path.exists():
                counter = 1
                base_name = file_path.stem
                extension = file_path.suffix
                while file_path.exists():
                    file_path = (
                        self.upload_directory / f"{base_name}_{counter}{extension}"
                    )
                    counter += 1

            file_path.write_bytes(file_content)
            return str(file_path)
        except Exception as e:
            raise StorageError(f"Failed to save file: {str(e)}") from e

    def get_file(self, file_path: str) -> bytes:
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.upload_directory / path

            if not path.exists():
                raise StorageError(f"File not found: {file_path}")

            return path.read_bytes()
        except Exception as e:
            raise StorageError(f"Failed to read file: {str(e)}") from e

    def delete_file(self, file_path: str) -> None:
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.upload_directory / path

            if path.exists():
                path.unlink()
        except Exception as e:
            raise StorageError(f"Failed to delete file: {str(e)}") from e

    def file_exists(self, file_path: str) -> bool:
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.upload_directory / path
            return path.exists()
        except Exception:
            return False

    def _sanitize_filename(self, filename: str) -> str:
        import re

        filename = re.sub(r"[^\w\s.-]", "", filename)
        filename = re.sub(r"\s+", "_", filename)
        return filename[:255]
