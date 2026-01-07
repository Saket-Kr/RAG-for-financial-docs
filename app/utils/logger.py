import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from app.core.config import Settings


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        if hasattr(record, "document_id"):
            log_data["document_id"] = record.document_id

        return json.dumps(log_data)


def setup_logging(settings: Settings) -> None:
    log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)

    log_file = Path(settings.logging.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handlers = []

    if settings.logging.format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    logging.basicConfig(level=log_level, handlers=handlers, force=True)
