import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.routes import router as v1_router
from app.core.config import Settings
from app.core.database import Database
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    DocumentQAException,
    GatekeepingError,
    ValidationError,
)
from app.utils.logger import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_yaml()
    setup_logging(settings)

    logger.info("Starting application...")

    db = Database(settings)
    db.init_db()
    logger.info("Database initialized")

    yield

    logger.info("Shutting down application...")


app = FastAPI(title="Document Query Answerer", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    import uuid

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    logger.info(f"Request {request_id}: {request.method} {request.url.path}")

    response = await call_next(request)

    logger.info(f"Request {request_id}: Status {response.status_code}")

    return response


@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
    return JSONResponse(
        status_code=404, content={"error": "Document not found", "detail": str(exc)}
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400, content={"error": "Validation error", "detail": str(exc)}
    )


@app.exception_handler(GatekeepingError)
async def gatekeeping_error_handler(request: Request, exc: GatekeepingError):
    return JSONResponse(
        status_code=400, content={"error": "Query not relevant", "detail": str(exc)}
    )


@app.exception_handler(DocumentProcessingError)
async def processing_error_handler(request: Request, exc: DocumentProcessingError):
    return JSONResponse(
        status_code=500,
        content={"error": "Document processing error", "detail": str(exc)},
    )


@app.exception_handler(DocumentQAException)
async def general_exception_handler(request: Request, exc: DocumentQAException):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500, content={"error": "Internal server error", "detail": str(exc)}
    )


app.include_router(v1_router, prefix="/api/v1", tags=["v1"])


@app.get("/")
async def root():
    return {"message": "Document Query Answerer API", "version": "1.0.0"}
