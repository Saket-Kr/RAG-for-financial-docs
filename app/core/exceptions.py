class DocumentQAException(Exception):
    pass


class DocumentNotFoundError(DocumentQAException):
    pass


class DocumentProcessingError(DocumentQAException):
    pass


class ParserError(DocumentQAException):
    pass


class EmbeddingError(DocumentQAException):
    pass


class VectorDBError(DocumentQAException):
    pass


class QueryError(DocumentQAException):
    pass


class LLMError(DocumentQAException):
    pass


class StorageError(DocumentQAException):
    pass


class ValidationError(DocumentQAException):
    pass


class GatekeepingError(DocumentQAException):
    pass
