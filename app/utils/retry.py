"""HTTP retry decorator for handling transient failures in HTTP operations."""

import asyncio
import inspect
import logging
import random
import time
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

import httpx

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _is_retryable_http_error(exception: Exception, retryable_status_codes: tuple[int, ...] = (500, 502, 503, 504)) -> bool:
    """
    Determine if an HTTP exception should be retried.

    Retries on:
    - HTTP server errors matching retryable_status_codes
    - Connection errors (ConnectError, TimeoutException)

    Does NOT retry on:
    - HTTP 4xx client errors (400, 401, 403, 404, etc.)
    - Other non-HTTP exceptions

    Args:
        exception: The exception to check
        retryable_status_codes: Tuple of HTTP status codes to retry

    Returns:
        True if the exception should be retried, False otherwise
    """
    # Retry on connection and timeout errors
    if isinstance(exception, (httpx.ConnectError, httpx.TimeoutException)):
        return True

    # Retry on HTTP status errors matching retryable codes
    if isinstance(exception, httpx.HTTPStatusError):
        status_code = exception.response.status_code
        return status_code in retryable_status_codes

    return False


def _calculate_backoff(attempt: int, initial_backoff: float, max_backoff: float) -> float:
    """
    Calculate exponential backoff with jitter.

    Args:
        attempt: Current retry attempt (0-indexed)
        initial_backoff: Initial backoff delay in seconds
        max_backoff: Maximum backoff delay in seconds

    Returns:
        Backoff delay in seconds
    """
    # Exponential backoff: initial * 2^attempt
    backoff = min(initial_backoff * (2 ** attempt), max_backoff)

    # Add jitter (random between 0 and 20% of backoff)
    jitter = random.uniform(0, backoff * 0.2)
    return backoff + jitter


def _is_async_function(func: Callable[..., Any]) -> bool:
    """Check if a function is async."""
    return inspect.iscoroutinefunction(func)


def retry_http(
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 10.0,
    retryable_status_codes: tuple[int, ...] = (500, 502, 503, 504),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to retry HTTP operations on transient failures.

    Retries on:
    - HTTP 5xx server errors (configurable via retryable_status_codes)
    - Connection errors (httpx.ConnectError, httpx.TimeoutException)

    Does NOT retry on:
    - HTTP 4xx client errors (400, 401, 403, 404, etc.)
    - Other non-HTTP exceptions

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_backoff: Initial backoff delay in seconds (default: 1.0)
        max_backoff: Maximum backoff delay in seconds (default: 10.0)
        retryable_status_codes: Tuple of HTTP status codes to retry (default: 5xx)

    Returns:
        Decorated function with retry logic

    Example:
        ```python
        @retry_http(max_retries=3, initial_backoff=1.0)
        def make_http_request():
            with httpx.Client() as client:
                response = client.get("https://api.example.com")
                response.raise_for_status()
                return response.json()
        ```
    """
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if _is_async_function(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                last_exception: Exception | None = None

                for attempt in range(max_retries):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e

                        # Check if this exception should be retried
                        if not _is_retryable_http_error(e, retryable_status_codes):
                            # Don't retry, re-raise immediately
                            raise

                        # If this is the last attempt, don't wait
                        if attempt < max_retries - 1:
                            backoff = _calculate_backoff(attempt, initial_backoff, max_backoff)
                            func_name = func.__name__
                            logger.warning(
                                f"Retryable error in {func_name} (attempt {attempt + 1}/{max_retries}): "
                                f"{type(e).__name__}: {str(e)}. Retrying in {backoff:.2f}s"
                            )
                            await asyncio.sleep(backoff)
                        else:
                            # Last attempt failed, log and re-raise
                            func_name = func.__name__
                            logger.error(
                                f"Max retries exceeded in {func_name} after {max_retries} attempts: "
                                f"{type(e).__name__}: {str(e)}"
                            )

                # Should never reach here, but for type safety
                if last_exception:
                    raise last_exception
                raise RuntimeError("Unexpected retry logic error")

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                last_exception: Exception | None = None

                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e

                        # Check if this exception should be retried
                        if not _is_retryable_http_error(e, retryable_status_codes):
                            # Don't retry, re-raise immediately
                            raise

                        # If this is the last attempt, don't wait
                        if attempt < max_retries - 1:
                            backoff = _calculate_backoff(attempt, initial_backoff, max_backoff)
                            func_name = func.__name__
                            logger.warning(
                                f"Retryable error in {func_name} (attempt {attempt + 1}/{max_retries}): "
                                f"{type(e).__name__}: {str(e)}. Retrying in {backoff:.2f}s"
                            )
                            time.sleep(backoff)
                        else:
                            # Last attempt failed, log and re-raise
                            func_name = func.__name__
                            logger.error(
                                f"Max retries exceeded in {func_name} after {max_retries} attempts: "
                                f"{type(e).__name__}: {str(e)}"
                            )

                # Should never reach here, but for type safety
                if last_exception:
                    raise last_exception
                raise RuntimeError("Unexpected retry logic error")

            return sync_wrapper

    return decorator
