"""Structured JSON logging with correlation ID support.

Provides production-ready logging that outputs JSON-formatted log entries
with automatic correlation_id injection via contextvars. Every prediction
event is logged with input_hash, decision, probability, and latency.
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path

from pythonjsonlogger.json import JsonFormatter

# Context variable for request correlation tracking
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="no-correlation-id")


class CorrelationIdFilter(logging.Filter):
    """Inject correlation_id from contextvars into every log record.

    This filter reads the current correlation_id from the context variable
    and attaches it to the log record so it appears in JSON output.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation_id attribute to the log record.

        Args:
            record: The log record to enrich.

        Returns:
            bool: Always True (never filters out records).
        """
        record.correlation_id = correlation_id_var.get()  # type: ignore[attr-defined]
        return True


def setup_logging(log_level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure structured JSON logging for the application.

    Sets up two handlers:
    - Console (stdout): All log messages in JSON format.
    - File (logs/app.json): Persistent log storage, created if log_dir is provided.

    Args:
        log_level: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. If None, only console logging is used.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicates on re-initialization
    root_logger.handlers.clear()

    # JSON formatter with structured fields
    json_formatter = JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s %(correlation_id)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        json_ensure_ascii=False,
    )

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    console_handler.addFilter(CorrelationIdFilter())
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "app.json", encoding="utf-8")
        file_handler.setFormatter(json_formatter)
        file_handler.addFilter(CorrelationIdFilter())
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


def generate_correlation_id() -> str:
    """Generate a new UUID-based correlation ID and set it in the context.

    Returns:
        str: The generated correlation ID.
    """
    cid = str(uuid.uuid4())
    correlation_id_var.set(cid)
    return cid


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)
