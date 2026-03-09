"""Logging configuration for the thesis paper agents."""

import logging
import os
from datetime import datetime
from pathlib import Path

_JSON_MODE = os.environ.get("TPA_LOG_FORMAT", "").lower() == "json"


def _make_json_formatter() -> logging.Formatter | None:
    """Try to build a JSON formatter; return None if the library is missing."""
    try:
        from pythonjsonlogger.json import JsonFormatter
    except ImportError:
        return None
    return JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def setup_logger(name: str, log_dir: str = "logs", level: int = logging.DEBUG) -> logging.Logger:
    """Set up a logger with console (INFO) and file (DEBUG) handlers.

    When the environment variable ``TPA_LOG_FORMAT=json`` is set, the file
    handler writes JSON-lines (one JSON object per line).  The console
    handler always uses plain text for readability.

    Args:
        name: Logger name (module name).
        log_dir: Directory for log files.
        level: Minimum logging level for the file handler.

    Returns:
        Configured logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    plain_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — DEBUG level
    today = datetime.now().strftime("%Y-%m-%d")
    json_formatter = _make_json_formatter() if _JSON_MODE else None
    file_ext = ".jsonl" if json_formatter else ".log"
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{today}{file_ext}"), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(json_formatter or plain_format)
    logger.addHandler(file_handler)

    # Console handler — INFO level (always plain text)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(plain_format)
    logger.addHandler(console_handler)

    return logger
