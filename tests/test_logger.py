"""Tests for the logging configuration module."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

from src.utils.logger import setup_logger


class TestLoggerCreation:
    def test_creates_logger_with_handlers(self, tmp_path: Path):
        logger = setup_logger("test_basic", log_dir=str(tmp_path))
        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) == 2
        # Cleanup
        logger.handlers.clear()

    def test_no_duplicate_handlers(self, tmp_path: Path):
        logger = setup_logger("test_dup", log_dir=str(tmp_path))
        handler_count = len(logger.handlers)
        logger2 = setup_logger("test_dup", log_dir=str(tmp_path))
        assert logger is logger2
        assert len(logger2.handlers) == handler_count
        # Cleanup
        logger.handlers.clear()

    def test_creates_log_directory(self, tmp_path: Path):
        log_dir = tmp_path / "subdir" / "logs"
        setup_logger("test_dir", log_dir=str(log_dir))
        assert log_dir.exists()
        # Cleanup
        logging.getLogger("test_dir").handlers.clear()

    def test_writes_plaintext_by_default(self, tmp_path: Path):
        logger = setup_logger("test_plain", log_dir=str(tmp_path))
        logger.info("hello plaintext")
        # Flush handlers
        for h in logger.handlers:
            h.flush()
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1
        content = log_files[0].read_text(encoding="utf-8")
        assert "hello plaintext" in content
        assert "test_plain" in content
        # Cleanup
        logger.handlers.clear()


class TestJsonLogging:
    def test_json_format_when_env_set(self, tmp_path: Path):
        with patch("src.utils.logger._JSON_MODE", True):
            # Need a unique logger name to avoid handler reuse
            logger = setup_logger("test_json", log_dir=str(tmp_path))
            logger.info("hello json")
            for h in logger.handlers:
                h.flush()
            jsonl_files = list(tmp_path.glob("*.jsonl"))
            assert len(jsonl_files) == 1
            content = jsonl_files[0].read_text(encoding="utf-8").strip()
            record = json.loads(content)
            assert record["message"] == "hello json"
            assert record["levelname"] == "INFO"
            # Cleanup
            logger.handlers.clear()
