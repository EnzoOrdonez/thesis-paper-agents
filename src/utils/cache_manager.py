"""Cache cleanup utility for removing stale cache files."""

from __future__ import annotations

import time
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger("cache_manager")


def cleanup_cache(cache_dir: str | Path, max_age_days: int = 14) -> int:
    """Remove cache files older than max_age_days. Returns count of deleted files."""
    cache_path = Path(cache_dir)
    if not cache_path.is_dir():
        return 0

    cutoff = time.time() - (max_age_days * 86400)
    deleted = 0

    for file in cache_path.iterdir():
        if not file.is_file():
            continue
        try:
            if file.stat().st_mtime < cutoff:
                file.unlink()
                deleted += 1
        except OSError:
            continue

    if deleted:
        logger.info("Cleaned up %d cache files older than %d days from %s", deleted, max_age_days, cache_path)
    return deleted
