# CLAUDE.md

## Build and Run

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Lint and format
ruff check .
ruff check --fix .
ruff format .

# Type check
mypy src/

# Start web UI
python web_monitor.py

# Run full pipeline
python run_all.py

# Run specific phase
python run_all.py --phase search
python run_all.py --phase compile
python run_all.py --phase metadata
```

## Project Architecture

```
src/
  agents/          # Pipeline orchestration (daily_researcher, paper_compiler)
  apis/            # Academic API clients (arxiv, crossref, openalex, semantic_scholar)
  models/          # Pydantic data models (Paper, ExistingPaper)
  utils/           # Shared utilities (sqlite_store, monitor_store, duplicate_detector, relevance_scorer, reference_formatter, cache_manager)
  web/             # FastAPI web application (app.py, templates/, static/, proxy.py)
config/            # YAML configuration files
data/              # SQLite database, JSON exports, cache
output/            # Generated reports, thesis artifacts, daily outputs
tests/             # pytest test suite
```

## Key Conventions

- Python 3.11+ with type hints throughout
- Pydantic models for data validation (`src/models/paper.py`)
- SQLite as primary storage with FTS5 full-text search
- JSON export maintained for compatibility
- All file I/O uses `encoding="utf-8"`
- Ruff for linting (line-length 120) and formatting
- Tests use pytest with fixtures in `tests/conftest.py`

## Adding a New API Client

1. Create `src/apis/new_api.py` with a class following the pattern in existing clients
2. Implement a `search(query, limit, year_range) -> list[Paper]` method
3. Register it in `src/agents/daily_researcher.py` (`SEARCH_API_ORDER`, `_build_api_client()`)
4. Add the API name to `config/config.yaml` under `apis`
5. Add tests in `tests/`

## Database

- Primary: SQLite at `data/papers_database.sqlite`
- Schema managed by `src/utils/sqlite_store.py` (papers table, FTS5 index, sync metadata)
- Runtime tables (job_runs, runtime_locks) managed by `src/utils/monitor_store.py`
- JSON mirror at `data/papers_database.json` updated on web UI changes
