# Contributing

Thank you for your interest in contributing to **Thesis Paper Agents**.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/EnzoOrdonez/thesis-paper-agents.git
cd thesis-paper-agents

# Create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # macOS / Linux

# Install with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest -v
```

All tests must pass before submitting a pull request.

## Code Quality

```bash
# Lint
ruff check .

# Auto-fix lint issues
ruff check --fix .

# Format
ruff format .

# Type check
mypy src/
```

The CI pipeline runs `ruff check`, `ruff format --check`, and `pytest` on every push and pull request.

## Project Structure

```
src/
  agents/    # Pipeline orchestration
  apis/      # Academic API clients (arXiv, CrossRef, OpenAlex, Semantic Scholar)
  models/    # Pydantic data models
  utils/     # Shared utilities (SQLite store, dedup, scoring, references, cache)
  web/       # FastAPI web application (templates, static assets, proxy)
tests/       # pytest test suite
config/      # YAML configuration files
```

## Commit Messages

- Use imperative mood: `add feature`, `fix bug`, not `added` or `fixes`
- Keep the first line under 72 characters
- Reference the area of change when helpful: `web: add batch operations`

## Adding a New API Client

1. Create `src/apis/new_api.py` following the pattern in existing clients
2. Implement `search(query, limit, year_range) -> list[Paper]`
3. Register it in `src/agents/daily_researcher.py`
4. Add the API name to `config/config.yaml`
5. Add tests in `tests/`

## Code Conventions

- Python 3.11+ with type hints on all public functions
- Pydantic models for data validation
- All file I/O uses `encoding="utf-8"`
- Line length: 120 characters (configured in ruff)
- SQLite as primary storage; JSON export for compatibility
