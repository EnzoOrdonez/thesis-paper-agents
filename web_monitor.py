#!/usr/bin/env python3
"""Launch the local monitoring web UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import uvicorn
except ModuleNotFoundError as exc:
    if exc.name == "uvicorn":
        print("Falta la dependencia 'uvicorn'.")
        print("Instala las dependencias con uno de estos comandos:")
        print("  python -m pip install -r requirements.txt")
        print("  py -m pip install -r requirements.txt")
        print("Si pip no existe en tu entorno, prueba primero:")
        print("  python -m ensurepip --upgrade")
        raise SystemExit(1)
    raise

from src.agents.daily_researcher import load_config


def main() -> None:
    config = load_config()
    web_config = config.get("web", {})

    parser = argparse.ArgumentParser(description="Run the Thesis Paper Agents web monitor.")
    parser.add_argument("--host", default=web_config.get("host", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(web_config.get("port", 8000)))
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    uvicorn.run("src.web.app:create_app", factory=True, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()