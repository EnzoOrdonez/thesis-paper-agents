#!/usr/bin/env python3
"""Entry point for Agent 1: Daily Researcher.

Usage:
    python daily_researcher.py                              # Search last 7 days
    python daily_researcher.py --days 30                    # Search last 30 days
    python daily_researcher.py --dry-run                    # Simulate without writing files
    python daily_researcher.py --api semantic_scholar       # Run only one provider
    python daily_researcher.py --api arxiv --api openalex   # Run a subset of providers
    python daily_researcher.py --api-status                 # Show persisted provider state
    python daily_researcher.py --schedule 08:00             # Run daily at 08:00
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.daily_researcher import get_enabled_search_apis, run_daily_search, show_api_status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily Researcher Agent - Search for new papers related to the thesis."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days back to search (default: 7)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate search without writing output files",
    )
    parser.add_argument(
        "--api",
        action="append",
        choices=get_enabled_search_apis(),
        help="Run only the selected API. Can be repeated.",
    )
    parser.add_argument(
        "--api-status",
        action="store_true",
        help="Show the persisted runtime status for each search API",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        help="Run daily at specified time (HH:MM), e.g. --schedule 08:00",
    )

    args = parser.parse_args()

    if args.api_status:
        show_api_status()
        return

    if args.schedule:
        try:
            import schedule
            import time
            from rich.console import Console

            console = Console()
            console.print(f"[bold cyan]Scheduling daily search at {args.schedule}[/bold cyan]")

            def job() -> None:
                run_daily_search(days=args.days, dry_run=args.dry_run, api_names=args.api)

            schedule.every().day.at(args.schedule).do(job)

            console.print("[dim]Daemon mode active. Press Ctrl+C to stop.[/dim]")
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nScheduler stopped.")
    else:
        run_daily_search(days=args.days, dry_run=args.dry_run, api_names=args.api)


if __name__ == "__main__":
    main()