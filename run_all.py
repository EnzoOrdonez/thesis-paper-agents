#!/usr/bin/env python3
"""Run the complete pipeline: search + compile.

Usage:
    python run_all.py                    # Full pipeline
    python run_all.py --notify           # With Telegram notifications
    python run_all.py --dry-run          # Simulate without writing
    python run_all.py --search-only      # Only run daily search
    python run_all.py --compile-only     # Only run compiler
    python run_all.py --days 14          # Search last 14 days
    python run_all.py --auto             # Auto mode: run daily at 08:00
    python run_all.py --auto --time 10:30  # Auto mode at custom time
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import schedule
from rich.console import Console

from src.agents.daily_researcher import run_daily_search
from src.agents.paper_compiler import compile_all, load_config
from src.models.paper import RelevanceLevel
from src.utils.logger import setup_logger

logger = setup_logger("run_all")
console = Console()


def send_telegram_notification(papers: list, config: dict) -> None:
    """Send Telegram notification for high-relevance papers."""
    notif_config = config.get("notifications", {})
    if not notif_config.get("enabled"):
        return

    bot_token = notif_config.get("telegram_bot_token") or os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = notif_config.get("telegram_chat_id") or os.getenv("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        logger.warning("Telegram credentials not configured")
        return

    high_papers = [p for p in papers if p.relevance_level.value == "ALTA"]

    if not high_papers and notif_config.get("notify_on_high_relevance"):
        logger.info("No high-relevance papers to notify about")
        return

    try:
        import requests

        for paper in high_papers[:5]:  # Limit to 5 notifications
            doi_link = f"https://doi.org/{paper.doi}" if paper.doi else paper.url or "N/A"
            message = (
                f"Nuevo paper relevante encontrado!\n"
                f"{paper.title}\n"
                f"Autores: {', '.join(paper.authors[:3])}\n"
                f"Ano: {paper.year}\n"
                f"Fuente: {paper.venue}\n"
                f"Relevancia: {paper.relevance_score}/100\n"
                f"Link: {doi_link}"
            )

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
            logger.info(f"Telegram notification sent for: {paper.title}")

    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")


def write_highlights(papers: list) -> None:
    """Write high-relevance papers to output/highlights.md."""
    high_papers = [p for p in papers if p.relevance_level == RelevanceLevel.HIGH]
    if not high_papers:
        return

    highlights_path = Path("output/highlights.md")
    highlights_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Append to existing file
    lines: list[str] = []
    if highlights_path.exists():
        lines.append(highlights_path.read_text(encoding="utf-8"))

    lines.append(f"\n## Highlights - {now}\n")
    lines.append(f"Papers de ALTA relevancia encontrados: {len(high_papers)}\n")

    for p in high_papers[:10]:
        doi_str = f"https://doi.org/{p.doi}" if p.doi else "N/A"
        lines.append(f"### {p.title}")
        lines.append(f"- **Autores:** {', '.join(p.authors[:3])}")
        lines.append(f"- **Ano:** {p.year} | **Score:** {p.relevance_score}/100")
        lines.append(f"- **Categorias:** {', '.join(p.categories)}")
        lines.append(f"- **DOI:** {doi_str}")
        if p.covers_gap:
            lines.append(f"- **Cubre gap:** {p.covers_gap}")
        lines.append("")

    highlights_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Highlights written: {len(high_papers)} papers to {highlights_path}")


def log_auto_execution(papers: list, success: bool, error_msg: str | None = None) -> None:
    """Log an automatic execution result."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "auto_execution.log"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    high_count = sum(1 for p in papers if p.relevance_level == RelevanceLevel.HIGH) if papers else 0

    entry = {
        "timestamp": now,
        "success": success,
        "papers_found": len(papers) if papers else 0,
        "high_relevance": high_count,
        "error": error_msg,
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {'OK' if success else 'ERROR'} | "
                f"Papers: {entry['papers_found']} | "
                f"Alta relevancia: {high_count}"
                + (f" | Error: {error_msg}" if error_msg else "")
                + "\n")


def run_pipeline(days: int, notify: bool) -> None:
    """Execute the full pipeline once (used by both manual and auto modes)."""
    config = load_config()
    papers = []

    try:
        console.print(f"\n[bold cyan]===== Ejecucion: {datetime.now().strftime('%Y-%m-%d %H:%M')} =====[/bold cyan]\n")

        console.print("[bold]Phase 1: Daily Search[/bold]")
        papers = run_daily_search(days=days, dry_run=False)

        if notify:
            send_telegram_notification(papers, config)

        console.print("\n[bold]Phase 2: Compilation[/bold]")
        compile_all(config)

        # Write highlights for high-relevance papers
        write_highlights(papers)

        log_auto_execution(papers, success=True)
        console.print("\n[bold green]===== Pipeline Complete =====[/bold green]")

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        log_auto_execution(papers, success=False, error_msg=str(e))
        console.print(f"\n[bold red]Pipeline error: {e}[/bold red]")


def run_auto_mode(run_time: str, days: int, notify: bool) -> None:
    """Run in automatic scheduled mode."""
    console.print(f"\n[bold cyan]===== Modo Automatico =====[/bold cyan]")
    console.print(f"Programado para ejecutarse diariamente a las [bold]{run_time}[/bold]")
    console.print(f"Presiona [bold]Ctrl+C[/bold] para detener\n")

    schedule.every().day.at(run_time).do(run_pipeline, days=days, notify=notify)

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        console.print("\n[yellow]Modo automatico detenido por el usuario.[/yellow]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete paper search and compilation pipeline.")
    parser.add_argument("--days", type=int, default=7, help="Days back to search (default: 7)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing")
    parser.add_argument("--search-only", action="store_true", help="Only run daily search")
    parser.add_argument("--compile-only", action="store_true", help="Only run compiler")
    parser.add_argument("--notify", action="store_true", help="Send Telegram notifications")
    parser.add_argument("--auto", action="store_true", help="Run automatically on a daily schedule")
    parser.add_argument("--time", type=str, default="08:00", help="Time for auto mode (HH:MM, default: 08:00)")

    args = parser.parse_args()

    if args.auto:
        run_auto_mode(run_time=args.time, days=args.days, notify=args.notify)
        return

    config = load_config()

    console.print("[bold cyan]===== Thesis Paper Agents — Full Pipeline =====[/bold cyan]\n")

    papers = []

    if not args.compile_only:
        console.print("[bold]Phase 1: Daily Search[/bold]")
        papers = run_daily_search(days=args.days, dry_run=args.dry_run)

        if args.notify:
            send_telegram_notification(papers, config)

    if not args.search_only and not args.dry_run:
        console.print("\n[bold]Phase 2: Compilation[/bold]")
        compile_all(config)

    console.print("\n[bold green]===== Pipeline Complete =====[/bold green]")


if __name__ == "__main__":
    main()
