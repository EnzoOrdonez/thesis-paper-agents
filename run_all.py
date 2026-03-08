#!/usr/bin/env python3
"""Run the pipeline phases in an orchestrated and schedulable way.

Usage:
    python run_all.py                                      # search + compile + metadata
    python run_all.py --phase search                       # only search phase
    python run_all.py --phase compile                      # only compilation core + finalize
    python run_all.py --phase metadata                     # only metadata validation + finalize
    python run_all.py --phase search --phase compile       # search then compilation core
    python run_all.py --phase search --api openalex        # search a subset of APIs
    python run_all.py --notify                             # send Telegram notifications after search
    python run_all.py --dry-run                            # search only, without writing
    python run_all.py --auto --time 10:30                 # schedule selected phases daily
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import schedule
from rich.console import Console

from src.agents.daily_researcher import get_enabled_search_apis, run_daily_search
from src.agents.paper_compiler import (
    finalize_compiler_outputs,
    load_config,
    run_compilation_phase,
    run_metadata_phase,
)
from src.models.paper import RelevanceLevel
from src.utils.logger import setup_logger
from src.utils.monitor_store import acquire_runtime_lock, create_job_run, finish_job_run, release_runtime_lock

logger = setup_logger("run_all")
console = Console()

PHASE_ORDER = ("search", "compile", "metadata")
PHASE_LOCK_KEY = "pipeline_run"


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

    high_papers = [paper for paper in papers if paper.relevance_level.value == "ALTA"]

    if not high_papers and notif_config.get("notify_on_high_relevance"):
        logger.info("No high-relevance papers to notify about")
        return

    try:
        import requests

        for paper in high_papers[:5]:
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
            logger.info("Telegram notification sent for: %s", paper.title)

    except Exception as exc:
        logger.error("Failed to send Telegram notification: %s", exc)


def write_highlights(papers: list) -> None:
    """Write high-relevance papers to output/highlights.md."""
    high_papers = [paper for paper in papers if paper.relevance_level == RelevanceLevel.HIGH]
    if not high_papers:
        return

    highlights_path = Path("output/highlights.md")
    highlights_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    if highlights_path.exists():
        lines.append(highlights_path.read_text(encoding="utf-8"))

    lines.append(f"\n## Highlights - {now}\n")
    lines.append(f"Papers de ALTA relevancia encontrados: {len(high_papers)}\n")

    for paper in high_papers[:10]:
        doi_str = f"https://doi.org/{paper.doi}" if paper.doi else "N/A"
        lines.append(f"### {paper.title}")
        lines.append(f"- **Autores:** {', '.join(paper.authors[:3])}")
        lines.append(f"- **Ano:** {paper.year} | **Score:** {paper.relevance_score}/100")
        lines.append(f"- **Categorias:** {', '.join(paper.categories)}")
        lines.append(f"- **DOI:** {doi_str}")
        if paper.covers_gap:
            lines.append(f"- **Cubre gap:** {paper.covers_gap}")
        lines.append("")

    highlights_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Highlights written: %s papers to %s", len(high_papers), highlights_path)


def log_auto_execution(
    phases: list[str],
    papers: list,
    success: bool,
    error_msg: str | None = None,
) -> None:
    """Log an automatic execution result."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "auto_execution.log"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    high_count = sum(1 for paper in papers if paper.relevance_level == RelevanceLevel.HIGH) if papers else 0

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"[{now}] {'OK' if success else 'ERROR'} | "
            f"Phases: {','.join(phases)} | "
            f"Papers: {len(papers) if papers else 0} | "
            f"Alta relevancia: {high_count}" + (f" | Error: {error_msg}" if error_msg else "") + "\n"
        )


def _resolve_phases(args: argparse.Namespace) -> list[str]:
    if args.search_only:
        return ["search"]
    if args.compile_only:
        return ["compile", "metadata"]
    if args.metadata_only:
        return ["metadata"]
    if args.phase:
        requested = set(args.phase)
        return [phase for phase in PHASE_ORDER if phase in requested]
    return list(PHASE_ORDER)


def _phase_title(index: int, name: str) -> str:
    return f"[bold]Phase {index}: {name}[/bold]"


def _runtime_storage_path(config: dict) -> str:
    output = config.get("output", {})
    sqlite_path = output.get("sqlite_database_path")
    if sqlite_path:
        return str(sqlite_path)
    return str(Path(output.get("database_path", "data/papers_database.json")).with_suffix(".sqlite"))


def run_selected_phases(
    phases: list[str],
    days: int,
    notify: bool,
    dry_run: bool,
    api_names: list[str] | None = None,
    record_log: bool = False,
) -> dict[str, Any]:
    """Execute the selected pipeline phases once."""
    config = load_config()
    runtime_path = _runtime_storage_path(config)
    lock_owner = str(uuid.uuid4())
    papers: list = []
    search_summary: dict[str, Any] = {}
    compile_summary: dict[str, Any] = {}
    metadata_summary: dict[str, Any] = {}
    finalize_summary: dict[str, Any] = {}

    acquired, lock_info = acquire_runtime_lock(
        runtime_path,
        lock_key=PHASE_LOCK_KEY,
        owner_id=lock_owner,
        ttl_seconds=4 * 60 * 60,
    )
    if not acquired:
        owner = lock_info.get("owner_id") if lock_info else "unknown"
        expires = lock_info.get("expires_at") if lock_info else "unknown"
        console.print(f"\n[bold red]Pipeline busy: another run is active ({owner}, lock until {expires}).[/bold red]")
        return {
            "search": search_summary,
            "compile": compile_summary,
            "metadata": metadata_summary,
            "finalize": finalize_summary,
            "status": "blocked",
        }

    job_id = create_job_run(
        runtime_path,
        trigger="scheduled" if record_log else "manual",
        phases=phases,
        api_names=api_names,
        dry_run=dry_run,
        job_id=lock_owner,
    )

    try:
        console.print(f"\n[bold cyan]===== Execution: {datetime.now().strftime('%Y-%m-%d %H:%M')} =====[/bold cyan]\n")
        console.print(f"Selected phases: {', '.join(phases)}")
        if api_names:
            console.print(f"Selected search APIs: {', '.join(api_names)}")

        phase_index = 1

        if "search" in phases:
            console.print(f"\n{_phase_title(phase_index, 'Search')}\n")
            papers, search_summary = run_daily_search(days=days, dry_run=dry_run, api_names=api_names)
            phase_index += 1

            if notify and not dry_run:
                send_telegram_notification(papers, config)
            if not dry_run:
                write_highlights(papers)

        write_phases = [phase for phase in phases if phase in {"compile", "metadata"}]
        if dry_run and write_phases:
            console.print(f"\n[yellow]Dry run active. Skipping write phases: {', '.join(write_phases)}[/yellow]")
        else:
            if "compile" in phases:
                console.print(f"\n{_phase_title(phase_index, 'Compilation Core')}\n")
                compile_summary = run_compilation_phase(config, finalize_outputs=False)
                phase_index += 1

            if "metadata" in phases:
                console.print(f"\n{_phase_title(phase_index, 'Metadata Validation')}\n")
                metadata_summary = run_metadata_phase(config, finalize_outputs=False)
                phase_index += 1

            if write_phases:
                console.print(f"\n{_phase_title(phase_index, 'Finalize Outputs')}\n")
                finalize_summary = finalize_compiler_outputs(
                    config,
                    reports_step="Generating reports...",
                    sync_step="Syncing {storage_label}...",
                )

        finish_job_run(
            runtime_path,
            job_id,
            status="success",
            search_summary=search_summary,
            compile_summary=compile_summary,
            metadata_summary=metadata_summary,
        )
        if record_log:
            log_auto_execution(phases, papers, success=True)
        console.print("\n[bold green]===== Pipeline Complete =====[/bold green]")
        return {
            "search": search_summary,
            "compile": compile_summary,
            "metadata": metadata_summary,
            "finalize": finalize_summary,
            "status": "success",
        }

    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        finish_job_run(
            runtime_path,
            job_id,
            status="failed",
            error_message=str(exc),
            search_summary=search_summary,
            compile_summary=compile_summary,
            metadata_summary=metadata_summary,
        )
        if record_log:
            log_auto_execution(phases, papers, success=False, error_msg=str(exc))
        console.print(f"\n[bold red]Pipeline error: {exc}[/bold red]")
        return {
            "search": search_summary,
            "compile": compile_summary,
            "metadata": metadata_summary,
            "finalize": finalize_summary,
            "status": "failed",
            "error": str(exc),
        }
    finally:
        release_runtime_lock(runtime_path, lock_key=PHASE_LOCK_KEY, owner_id=lock_owner)


def run_auto_mode(
    run_time: str,
    phases: list[str],
    days: int,
    notify: bool,
    dry_run: bool,
    api_names: list[str] | None = None,
) -> None:
    """Run in automatic scheduled mode."""
    console.print("\n[bold cyan]===== Automatic Mode =====[/bold cyan]")
    console.print(f"Scheduled daily at [bold]{run_time}[/bold]")
    console.print(f"Selected phases: [bold]{', '.join(phases)}[/bold]")
    if api_names:
        console.print(f"Selected search APIs: [bold]{', '.join(api_names)}[/bold]")
    console.print("Press [bold]Ctrl+C[/bold] to stop\n")

    schedule.every().day.at(run_time).do(
        run_selected_phases,
        phases=phases,
        days=days,
        notify=notify,
        dry_run=dry_run,
        api_names=api_names,
        record_log=True,
    )

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        console.print("\n[yellow]Automatic mode stopped by the user.[/yellow]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the thesis paper pipeline in selectable phases.")
    parser.add_argument("--days", type=int, default=7, help="Days back to search (default: 7)")
    parser.add_argument("--dry-run", action="store_true", help="Run search without writing and skip write phases")
    parser.add_argument("--phase", action="append", choices=PHASE_ORDER, help="Pipeline phase to run. Can be repeated.")
    parser.add_argument(
        "--api", action="append", choices=get_enabled_search_apis(), help="Restrict search phase to one or more APIs"
    )
    parser.add_argument("--search-only", action="store_true", help="Legacy alias for --phase search")
    parser.add_argument("--compile-only", action="store_true", help="Legacy alias for --phase compile --phase metadata")
    parser.add_argument("--metadata-only", action="store_true", help="Only run the metadata validation phase")
    parser.add_argument("--notify", action="store_true", help="Send Telegram notifications after the search phase")
    parser.add_argument("--auto", action="store_true", help="Run automatically on a daily schedule")
    parser.add_argument("--time", type=str, default="08:00", help="Time for auto mode (HH:MM, default: 08:00)")

    args = parser.parse_args()
    phases = _resolve_phases(args)

    if args.auto:
        run_auto_mode(
            run_time=args.time,
            phases=phases,
            days=args.days,
            notify=args.notify,
            dry_run=args.dry_run,
            api_names=args.api,
        )
        return

    console.print("[bold cyan]===== Thesis Paper Agents - Phase Runner =====[/bold cyan]")
    run_selected_phases(
        phases=phases,
        days=args.days,
        notify=args.notify,
        dry_run=args.dry_run,
        api_names=args.api,
    )


if __name__ == "__main__":
    main()
