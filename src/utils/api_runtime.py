"""Persistent runtime state for search API ingestion."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_PROVIDER_STATE = {
    "disabled_until": None,
    "consecutive_failures": 0,
    "last_error": "",
    "last_status": "never",
    "last_run_started_at": None,
    "last_run_finished_at": None,
    "last_queries_submitted": 0,
    "last_results_returned": 0,
    "lifetime_queries": 0,
    "lifetime_results": 0,
    "successful_runs": 0,
    "failed_runs": 0,
    "skipped_runs": 0,
}


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


class APIRuntimeTracker:
    """Persist per-provider ingestion health and last-run metrics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"providers": {}}
        try:
            with open(self.path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {"providers": {}}

        if not isinstance(payload, dict):
            return {"providers": {}}

        providers = payload.get("providers")
        if not isinstance(providers, dict):
            payload["providers"] = {}
        return payload

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.state, ensure_ascii=False, indent=2)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
        tmp_path.replace(self.path)

    def get_provider(self, api_name: str) -> dict[str, Any]:
        providers = self.state.setdefault("providers", {})
        provider = providers.setdefault(api_name, {})
        for key, value in DEFAULT_PROVIDER_STATE.items():
            provider.setdefault(key, value)
        return provider

    def apply_to_client(self, api_name: str, client: Any) -> None:
        if hasattr(client, "restore_runtime_state"):
            client.restore_runtime_state(self.get_provider(api_name))

    def mark_started(self, api_name: str, query_count: int) -> None:
        provider = self.get_provider(api_name)
        provider["last_run_started_at"] = _utcnow_iso()
        provider["last_queries_submitted"] = query_count
        provider["last_results_returned"] = 0

    def mark_completed(self, api_name: str, client: Any, query_count: int, results_count: int) -> None:
        provider = self.get_provider(api_name)
        exported = client.export_runtime_state() if hasattr(client, "export_runtime_state") else {}
        provider["disabled_until"] = exported.get("disabled_until")
        provider["consecutive_failures"] = exported.get("consecutive_failures", provider["consecutive_failures"])
        provider["last_error"] = exported.get("last_error", provider["last_error"])
        provider["last_status"] = (
            "cooldown" if provider.get("disabled_until") else ("failed" if provider.get("last_error") else "ok")
        )
        provider["last_run_finished_at"] = _utcnow_iso()
        provider["last_queries_submitted"] = query_count
        provider["last_results_returned"] = results_count
        provider["lifetime_queries"] = int(provider.get("lifetime_queries", 0)) + query_count
        provider["lifetime_results"] = int(provider.get("lifetime_results", 0)) + results_count
        provider["successful_runs"] = int(provider.get("successful_runs", 0)) + 1
        if provider["last_status"] != "ok":
            provider["failed_runs"] = int(provider.get("failed_runs", 0)) + 1

    def mark_skipped(self, api_name: str, client: Any, query_count: int, reason: str = "cooldown") -> None:
        provider = self.get_provider(api_name)
        exported = client.export_runtime_state() if hasattr(client, "export_runtime_state") else {}
        provider["disabled_until"] = exported.get("disabled_until")
        provider["consecutive_failures"] = exported.get("consecutive_failures", provider["consecutive_failures"])
        provider["last_error"] = exported.get("last_error") or reason
        provider["last_status"] = reason
        now = _utcnow_iso()
        provider["last_run_started_at"] = now
        provider["last_run_finished_at"] = now
        provider["last_queries_submitted"] = query_count
        provider["last_results_returned"] = 0
        provider["skipped_runs"] = int(provider.get("skipped_runs", 0)) + 1

    def provider_rows(self, api_names: list[str]) -> list[dict[str, Any]]:
        return [self.get_provider(api_name) for api_name in api_names]
