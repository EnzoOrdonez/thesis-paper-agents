from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import yaml
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.agents.daily_researcher import SEARCH_API_ORDER, get_api_runtime_tracker, get_enabled_search_apis, load_config
from src.agents.paper_compiler import CROSSREF_RUNTIME_KEY, OPENALEX_SCOPUS_RUNTIME_KEY, load_database
from src.utils.gap_analyzer import analyze_gap_coverage, load_pending_gaps, load_thesis_categories
from src.utils.monitor_store import (
    ensure_runtime_schema,
    get_dashboard_metrics,
    get_paper_by_id,
    get_runtime_lock,
    list_distinct_filter_values,
    list_job_runs,
    list_papers_paginated,
    list_recent_high_papers,
    refresh_json_export_from_sqlite,
    update_paper_categories,
    update_paper_notes,
    update_paper_status,
)
from src.web.proxy import build_access_links

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
WEB_PROXY_SETTINGS_PATH = PROJECT_ROOT / "data" / "web_proxy_settings.yaml"
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
THESIS_CATEGORIES = load_thesis_categories()
PIPELINE_LOCK_KEY = "pipeline_run"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _load_app_config() -> dict[str, Any]:
    base_config = load_config(str(CONFIG_PATH))
    if WEB_PROXY_SETTINGS_PATH.exists():
        with open(WEB_PROXY_SETTINGS_PATH, encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        if isinstance(override, dict):
            return _deep_merge(base_config, override)
    return base_config


def _write_proxy_override(proxy_config: dict[str, Any]) -> None:
    WEB_PROXY_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"web": {"proxy": proxy_config}}
    with open(WEB_PROXY_SETTINGS_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _reset_proxy_override() -> None:
    if WEB_PROXY_SETTINGS_PATH.exists():
        WEB_PROXY_SETTINGS_PATH.unlink()


def _storage_paths(config: dict[str, Any]) -> tuple[str, str]:
    output = config.get("output", {})
    sqlite_path = str(
        output.get("sqlite_database_path")
        or Path(output.get("database_path", "data/papers_database.json")).with_suffix(".sqlite")
    )
    json_path = str(output.get("database_path", "data/papers_database.json"))
    return sqlite_path, json_path


def _runtime_status_rows(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    tracker = get_api_runtime_tracker(config)
    search_rows: list[dict[str, Any]] = []
    for api_name in SEARCH_API_ORDER:
        provider = tracker.get_provider(api_name)
        search_rows.append(
            {
                "name": api_name,
                "enabled": api_name in set(get_enabled_search_apis(config)),
                "status": provider.get("last_status", "never") or "never",
                "disabled_until": provider.get("disabled_until") or "-",
                "last_run": provider.get("last_run_finished_at") or "-",
                "queries": provider.get("last_queries_submitted", 0) or 0,
                "results": provider.get("last_results_returned", 0) or 0,
                "last_error": provider.get("last_error", "") or "-",
            }
        )

    metadata_rows: list[dict[str, Any]] = []
    for label, key in (
        ("CrossRef DOI validation", CROSSREF_RUNTIME_KEY),
        ("OpenAlex Scopus check", OPENALEX_SCOPUS_RUNTIME_KEY),
    ):
        provider = tracker.get_provider(key)
        metadata_rows.append(
            {
                "name": label,
                "status": provider.get("last_status", "never") or "never",
                "disabled_until": provider.get("disabled_until") or "-",
                "last_run": provider.get("last_run_finished_at") or "-",
                "batch": provider.get("last_queries_submitted", 0) or 0,
                "processed": provider.get("last_results_returned", 0) or 0,
                "last_error": provider.get("last_error", "") or "-",
            }
        )

    return {"search": search_rows, "metadata": metadata_rows}


def _paper_view_model(paper: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(paper)
    enriched["access_links"] = build_access_links(paper, config)
    return enriched


def _render(request: Request, template_name: str, context: dict[str, Any], status_code: int = 200) -> HTMLResponse:
    merged = {"request": request, **context}
    return TEMPLATES.TemplateResponse(template_name, merged, status_code=status_code)


def _default_sort_direction(sort: str) -> str:
    if sort in {"title", "source_api", "status"}:
        return "asc"
    return "desc"


def _normalize_sort_direction(sort: str, sort_dir: str | None) -> str:
    normalized = str(sort_dir or "").strip().lower()
    if normalized in {"asc", "desc"}:
        return normalized
    return _default_sort_direction(sort)


def _build_papers_query(filters: dict[str, Any], sort: str, sort_dir: str, page: int = 1) -> str:
    params: dict[str, Any] = {
        "sort": sort,
        "sort_dir": sort_dir,
        "page": max(1, int(page)),
    }
    for key, value in filters.items():
        if key == "only_pending":
            if value:
                params[key] = "true"
            continue
        if value not in {None, ""}:
            params[key] = value
    return urlencode(params)


def _normalize_proxy_rule(rule: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(rule or {})
    domains = raw.get("domains") or []
    if isinstance(domains, str):
        domains_text = domains
    else:
        domains_text = "\n".join(str(domain) for domain in domains if domain)
    return {
        "name": str(raw.get("name") or ""),
        "strategy": str(raw.get("strategy") or "host_rewrite"),
        "domains_text": domains_text,
        "provider_host": str(raw.get("provider_host") or ""),
        "prefix_url": str(raw.get("prefix_url") or ""),
        "target_template": str(raw.get("target_template") or ""),
        "encode_target": "true" if raw.get("encode_target", True) else "false",
    }


def _blank_proxy_rule() -> dict[str, Any]:
    return _normalize_proxy_rule()


def _proxy_settings_context(config: dict[str, Any]) -> dict[str, Any]:
    proxy_config = config.get("web", {}).get("proxy", {})
    rules = [_normalize_proxy_rule(rule) for rule in proxy_config.get("rules") or []]
    while len(rules) < max(3, len(rules) + 1):
        rules.append(_blank_proxy_rule())
        if len(rules) >= 5:
            break
    return {
        "proxy_mode": str(proxy_config.get("mode") or "dual"),
        "prefer_proxy_button": bool(proxy_config.get("prefer_proxy_button", True)),
        "rules": rules,
        "settings_path": str(WEB_PROXY_SETTINGS_PATH.relative_to(PROJECT_ROOT)),
        "override_enabled": WEB_PROXY_SETTINGS_PATH.exists(),
    }


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "si"}


def _parse_domains(value: str) -> list[str]:
    return [chunk.strip() for chunk in re.split(r"[\r\n,]+", value) if chunk.strip()]


def _build_proxy_rules_from_form(form: Any) -> list[dict[str, Any]]:
    names = [str(item) for item in form.getlist("rule_name")]
    strategies = [str(item) for item in form.getlist("rule_strategy")]
    domains_blocks = [str(item) for item in form.getlist("rule_domains")]
    provider_hosts = [str(item) for item in form.getlist("rule_provider_host")]
    prefix_urls = [str(item) for item in form.getlist("rule_prefix_url")]
    target_templates = [str(item) for item in form.getlist("rule_target_template")]
    encode_targets = [str(item) for item in form.getlist("rule_encode_target")]

    row_count = max(
        len(names),
        len(strategies),
        len(domains_blocks),
        len(provider_hosts),
        len(prefix_urls),
        len(target_templates),
        len(encode_targets),
    )
    rules: list[dict[str, Any]] = []

    for index in range(row_count):
        name = names[index].strip() if index < len(names) else ""
        strategy = strategies[index].strip().lower() if index < len(strategies) else "host_rewrite"
        domains_text = domains_blocks[index] if index < len(domains_blocks) else ""
        provider_host = provider_hosts[index].strip() if index < len(provider_hosts) else ""
        prefix_url = prefix_urls[index].strip() if index < len(prefix_urls) else ""
        target_template = target_templates[index].strip() if index < len(target_templates) else ""
        encode_target = _parse_bool(encode_targets[index] if index < len(encode_targets) else "true", default=True)
        domains = _parse_domains(domains_text)

        has_any_content = any([name, domains_text.strip(), provider_host, prefix_url, target_template])
        if not has_any_content:
            continue
        if not domains:
            continue

        rule: dict[str, Any] = {
            "name": name or f"Proxy rule {len(rules) + 1}",
            "strategy": strategy if strategy in {"host_rewrite", "prefix"} else "host_rewrite",
            "domains": domains,
        }

        if rule["strategy"] == "host_rewrite":
            if not provider_host:
                continue
            rule["provider_host"] = provider_host
        else:
            if prefix_url:
                rule["prefix_url"] = prefix_url
            if target_template:
                rule["target_template"] = target_template
            if not prefix_url and not target_template:
                continue
            rule["encode_target"] = encode_target

        rules.append(rule)

    return rules


def create_app() -> FastAPI:
    config = _load_app_config()
    sqlite_path, json_path = _storage_paths(config)
    ensure_runtime_schema(sqlite_path)

    app = FastAPI(title="Thesis Paper Agents Web", version="1.0.0")
    app.state.config = config
    app.state.config_path = str(CONFIG_PATH)
    app.state.sqlite_path = sqlite_path
    app.state.json_path = json_path
    app.state.project_root = PROJECT_ROOT
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

    @app.get("/health")
    def health() -> JSONResponse:
        lock_info = get_runtime_lock(app.state.sqlite_path, PIPELINE_LOCK_KEY)
        return JSONResponse({"ok": True, "locked": bool(lock_info)})

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        metrics = get_dashboard_metrics(app.state.sqlite_path)
        high_papers = [
            _paper_view_model(item, app.state.config)
            for item in list_recent_high_papers(app.state.sqlite_path, app.state.config, limit=5)
        ]
        jobs = list_job_runs(app.state.sqlite_path, limit=10)
        lock_info = get_runtime_lock(app.state.sqlite_path, PIPELINE_LOCK_KEY)
        gap_status = analyze_gap_coverage(
            load_database(app.state.config["output"]["database_path"], config=app.state.config)
        )
        covered_gaps = sum(1 for gap in gap_status if gap["covered"])
        total_gaps = len(load_pending_gaps())
        return _render(
            request,
            "dashboard.html",
            {
                "title": "Dashboard",
                "metrics": metrics,
                "high_papers": high_papers,
                "jobs": jobs,
                "lock_info": lock_info,
                "runtime": _runtime_status_rows(app.state.config),
                "gap_status": gap_status,
                "covered_gaps": covered_gaps,
                "total_gaps": total_gaps,
                "enabled_search_apis": get_enabled_search_apis(app.state.config),
            },
        )

    @app.get("/papers", response_class=HTMLResponse)
    def papers(
        request: Request,
        q: str = "",
        status: str = "",
        relevance_level: str = "",
        source_api: str = "",
        source_trusted: str = "",
        category: str = "",
        year: str = "",
        doi_verified: str = "",
        scopus_indexed: str = "",
        date_found: str = "",
        only_pending: bool = False,
        sort: str = "ranking_score",
        sort_dir: str = "",
        page: int = 1,
    ) -> HTMLResponse:
        page_size = int(app.state.config.get("web", {}).get("page_size", 50))
        filters = {
            "q": q,
            "status": status,
            "relevance_level": relevance_level,
            "source_api": source_api,
            "source_trusted": source_trusted,
            "category": category,
            "year": year,
            "doi_verified": doi_verified,
            "scopus_indexed": scopus_indexed,
            "date_found": date_found,
            "only_pending": only_pending,
        }
        normalized_sort_dir = _normalize_sort_direction(sort, sort_dir)
        result = list_papers_paginated(
            app.state.sqlite_path,
            app.state.config,
            filters=filters,
            page=page,
            page_size=page_size,
            sort=sort,
            descending=normalized_sort_dir == "desc",
        )
        result["items"] = [_paper_view_model(item, app.state.config) for item in result["items"]]
        context = {
            "title": "Papers",
            "papers_page": result,
            "filters": filters,
            "sort": result["sort"],
            "sort_dir": normalized_sort_dir,
            "papers_query": _build_papers_query,
            "filter_values": list_distinct_filter_values(app.state.sqlite_path),
        }
        if request.headers.get("HX-Request") == "true":
            return _render(request, "_papers_table.html", context)
        return _render(request, "papers.html", context)

    @app.get("/papers/{paper_id}", response_class=HTMLResponse)
    def paper_detail(request: Request, paper_id: str) -> HTMLResponse:
        paper = get_paper_by_id(app.state.sqlite_path, paper_id, app.state.config)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        return _render(
            request,
            "paper_detail.html",
            {
                "title": paper["title"],
                "paper": _paper_view_model(paper, app.state.config),
                "thesis_categories": THESIS_CATEGORIES,
            },
        )

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs(request: Request, message: str = Query(default="")) -> HTMLResponse:
        return _render(
            request,
            "jobs.html",
            {
                "title": "Jobs",
                "jobs": list_job_runs(app.state.sqlite_path, limit=40),
                "lock_info": get_runtime_lock(app.state.sqlite_path, PIPELINE_LOCK_KEY),
                "message": message,
                "enabled_search_apis": get_enabled_search_apis(app.state.config),
            },
        )

    @app.get("/settings/proxy", response_class=HTMLResponse)
    def proxy_settings(request: Request, message: str = Query(default="")) -> HTMLResponse:
        return _render(
            request,
            "proxy_settings.html",
            {
                "title": "Proxy Settings",
                "message": message,
                **_proxy_settings_context(app.state.config),
            },
        )

    @app.post("/settings/proxy")
    async def save_proxy_settings(request: Request) -> RedirectResponse:
        form = await request.form()
        action = str(form.get("action") or "save").strip().lower()

        if action == "reset":
            _reset_proxy_override()
            app.state.config = _load_app_config()
            return RedirectResponse(url="/settings/proxy?message=Configuracion+de+proxy+restablecida.", status_code=303)

        proxy_config = {
            "mode": str(form.get("mode") or "dual").strip() or "dual",
            "prefer_proxy_button": _parse_bool(str(form.get("prefer_proxy_button") or "true"), default=True),
            "rules": _build_proxy_rules_from_form(form),
        }
        _write_proxy_override(proxy_config)
        app.state.config = _load_app_config()
        return RedirectResponse(url="/settings/proxy?message=Configuracion+de+proxy+guardada.", status_code=303)

    @app.post("/papers/{paper_id}/status", response_class=HTMLResponse, response_model=None)
    async def paper_status(request: Request, paper_id: str, status: str = Form(...)) -> HTMLResponse | RedirectResponse:
        paper = update_paper_status(app.state.sqlite_path, paper_id, status, app.state.config)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        refresh_json_export_from_sqlite(app.state.sqlite_path, app.state.json_path)
        context = {
            "paper": _paper_view_model(paper, app.state.config),
            "thesis_categories": THESIS_CATEGORIES,
            "flash": "Estado actualizado.",
        }
        if request.headers.get("HX-Request") == "true":
            return _render(request, "_paper_review_panel.html", context)
        return RedirectResponse(url=f"/papers/{paper_id}", status_code=303)

    @app.post("/papers/{paper_id}/notes", response_class=HTMLResponse, response_model=None)
    async def paper_notes(
        request: Request, paper_id: str, notes: str = Form(default="")
    ) -> HTMLResponse | RedirectResponse:
        paper = update_paper_notes(app.state.sqlite_path, paper_id, notes, app.state.config)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        refresh_json_export_from_sqlite(app.state.sqlite_path, app.state.json_path)
        context = {
            "paper": _paper_view_model(paper, app.state.config),
            "thesis_categories": THESIS_CATEGORIES,
            "flash": "Notas guardadas.",
        }
        if request.headers.get("HX-Request") == "true":
            return _render(request, "_paper_review_panel.html", context)
        return RedirectResponse(url=f"/papers/{paper_id}", status_code=303)

    @app.post("/papers/{paper_id}/categories", response_class=HTMLResponse, response_model=None)
    async def paper_categories(request: Request, paper_id: str) -> HTMLResponse | RedirectResponse:
        form = await request.form()
        categories = [str(item) for item in form.getlist("categories") if item]
        paper = update_paper_categories(app.state.sqlite_path, paper_id, categories, app.state.config)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        refresh_json_export_from_sqlite(app.state.sqlite_path, app.state.json_path)
        context = {
            "paper": _paper_view_model(paper, app.state.config),
            "thesis_categories": THESIS_CATEGORIES,
            "flash": "Categorias actualizadas.",
        }
        if request.headers.get("HX-Request") == "true":
            return _render(request, "_paper_review_panel.html", context)
        return RedirectResponse(url=f"/papers/{paper_id}", status_code=303)

    @app.post("/papers/batch")
    async def papers_batch(request: Request) -> RedirectResponse:
        form = await request.form()
        paper_ids = [str(item) for item in form.getlist("paper_ids") if item]
        action = str(form.get("action") or "").strip()
        if not paper_ids or action not in {"accepted", "rejected", "reviewed"}:
            return RedirectResponse(url="/papers", status_code=303)
        for paper_id in paper_ids:
            update_paper_status(app.state.sqlite_path, paper_id, action, app.state.config)
        refresh_json_export_from_sqlite(app.state.sqlite_path, app.state.json_path)
        return RedirectResponse(url="/papers", status_code=303)

    @app.post("/jobs/run")
    async def run_job(request: Request) -> RedirectResponse:
        form = await request.form()
        phases = [str(item) for item in form.getlist("phases") if item]
        api_names = [str(item) for item in form.getlist("api_names") if item]
        dry_run = str(form.get("dry_run") or "").lower() in {"1", "true", "on", "yes"}
        if not phases:
            return RedirectResponse(url="/jobs?message=Selecciona+al+menos+una+fase.", status_code=303)

        lock_info = get_runtime_lock(app.state.sqlite_path, PIPELINE_LOCK_KEY)
        if lock_info:
            return RedirectResponse(url="/jobs?message=Ya+hay+una+ejecucion+en+curso.", status_code=303)

        command = [sys.executable, str(app.state.project_root / "run_all.py")]
        for phase in phases:
            command.extend(["--phase", phase])
        for api_name in api_names:
            command.extend(["--api", api_name])
        if dry_run:
            command.append("--dry-run")

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(command, cwd=str(app.state.project_root), creationflags=creationflags)
        return RedirectResponse(url="/jobs?message=Ejecucion+lanzada.", status_code=303)

    return app
