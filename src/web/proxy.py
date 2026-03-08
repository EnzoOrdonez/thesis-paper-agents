from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit


def direct_paper_url(paper: dict[str, Any]) -> str | None:
    url = str(paper.get("url") or "").strip()
    if url:
        return url
    doi = str(paper.get("doi") or "").strip()
    if doi:
        return f"https://doi.org/{doi}"
    return None


def _normalized_domains(rule: dict[str, Any]) -> set[str]:
    return {str(domain).strip().lower() for domain in rule.get("domains", []) if domain}


def _apply_host_rewrite(url: str, rule: dict[str, Any]) -> str | None:
    provider_host = str(rule.get("provider_host") or "").strip().lower()
    if not provider_host:
        return None

    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if not host:
        return None

    rewritten_host = f"{host.replace('.', '-')}.{provider_host}"
    scheme = parts.scheme or "https"
    return urlunsplit((scheme, rewritten_host, parts.path, parts.query, parts.fragment))


def _apply_prefix(url: str, rule: dict[str, Any]) -> str | None:
    template = str(rule.get("target_template") or "").strip()
    prefix_url = str(rule.get("prefix_url") or "").strip()
    encode_target = bool(rule.get("encode_target", True))
    encoded_url = quote(url, safe="") if encode_target else url

    if template:
        return template.replace("{url}", encoded_url).replace("{raw_url}", url)
    if prefix_url:
        return f"{prefix_url}{encoded_url}"
    return None


def _legacy_rule(proxy_config: dict[str, Any]) -> dict[str, Any] | None:
    provider_host = str(proxy_config.get("provider_host") or "").strip()
    domains = proxy_config.get("supported_domains", [])
    if not provider_host or not domains:
        return None
    return {
        "name": "legacy-proxy",
        "domains": domains,
        "strategy": str(proxy_config.get("strategy") or "host_rewrite"),
        "provider_host": provider_host,
        "prefix_url": proxy_config.get("prefix_url"),
        "target_template": proxy_config.get("target_template"),
        "encode_target": proxy_config.get("encode_target", True),
    }


def proxy_rule_for_url(url: str | None, config: dict[str, Any]) -> dict[str, Any] | None:
    if not url:
        return None

    proxy_config = config.get("web", {}).get("proxy", {})
    rules = proxy_config.get("rules") or []
    if not rules:
        legacy = _legacy_rule(proxy_config)
        rules = [legacy] if legacy else []

    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if not host:
        return None

    for raw_rule in rules:
        rule = dict(raw_rule or {})
        domains = _normalized_domains(rule)
        if host in domains:
            return rule
    return None


def proxied_url(url: str | None, config: dict[str, Any]) -> str | None:
    if not url:
        return None

    rule = proxy_rule_for_url(url, config)
    if not rule:
        return None

    strategy = str(rule.get("strategy") or "host_rewrite").strip().lower()
    if strategy == "host_rewrite":
        return _apply_host_rewrite(url, rule)
    if strategy == "prefix":
        return _apply_prefix(url, rule)
    return None


def build_access_links(paper: dict[str, Any], config: dict[str, Any]) -> dict[str, str | None]:
    direct = direct_paper_url(paper)
    original_url = str(paper.get("url") or "").strip() or None
    rule = proxy_rule_for_url(original_url, config)
    proxy = proxied_url(original_url, config)
    return {
        "direct": direct,
        "proxy": proxy,
        "proxy_rule_name": str(rule.get("name") or "") if rule else None,
    }
