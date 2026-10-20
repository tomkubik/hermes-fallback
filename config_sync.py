"""Configuration management, heuristic settings, and zero-LLM-call orchestrator sync.

Provides:
1. Backward-compatible ``get_fallback_chain``, ``get_stored_fallback_rules``,
   ``is_fallback_heuristics_enabled``, and ``scoped_fallback_chain``.
2. Standalone heuristic plugin settings management (``get_plugin_settings``,
   ``update_plugin_settings``, ``build_criteria_from_settings``) that works
   completely without an LLM call.
3. Automatic materialization of heuristic candidates from the user's **existing
   authenticated/configured providers** into ``config.yaml``'s ``fallback_providers``
   (``sync_fallback_chain_to_config``), solving the problem of undefined or
   hardcoded-and-unavailable fallback models.
4. Direct provider + model switching via heuristic criteria (``resolve_best_available_model``,
   ``switch_model_with_heuristic``, ``ensure_available_model_or_fallback``).
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

try:
    from .heuristics import (
        MULTI_PROVIDER_TOKENS,
        FallbackCriteria,
        _get_hermes_home,
        _parse_param_size_spec,
        _parse_token_count_spec,
        _read_secret_or_env,
        discover_existing_providers,
        filter_and_rank_candidates,
        get_candidate_models,
        is_provider_available,
        normalize_provider_name,
        parse_fallback_criteria,
        parse_provider_list,
        resolve_fallback_entry,
    )
except ImportError:
    from heuristics import (  # type: ignore
        MULTI_PROVIDER_TOKENS,
        FallbackCriteria,
        _get_hermes_home,
        _parse_param_size_spec,
        _parse_token_count_spec,
        _read_secret_or_env,
        discover_existing_providers,
        filter_and_rank_candidates,
        get_candidate_models,
        is_provider_available,
        normalize_provider_name,
        parse_fallback_criteria,
        parse_provider_list,
        resolve_fallback_entry,
    )

logger = logging.getLogger(__name__)

PLUGIN_ID = "model-fallback-heuristic"

DEFAULT_HEURISTIC_SETTINGS: dict[str, Any] = {
    "enabled": True,
    "sort_by": "largest_parameter_count",
    "provider": "auto",
    "providers": [],
    "exclude_providers": [],
    "allow_provider_switch": True,
    "prefer_current_provider": False,
    "free_only": False,
    "require_tools": True,
    "flash_only": False,
    "min_context": 0,
    "max_context": 0,
    "min_params": 0.0,
    "max_params": 0.0,
    "vendor": "",
    "name_filter": "",
    "reasoning_effort": "",
    "max_candidates": 3,
    "auto_sync_config": True,
    "auto_switch_primary_when_unavailable": True,
    "first_launch_completed": False,
}

DEFAULT_PLUGIN_SETTINGS = DEFAULT_HEURISTIC_SETTINGS


def _normalized_base_url(value: Any) -> str:
    return value.strip().rstrip("/") if isinstance(value, str) else ""


def _normalize_provider_id(value: str) -> str:
    """Collapse provider aliases to the canonical id."""
    try:
        from hermes_cli.providers import normalize_provider

        return normalize_provider(value)
    except Exception:
        return normalize_provider_name(value) or value.lower()


def load_hermes_config() -> dict[str, Any]:
    """Load config.yaml via hermes_cli.config.load_config or direct YAML fallback."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        pass

    try:
        cfg_path = _get_hermes_home() / "config.yaml"
        if cfg_path.is_file():
            try:
                import hermes_yaml as yaml
            except ImportError:
                import yaml  # type: ignore[no-redef]
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                return loaded
    except Exception as exc:
        logger.debug("Fallback config.yaml read failed: %s", exc)
    return {}


def save_hermes_config(config: dict[str, Any]) -> None:
    """Save config.yaml via hermes_cli.config.save_config or direct YAML fallback."""
    try:
        from hermes_cli.config import save_config

        save_config(config)
        return
    except Exception:
        pass

    try:
        cfg_path = _get_hermes_home() / "config.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import hermes_yaml as yaml
        except ImportError:
            import yaml  # type: ignore[no-redef]
        content = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        cfg_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write config.yaml: %s", exc)


def resolve_entry_api_key(entry: dict[str, Any] | None) -> str | None:
    """API key for one fallback entry: inline ``api_key``, else ``key_env``."""
    if not isinstance(entry, dict):
        return None
    if inline := str(entry.get("api_key") or "").strip():
        return inline
    if key_env := str(entry.get("key_env") or entry.get("api_key_env") or "").strip():
        return _read_secret_or_env(key_env) or None
    return None


def effective_runtime_provider(
    entry: dict[str, Any] | None, runtime: dict[str, Any] | None
) -> str:
    """Provider identity to persist/display for a resolved fallback entry."""
    runtime = runtime or {}
    resolved = str(runtime.get("provider") or "").strip()
    if resolved.lower() != "custom":
        return resolved
    requested = str(
        runtime.get("requested_provider") or (entry or {}).get("provider") or ""
    ).strip()
    if requested and requested.lower() != "custom":
        return requested
    return resolved


def pre_agent_fallback_notice(
    primary_provider: Any, primary_model: Any, fallback_provider: Any, fallback_model: Any
) -> str:
    """User-visible one-shot line for a provider switch made during credential resolution."""
    primary_desc = (
        "/".join(str(p).strip() for p in (primary_provider, primary_model) if p) or "primary"
    )
    fallback_desc = (
        "/".join(str(p).strip() for p in (fallback_provider, fallback_model) if p) or "fallback"
    )
    return f"⚠️ Provider fallback: {primary_desc} unavailable; using {fallback_desc} for this response."


# ─── Heuristic Enablement & Plugin Settings ─────────────────────────────────


def is_fallback_heuristics_enabled(
    config: dict[str, Any] | None = None,
    entry: dict[str, Any] | None = None,
) -> bool:
    """Check whether dynamic fallback heuristics are enabled in config or entry."""
    if entry and isinstance(entry, dict):
        if entry.get("heuristic_enabled") is True or entry.get("enable_heuristic") is True:
            return True
        if entry.get("heuristic_enabled") is False or entry.get("enable_heuristic") is False:
            return False

    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly() or {}
        except Exception:
            config = load_hermes_config()

    if not isinstance(config, dict):
        return False

    if "fallback_heuristics" in config:
        return bool(config["fallback_heuristics"])
    if "fallback_heuristics_enabled" in config:
        return bool(config["fallback_heuristics_enabled"])

    agent_cfg = config.get("agent")
    if isinstance(agent_cfg, dict):
        if "fallback_heuristics" in agent_cfg:
            return bool(agent_cfg["fallback_heuristics"])
        if "fallback_heuristics_enabled" in agent_cfg:
            return bool(agent_cfg["fallback_heuristics_enabled"])

    fallback_cfg = config.get("fallback")
    if isinstance(fallback_cfg, dict):
        if "heuristics" in fallback_cfg:
            return bool(fallback_cfg["heuristics"])
        if "heuristics_enabled" in fallback_cfg:
            return bool(fallback_cfg["heuristics_enabled"])

    # Also check plugin settings entry if configured
    plugins_cfg = config.get("plugins")
    if isinstance(plugins_cfg, dict):
        entries_cfg = plugins_cfg.get("entries")
        if isinstance(entries_cfg, dict):
            p_entry = entries_cfg.get(PLUGIN_ID)
            if isinstance(p_entry, dict):
                settings = p_entry.get("settings") or p_entry.get("config") or {}
                if isinstance(settings, dict) and "enabled" in settings:
                    return bool(settings["enabled"])

    return False


def get_plugin_settings(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Read effective heuristic settings from config.yaml without making any model call."""
    cfg = config if isinstance(config, dict) else load_hermes_config()
    merged = copy.deepcopy(DEFAULT_HEURISTIC_SETTINGS)

    # Top-level fallback_heuristic_settings or fallback.heuristic_settings
    top_settings = cfg.get("fallback_heuristic_settings")
    if isinstance(top_settings, dict):
        merged.update({k: v for k, v in top_settings.items() if v is not None})

    # Standard Hermes plugin settings location: plugins.entries.<PLUGIN_ID>.settings
    plugins_cfg = cfg.get("plugins")
    if isinstance(plugins_cfg, dict):
        entries = plugins_cfg.get("entries")
        if isinstance(entries, dict):
            p_entry = entries.get(PLUGIN_ID)
            if isinstance(p_entry, dict):
                for subkey in ("config", "settings"):
                    sdict = p_entry.get(subkey)
                    if isinstance(sdict, dict):
                        merged.update({k: v for k, v in sdict.items() if v is not None})

    # Honor explicit top-level fallback_heuristics boolean if present
    if "fallback_heuristics" in cfg:
        merged["enabled"] = bool(cfg["fallback_heuristics"])

    merged["providers"] = list(parse_provider_list(merged.get("providers")))
    merged["exclude_providers"] = list(parse_provider_list(merged.get("exclude_providers")))
    return merged


def update_plugin_settings(
    updates: dict[str, Any],
    config: Optional[dict[str, Any]] = None,
    save: bool = True,
    sync_chain: bool = True,
) -> dict[str, Any]:
    """Update heuristic settings in config.yaml and optionally re-sync fallback_providers (no model call)."""
    cfg = config if isinstance(config, dict) else load_hermes_config()
    plugins_cfg = cfg.setdefault("plugins", {})
    if not isinstance(plugins_cfg, dict):
        plugins_cfg = {}
        cfg["plugins"] = plugins_cfg
    entries_cfg = plugins_cfg.setdefault("entries", {})
    if not isinstance(entries_cfg, dict):
        entries_cfg = {}
        plugins_cfg["entries"] = entries_cfg
    p_entry = entries_cfg.setdefault(PLUGIN_ID, {"enabled": True})
    if not isinstance(p_entry, dict):
        p_entry = {"enabled": True}
        entries_cfg[PLUGIN_ID] = p_entry
    settings = p_entry.setdefault("settings", {})
    if not isinstance(settings, dict):
        settings = {}
        p_entry["settings"] = settings

    # Normalize incoming keys
    key_aliases = {
        "heuristic": "sort_by",
        "strategy": "sort_by",
        "model_provider": "provider",
        "free": "free_only",
        "tools": "require_tools",
        "flash": "flash_only",
        "filter": "name_filter",
        "pattern": "name_filter",
        "lab": "vendor",
        "min_ctx": "min_context",
        "max_ctx": "max_context",
        "min_params_b": "min_params",
        "max_params_b": "max_params",
        "switch_provider": "allow_provider_switch",
        "thinking": "reasoning_effort",
        "reasoning": "reasoning_effort",
        "limit": "max_candidates",
    }

    for raw_k, raw_v in updates.items():
        if raw_v is None:
            continue
        k = key_aliases.get(raw_k, raw_k)
        if k in ("providers", "exclude_providers"):
            settings[k] = list(parse_provider_list(raw_v))
        elif k == "provider":
            if isinstance(raw_v, (list, tuple)):
                provs = list(parse_provider_list(raw_v))
                settings["providers"] = provs
                settings["provider"] = provs[0] if provs else "auto"
            else:
                s_val = str(raw_v).strip().lower()
                if "," in s_val or "|" in s_val:
                    provs = list(parse_provider_list(s_val))
                    settings["providers"] = provs
                    settings["provider"] = provs[0] if provs else "auto"
                else:
                    settings["provider"] = normalize_provider_name(s_val) or s_val
        elif k in (
            "enabled",
            "allow_provider_switch",
            "prefer_current_provider",
            "free_only",
            "require_tools",
            "flash_only",
            "auto_sync_config",
            "auto_switch_primary_when_unavailable",
            "first_launch_completed",
        ):
            if isinstance(raw_v, str):
                settings[k] = raw_v.strip().lower() in ("true", "1", "yes", "on")
            else:
                settings[k] = bool(raw_v)
        elif k in ("min_context", "max_context", "max_candidates"):
            settings[k] = (
                max(1, int(raw_v))
                if k == "max_candidates"
                else _parse_token_count_spec(raw_v)
            )
        elif k in ("min_params", "max_params"):
            settings[k] = _parse_param_size_spec(raw_v)
        else:
            settings[k] = raw_v

    if "enabled" in settings:
        cfg["fallback_heuristics"] = bool(settings["enabled"])

    effective = get_plugin_settings(cfg)
    if sync_chain and effective.get("auto_sync_config", True) and effective.get("enabled", True):
        sync_fallback_chain_to_config(cfg, save=False)

    if save:
        save_hermes_config(cfg)

    return effective


def build_criteria_from_settings(
    config_or_settings: Optional[dict[str, Any]] = None,
    overrides: Optional[dict[str, Any] | str | FallbackCriteria] = None,
    *,
    config: Optional[dict[str, Any]] = None,
) -> FallbackCriteria:
    """Construct a FallbackCriteria combining plugin settings and optional caller overrides."""
    if isinstance(overrides, FallbackCriteria):
        return overrides

    if config is not None and isinstance(config_or_settings, dict):
        cfg = config
        settings = {**copy.deepcopy(DEFAULT_HEURISTIC_SETTINGS), **config_or_settings}
    elif isinstance(config_or_settings, dict) and any(
        k in config_or_settings for k in ("sort_by", "min_params", "min_context", "allow_provider_switch")
    ) and "plugins" not in config_or_settings and "model" not in config_or_settings:
        cfg = load_hermes_config()
        settings = {**copy.deepcopy(DEFAULT_HEURISTIC_SETTINGS), **config_or_settings}
    else:
        cfg = config_or_settings if isinstance(config_or_settings, dict) else load_hermes_config()
        settings = get_plugin_settings(cfg)

    current_primary_provider = ""
    if isinstance(cfg.get("model"), dict):
        current_primary_provider = normalize_provider_name(cfg["model"].get("provider"))

    spec_dict: dict[str, Any] = {
        "sort_by": settings.get("sort_by", "largest_parameter_count"),
        "provider": settings.get("provider", "auto") or "auto",
        "providers": settings.get("providers") or [],
        "exclude_providers": settings.get("exclude_providers") or [],
        "allow_provider_switch": settings.get("allow_provider_switch", True),
        "prefer_current_provider": settings.get("prefer_current_provider", False),
        "current_provider": current_primary_provider or None,
        "free_only": settings.get("free_only", False),
        "require_tools": settings.get("require_tools", True),
        "flash_only": settings.get("flash_only", False),
        "min_context": settings.get("min_context", 0),
        "max_context": settings.get("max_context", 0),
        "min_params": settings.get("min_params", 0.0),
        "max_params": settings.get("max_params", 0.0),
        "vendor": settings.get("vendor", ""),
        "name_filter": settings.get("name_filter", ""),
        "max_candidates": settings.get("max_candidates", 3),
    }
    if settings.get("reasoning_effort"):
        spec_dict["reasoning_effort"] = settings["reasoning_effort"]

    if isinstance(overrides, str) and overrides.strip():
        spec_dict["heuristic"] = overrides.strip()
        # Let explicit shorthand tokens in overrides win over default settings
        parsed_override = parse_fallback_criteria(overrides.strip())
        spec_dict["sort_by"] = parsed_override.sort_by
        if parsed_override.provider != "nous" or overrides.strip().lower().startswith("nous"):
            spec_dict["provider"] = parsed_override.provider
        if parsed_override.providers:
            spec_dict["providers"] = list(parsed_override.providers)
        if parsed_override.exclude_providers:
            spec_dict["exclude_providers"] = list(parsed_override.exclude_providers)
        if "free" in overrides.lower():
            spec_dict["free_only"] = parsed_override.free_only
        if parsed_override.flash_only:
            spec_dict["flash_only"] = True
        if parsed_override.vendor:
            spec_dict["vendor"] = parsed_override.vendor
        if parsed_override.name_filter:
            spec_dict["name_filter"] = parsed_override.name_filter
        if parsed_override.min_context:
            spec_dict["min_context"] = parsed_override.min_context
        if parsed_override.max_context:
            spec_dict["max_context"] = parsed_override.max_context
        if parsed_override.min_params:
            spec_dict["min_params"] = parsed_override.min_params
        if parsed_override.max_params:
            spec_dict["max_params"] = parsed_override.max_params
        if parsed_override.reasoning_effort is not None:
            spec_dict["reasoning_effort"] = parsed_override.reasoning_effort
    elif isinstance(overrides, dict):
        for k, v in overrides.items():
            if v is not None:
                spec_dict[k] = v

    return parse_fallback_criteria(spec_dict)


# ─── Fallback Chain Iteration & Stored Rules (F1-F11 Compatible) ────────────


def _iter_fallback_entries(raw: Any, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    candidates = [raw] if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or entry.get("model_provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        is_already_materialized = bool(entry.get("_is_heuristic") or entry.get("criteria_matched"))
        has_criteria = not is_already_materialized and (
            bool(entry.get("criteria") or entry.get("heuristic") or entry.get("providers"))
            or model.startswith(("auto:", "heuristic:", "criteria:"))
        )
        if not provider and has_criteria:
            try:
                crit = parse_fallback_criteria(entry)
                provider = crit.provider or "nous"
            except (ValueError, TypeError, KeyError):
                provider = "nous"
            entry = {**entry, "provider": provider}
        if not provider:
            continue
        if has_criteria:
            heuristics_active = is_fallback_heuristics_enabled(config, entry)
            if not heuristics_active:
                logger.debug(
                    "Fallback heuristic on provider %s skipped because fallback heuristics are disabled. "
                    "Enable via 'fallback_heuristics: true' in config.yaml or 'hermes fallback-heuristic on'.",
                    provider,
                )
                if model and not model.startswith(("auto:", "heuristic:", "criteria:")):
                    normalized = {**entry, "provider": provider, "model": model}
                    normalized.pop("criteria", None)
                    normalized.pop("heuristic", None)
                    base_url = _normalized_base_url(entry.get("base_url"))
                    if base_url:
                        normalized["base_url"] = base_url
                    entries.append(normalized)
                continue

            try:
                resolved_entries = resolve_fallback_entry(entry, config=config)
            except Exception as exc:
                logger.warning(
                    "Failed to resolve fallback criteria for provider %s: %s",
                    provider,
                    exc,
                    exc_info=True,
                )
                resolved_entries = []
            for res in resolved_entries:
                res_provider = str(res.get("provider") or "").strip()
                res_model = str(res.get("model") or "").strip()
                if not res_provider or not res_model:
                    continue
                normalized = {**res, "provider": res_provider, "model": res_model}
                base_url = _normalized_base_url(res.get("base_url"))
                if base_url:
                    normalized["base_url"] = base_url
                entries.append(normalized)
            continue
        if not model:
            continue
        normalized = {**entry, "provider": provider, "model": model}
        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url
        entries.append(normalized)

    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for e in entries:
        ident = _entry_identity(e)
        if ident not in seen:
            seen.add(ident)
            deduped.append(e)
    return deduped


def get_stored_fallback_rules(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the raw, declarative fallback rules stored in config.

    Preserves inactive heuristic rules, does not query catalogs or materialize candidates,
    iterating over both fallback_providers and fallback_model, deduping by identity and
    marking legacy items with ``_from_fallback_model = True``.
    """
    if not isinstance(config, dict):
        return []

    rules: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    # If fallback_heuristic_rules holds the unmaterialized source rules while fallback_providers
    # has plugin-materialized entries, include them alongside user static entries.
    fp_raw = config.get("fallback_providers")
    fp_candidates = (
        [fp_raw] if isinstance(fp_raw, dict) else fp_raw if isinstance(fp_raw, list) else []
    )
    for entry in fp_candidates:
        if isinstance(entry, dict):
            if entry.get("_managed_by_heuristic_plugin") and config.get("fallback_heuristic_rules"):
                continue
            ident = _entry_identity(entry)
            if ident not in seen:
                seen.add(ident)
                rules.append(copy.deepcopy(entry))

    hr_raw = config.get("fallback_heuristic_rules")
    if isinstance(hr_raw, list):
        for entry in hr_raw:
            if isinstance(entry, dict):
                ident = _entry_identity(entry)
                if ident not in seen:
                    seen.add(ident)
                    rules.append(copy.deepcopy(entry))

    fm_raw = config.get("fallback_model")
    fm_candidates = (
        [fm_raw] if isinstance(fm_raw, dict) else fm_raw if isinstance(fm_raw, list) else []
    )
    for entry in fm_candidates:
        if isinstance(entry, dict):
            ident = _entry_identity(entry)
            if ident not in seen:
                seen.add(ident)
                e_copy = copy.deepcopy(entry)
                e_copy["_from_fallback_model"] = True
                rules.append(e_copy)

    return rules


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    heur_tag = ""
    if not entry.get("model") and (entry.get("heuristic") or entry.get("criteria")):
        heur_tag = str(entry.get("heuristic") or entry.get("criteria")).strip().lower()
    return (
        _normalize_provider_id(
            str(entry.get("provider") or entry.get("model_provider") or "").strip()
        ),
        (str(entry.get("model") or "").strip().lower()) or f"__rule__:{heur_tag}",
        _normalized_base_url(entry.get("base_url")).lower(),
    )


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys."""
    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key), config=config):
            identity = _entry_identity(entry)
            if identity not in seen:
                seen.add(identity)
                chain.append(entry)
    return chain


def scoped_fallback_chain(
    inherited: list[dict[str, Any]] | None,
    declared: Any,
    *,
    pinned: bool,
    owner: str,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    """Fallback chain for a route owner that can pin its own primary (delegated child, cron job)."""
    default = None if pinned else (inherited or None)
    if declared is None:
        return default
    if declared == []:
        return None
    if config is None:
        config = load_hermes_config()
    synth_cfg = {**config, "fallback_providers": declared}
    normalized = get_fallback_chain(synth_cfg)
    if not normalized:
        logger.warning(
            "%s fallback_providers has no usable routes; using the %s default",
            owner,
            "pinned" if pinned else "inherited",
        )
    return normalized or default


# ─── Materialization & Orchestrator Failover Sync (Zero LLM Calls) ──────────


def sync_fallback_chain_to_config(
    config: Optional[dict[str, Any]] = None,
    *,
    save: bool = True,
    exclude_providers: tuple[str, ...] | list[str] = (),
    exclude_models: tuple[str, ...] | list[str] = (),
) -> list[dict[str, Any]]:
    """Materialize heuristic fallback candidates into ``config['fallback_providers']``.

    Why this matters for a standalone Hermes plugin:
    Vanilla Hermes Core's ``get_fallback_chain(config)`` requires concrete
    ``{"provider": "<p>", "model": "<m>"}`` entries in ``fallback_providers`` and re-reads
    ``config.yaml`` at startup (``resolve_runtime_with_fallback``) and before every turn
    (``_sync_fallback_chain_with_config``). By materializing resolved candidates from the
    user's existing authenticated providers into ``fallback_providers`` (while preserving
    declarative rules in ``fallback_heuristic_rules``), Hermes Core's orchestrator seamlessly
    uses the heuristic models even when the primary model is unavailable before any LLM call!
    """
    cfg = config if isinstance(config, dict) else load_hermes_config()
    settings = get_plugin_settings(cfg)
    heuristics_on = bool(settings.get("enabled", True)) and is_fallback_heuristics_enabled(
        {**cfg, "fallback_heuristics": settings.get("enabled", True)}
    )

    # Identify current primary model/provider so we don't duplicate the primary as the first fallback
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    primary_provider = normalize_provider_name(model_cfg.get("provider"))
    primary_model = str(model_cfg.get("default") or "").strip()

    excl_provs = set(parse_provider_list(exclude_providers)) | set(
        parse_provider_list(settings.get("exclude_providers"))
    )
    excl_models = {str(m).strip().lower() for m in exclude_models if str(m).strip()}

    raw_fp = cfg.get("fallback_providers")
    fp_list = [raw_fp] if isinstance(raw_fp, dict) else list(raw_fp) if isinstance(raw_fp, list) else []

    # Separate user-authored static routes, declarative heuristic rules, and previously plugin-materialized entries
    user_static_entries: list[dict[str, Any]] = []
    declarative_rules: list[dict[str, Any]] = []

    existing_saved_rules = cfg.get("fallback_heuristic_rules")
    if isinstance(existing_saved_rules, list):
        for r in existing_saved_rules:
            if isinstance(r, dict):
                declarative_rules.append(copy.deepcopy(r))

    for entry in fp_list:
        if not isinstance(entry, dict):
            continue
        if entry.get("_managed_by_heuristic_plugin"):
            continue
        model_val = str(entry.get("model") or "").strip()
        is_rule = bool(
            entry.get("criteria")
            or entry.get("heuristic")
            or entry.get("providers")
            or model_val.startswith(("auto:", "heuristic:", "criteria:"))
        ) and not entry.get("_is_heuristic")
        if is_rule:
            if entry not in declarative_rules:
                declarative_rules.append(copy.deepcopy(entry))
        elif model_val:
            user_static_entries.append(copy.deepcopy(entry))

    if declarative_rules:
        cfg["fallback_heuristic_rules"] = copy.deepcopy(declarative_rules)

    if not heuristics_on:
        # When heuristics are turned off, keep only user static entries (plus any static fallback on rules)
        restored: list[dict[str, Any]] = list(user_static_entries)
        cfg["fallback_providers"] = restored
        if save:
            save_hermes_config(cfg)
        return restored

    # Filter user static entries to those whose providers are available and not excluded
    usable_static: list[dict[str, Any]] = []
    for st in user_static_entries:
        st_prov = normalize_provider_name(st.get("provider") or st.get("model_provider"))
        st_mod = str(st.get("model") or "").strip()
        if not st_prov or not st_mod:
            continue
        if st_prov in excl_provs or st_mod.lower() in excl_models:
            continue
        if is_provider_available(st_prov, config=cfg):
            usable_static.append(st)

    # Resolve declarative heuristic rules (or synthesize a rule from plugin settings if none exist)
    rules_to_resolve = list(declarative_rules)
    if not rules_to_resolve:
        criteria = build_criteria_from_settings(cfg)
        rules_to_resolve = [
            {
                "provider": criteria.provider,
                "providers": list(criteria.providers),
                "criteria": criteria.to_dict(),
            }
        ]

    materialized_heuristics: list[dict[str, Any]] = []
    for rule in rules_to_resolve:
        rule_copy = copy.deepcopy(rule)
        crit_obj = parse_fallback_criteria(rule_copy)
        if excl_provs:
            crit_obj.exclude_providers = tuple(
                dict.fromkeys([*crit_obj.exclude_providers, *excl_provs])
            )
        if excl_models:
            crit_obj.exclude_models = tuple(
                dict.fromkeys([*crit_obj.exclude_models, *excl_models])
            )
        if primary_provider and not crit_obj.current_provider:
            crit_obj.current_provider = primary_provider

        # If rule used default "nous" but "nous" is not authenticated and other providers exist,
        # allow resolving across existing providers when allow_provider_switch is True
        if (
            crit_obj.provider == "nous"
            and not crit_obj.providers
            and crit_obj.allow_provider_switch
            and not is_provider_available("nous", config=cfg)
        ):
            existing_provs = discover_existing_providers(config=cfg)
            if existing_provs:
                crit_obj.provider = "existing"
                if "free" not in str(rule_copy.get("heuristic") or "").lower() and not (
                    isinstance(rule_copy.get("criteria"), dict)
                    and rule_copy["criteria"].get("free_only") is True
                ):
                    crit_obj.free_only = bool(settings.get("free_only", False))

        rule_copy["criteria"] = crit_obj.to_dict()
        rule_copy["provider"] = crit_obj.provider
        if crit_obj.providers:
            rule_copy["providers"] = list(crit_obj.providers)

        resolved_list = resolve_fallback_entry(rule_copy, config=cfg)
        for item in resolved_list:
            # Skip if identical to the active primary model on the same provider
            item_prov = normalize_provider_name(item.get("provider"))
            item_mod = str(item.get("model") or "").strip()
            if (
                primary_provider
                and primary_model
                and item_prov == primary_provider
                and item_mod.lower() == primary_model.lower()
            ):
                continue
            item["_managed_by_heuristic_plugin"] = True
            materialized_heuristics.append(item)

    # Combine usable static routes + materialized heuristic routes, deduplicated
    combined: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in (*usable_static, *materialized_heuristics):
        ident = _entry_identity(entry)
        if ident not in seen:
            seen.add(ident)
            combined.append(entry)

    cfg["fallback_providers"] = combined

    # Sync per-model reasoning_overrides for materialized heuristic models when needed
    agent_cfg = cfg.setdefault("agent", {})
    if isinstance(agent_cfg, dict):
        overrides = agent_cfg.setdefault("reasoning_overrides", {})
        if isinstance(overrides, dict):
            for entry in materialized_heuristics:
                mod_id = str(entry.get("model") or "").strip()
                if not mod_id:
                    continue
                eff = entry.get("reasoning_effort")
                if eff is not None:
                    overrides[mod_id] = eff
                elif agent_cfg.get("reasoning_effort") and mod_id not in overrides:
                    overrides[mod_id] = "default"

    if save:
        save_hermes_config(cfg)

    return combined


def resolve_best_available_model(
    criteria_or_overrides: Optional[dict[str, Any] | str | FallbackCriteria] = None,
    *,
    config: Optional[dict[str, Any]] = None,
    exclude_providers: tuple[str, ...] | list[str] = (),
    exclude_models: tuple[str, ...] | list[str] = (),
) -> list[dict[str, Any]]:
    """Resolve the best available (provider, model) candidates across existing providers without an LLM call."""
    cfg = config if isinstance(config, dict) else load_hermes_config()
    criteria = build_criteria_from_settings(cfg, overrides=criteria_or_overrides)

    if exclude_providers:
        criteria.exclude_providers = tuple(
            dict.fromkeys([*criteria.exclude_providers, *parse_provider_list(exclude_providers)])
        )
    if exclude_models:
        criteria.exclude_models = tuple(
            dict.fromkeys(
                [*criteria.exclude_models, *(str(m).strip() for m in exclude_models if str(m).strip())]
            )
        )

    # If provider is "nous" by default and nous isn't authenticated, search existing providers
    if (
        criteria.provider == "nous"
        and not criteria.providers
        and criteria.allow_provider_switch
        and not is_provider_available("nous", config=cfg)
    ):
        existing = discover_existing_providers(config=cfg)
        if existing:
            criteria.provider = "existing"

    entry = {
        "provider": criteria.provider,
        "providers": list(criteria.providers),
        "criteria": criteria.to_dict(),
    }
    return resolve_fallback_entry(entry, max_candidates=criteria.max_candidates, config=cfg)


def switch_model_with_heuristic(
    criteria_or_overrides: Optional[dict[str, Any] | str | FallbackCriteria] = None,
    *,
    criteria: Optional[dict[str, Any] | str | FallbackCriteria] = None,
    config: Optional[dict[str, Any]] = None,
    save: bool = True,
    persist: Optional[bool] = None,
    exclude_current: bool = False,
    exclude_providers: tuple[str, ...] | list[str] = (),
    exclude_models: tuple[str, ...] | list[str] = (),
    agent: Any = None,
) -> Optional[dict[str, Any]]:
    """Switch the user's active primary model (and provider) using the heuristic across existing providers.

    Works completely without a model call:
    - Updates ``model.provider`` and ``model.default`` in ``config.yaml``
    - Populates ``fallback_providers`` with the next-best heuristic candidates
    - Optionally updates a live ``AIAgent`` instance if provided
    """
    effective_criteria = criteria if criteria is not None else criteria_or_overrides
    effective_save = persist if persist is not None else save

    cfg = config if isinstance(config, dict) else load_hermes_config()
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    old_provider = normalize_provider_name(model_cfg.get("provider")) or str(
        model_cfg.get("provider") or ""
    )
    old_model = str(model_cfg.get("default") or "").strip()

    excl_models = list(exclude_models)
    if exclude_current and old_model:
        excl_models.append(old_model)
        if old_provider:
            excl_models.append(f"{old_provider}/{old_model}")

    resolved = resolve_best_available_model(
        effective_criteria,
        config=cfg,
        exclude_providers=exclude_providers,
        exclude_models=excl_models,
    )
    if not resolved:
        return None

    chosen = resolved[0]
    new_provider = str(chosen["provider"]).strip()
    new_model = str(chosen["model"]).strip()

    if not isinstance(cfg.get("model"), dict):
        cfg["model"] = {}
    cfg["model"]["provider"] = new_provider
    cfg["model"]["default"] = new_model
    cfg["fallback_heuristics"] = True

    # Materialize remaining candidates into fallback_providers
    sync_fallback_chain_to_config(
        cfg,
        save=effective_save,
        exclude_providers=exclude_providers,
        exclude_models=[*excl_models, new_model],
    )

    if agent is not None:
        try:
            agent.model = new_model
            agent.provider = new_provider
            agent.requested_provider = new_provider
            if hasattr(agent, "_fallback_chain"):
                agent._fallback_chain = list(cfg.get("fallback_providers") or [])
                agent._fallback_index = 0
        except Exception as exc:
            logger.debug("Live agent update after heuristic switch skipped: %s", exc)

    return {
        "ok": True,
        "previous_provider": old_provider,
        "previous_model": old_model,
        "provider": new_provider,
        "model": new_model,
        "provider_switched": old_provider != new_provider,
        "criteria_matched": chosen.get("criteria_matched", ""),
        "fallback_chain": cfg.get("fallback_providers", []),
    }


def ensure_available_model_or_fallback(
    config: Optional[dict[str, Any]] = None,
    *,
    save: bool = True,
    failed_provider: Optional[str] = None,
    failed_model: Optional[str] = None,
    agent: Any = None,
) -> dict[str, Any]:
    """Ensure the orchestrator has a valid model and fallback chain from existing providers.

    If the selected primary model/provider is unavailable (or matches ``failed_provider``/``failed_model``)
    and ``auto_switch_primary_when_unavailable`` is enabled, switches the primary model/provider
    using the heuristic. Always ensures ``fallback_providers`` is populated with available routes.
    """
    cfg = config if isinstance(config, dict) else load_hermes_config()
    settings = get_plugin_settings(cfg)
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    primary_provider = normalize_provider_name(model_cfg.get("provider"))
    primary_model = str(model_cfg.get("default") or "").strip()

    excl_provs: list[str] = []
    excl_models: list[str] = []
    if failed_provider and failed_model:
        excl_models.append(failed_model)
        excl_models.append(f"{normalize_provider_name(failed_provider)}/{failed_model}")
    elif failed_provider:
        excl_provs.append(normalize_provider_name(failed_provider))

    primary_unavailable = (
        not primary_provider
        or not primary_model
        or not is_provider_available(primary_provider, config=cfg)
        or (
            failed_provider
            and normalize_provider_name(failed_provider) == primary_provider
            and (not failed_model or failed_model.lower() == primary_model.lower())
        )
    )

    if primary_unavailable and settings.get("auto_switch_primary_when_unavailable", True):
        switched = switch_model_with_heuristic(
            config=cfg,
            save=save,
            exclude_current=bool(primary_model),
            exclude_providers=excl_provs,
            exclude_models=excl_models,
            agent=agent,
        )
        if switched is not None:
            return {
                "status": "switched_primary",
                "primary_was_unavailable": True,
                **switched,
            }

    chain = sync_fallback_chain_to_config(
        cfg,
        save=save,
        exclude_providers=excl_provs,
        exclude_models=excl_models,
    )
    if agent is not None and hasattr(agent, "_fallback_chain"):
        try:
            agent._fallback_chain = list(chain)
        except Exception:
            pass

    return {
        "status": "synced_fallbacks",
        "primary_was_unavailable": primary_unavailable,
        "provider": primary_provider,
        "model": primary_model,
        "fallback_chain": chain,
    }
