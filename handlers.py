"""Tool handlers, slash command handlers, hooks, and middleware for Model Fallback Heuristic."""

from __future__ import annotations

import json
import logging
import shlex
from typing import Any, Optional

try:
    from .config_sync import (
        build_criteria_from_settings,
        ensure_available_model_or_fallback,
        get_fallback_chain,
        get_plugin_settings,
        get_stored_fallback_rules,
        load_hermes_config,
        save_hermes_config,
        switch_model_with_heuristic,
        sync_fallback_chain_to_config,
        update_plugin_settings,
    )
    from .heuristics import (
        _resolve_target_providers,
        discover_existing_providers,
        filter_and_rank_candidates,
        get_candidate_models,
        get_provider_status_list,
        normalize_provider_name,
        parse_provider_list,
        resolve_fallback_entry,
    )
    from .onboarding import (
        check_free_model_providers,
        format_first_launch_guide,
        is_first_launch,
        mark_first_launch_done,
    )
except ImportError:
    from config_sync import (  # type: ignore
        build_criteria_from_settings,
        ensure_available_model_or_fallback,
        get_fallback_chain,
        get_plugin_settings,
        get_stored_fallback_rules,
        load_hermes_config,
        save_hermes_config,
        switch_model_with_heuristic,
        sync_fallback_chain_to_config,
        update_plugin_settings,
    )
    from heuristics import (  # type: ignore
        _resolve_target_providers,
        discover_existing_providers,
        filter_and_rank_candidates,
        get_candidate_models,
        get_provider_status_list,
        normalize_provider_name,
        parse_provider_list,
        resolve_fallback_entry,
    )
    from onboarding import (  # type: ignore
        check_free_model_providers,
        format_first_launch_guide,
        is_first_launch,
        mark_first_launch_done,
    )

logger = logging.getLogger(__name__)

# Session-scoped tracking of models/providers observed failing during the session
_SESSION_UNAVAILABLE_MODELS: set[str] = set()
_SESSION_UNAVAILABLE_PROVIDERS: set[str] = set()
_FIRST_LAUNCH_BANNER_SHOWN: bool = False


def reset_session_unavailability() -> None:
    """Clear tracked unavailable models and providers for the session."""
    _SESSION_UNAVAILABLE_MODELS.clear()
    _SESSION_UNAVAILABLE_PROVIDERS.clear()


def record_unavailable_route(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    provider_wide: bool = False,
) -> None:
    """Record a failed model or provider route so subsequent heuristic resolutions skip it."""
    prov_norm = normalize_provider_name(provider)
    mod_clean = str(model or "").strip()
    if mod_clean:
        _SESSION_UNAVAILABLE_MODELS.add(mod_clean.lower())
        if prov_norm:
            _SESSION_UNAVAILABLE_MODELS.add(f"{prov_norm}/{mod_clean}".lower())
    if provider_wide and prov_norm:
        _SESSION_UNAVAILABLE_PROVIDERS.add(prov_norm)


# ─── Tool Handlers ──────────────────────────────────────────────────────────


def handle_resolve(args: dict[str, Any], **kwargs: Any) -> str:
    """Handler for ``fallback_heuristic_resolve`` tool."""
    try:
        cfg = load_hermes_config()
        overrides: dict[str, Any] = {}
        for k in (
            "heuristic",
            "sort_by",
            "provider",
            "providers",
            "exclude_providers",
            "exclude_models",
            "allow_provider_switch",
            "prefer_current_provider",
            "min_context",
            "max_context",
            "min_params",
            "max_params",
            "free_only",
            "require_tools",
            "flash_only",
            "vendor",
            "name_filter",
            "reasoning_effort",
            "max_candidates",
        ):
            if args.get(k) is not None:
                overrides[k] = args[k]

        # If caller passed shorthand heuristic string and no explicit sort_by/provider, parse it first
        if "heuristic" in overrides and len(overrides) == 1:
            criteria = build_criteria_from_settings(cfg, overrides=str(overrides["heuristic"]))
        else:
            criteria = build_criteria_from_settings(cfg, overrides=overrides)

        if _SESSION_UNAVAILABLE_MODELS:
            criteria.exclude_models = tuple(
                dict.fromkeys([*criteria.exclude_models, *_SESSION_UNAVAILABLE_MODELS])
            )
        if _SESSION_UNAVAILABLE_PROVIDERS:
            criteria.exclude_providers = tuple(
                dict.fromkeys([*criteria.exclude_providers, *_SESSION_UNAVAILABLE_PROVIDERS])
            )

        limit = int(args.get("max_candidates") or criteria.max_candidates or 5)
        criteria.max_candidates = limit

        existing_providers = discover_existing_providers(config=cfg)
        entry = {
            "provider": criteria.provider,
            "providers": list(criteria.providers),
            "criteria": criteria.to_dict(),
        }

        target_providers = _resolve_target_providers(
            criteria, explicit_provider=criteria.provider, config=cfg
        )
        candidates = get_candidate_models(
            provider="existing" if len(target_providers) > 1 else (target_providers[0] if target_providers else "nous"),
            config=cfg,
            providers=target_providers if len(target_providers) > 1 else None,
        )
        ranked = filter_and_rank_candidates(candidates, criteria)
        resolved_entries = resolve_fallback_entry(entry, max_candidates=limit, config=cfg)

        candidate_summaries = [
            {
                "rank": idx + 1,
                "provider": c.provider,
                "model": c.id,
                "name": c.name,
                "param_size_b": c.param_size_b,
                "context_window": c.context_window,
                "is_free": c.is_free,
                "supports_tools": c.supports_tools,
                "is_flash": c.is_flash,
                "vendor": c.vendor,
                "release_date": c.release_date,
                "source": c.source,
            }
            for idx, c in enumerate(ranked[:limit])
        ]

        return json.dumps(
            {
                "ok": True,
                "criteria": criteria.to_dict(),
                "criteria_summary": criteria.describe(),
                "target_providers": target_providers,
                "existing_providers": existing_providers,
                "total_matching": len(ranked),
                "candidates": candidate_summaries,
                "resolved_fallback_entries": resolved_entries,
            },
            indent=2,
        )
    except Exception as exc:
        logger.warning("fallback_heuristic_resolve failed: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


def handle_switch(args: dict[str, Any], **kwargs: Any) -> str:
    """Handler for ``fallback_heuristic_switch`` tool."""
    try:
        cfg = load_hermes_config()
        exclude_current = bool(args.get("exclude_current", False))
        excl_provs = list(parse_provider_list(args.get("exclude_providers"))) + list(
            _SESSION_UNAVAILABLE_PROVIDERS
        )
        excl_models = [
            *(str(m).strip() for m in (args.get("exclude_models") or []) if str(m).strip()),
            *_SESSION_UNAVAILABLE_MODELS,
        ]

        overrides: dict[str, Any] = {}
        for k in (
            "heuristic",
            "sort_by",
            "provider",
            "providers",
            "allow_provider_switch",
            "prefer_current_provider",
            "min_context",
            "max_context",
            "min_params",
            "max_params",
            "free_only",
            "require_tools",
            "flash_only",
            "vendor",
            "name_filter",
            "reasoning_effort",
        ):
            if args.get(k) is not None:
                overrides[k] = args[k]

        if "provider" not in overrides and "providers" not in overrides:
            settings = get_plugin_settings(cfg)
            if settings.get("provider") in (None, "", "nous"):
                overrides["provider"] = "existing"

        parent_agent = kwargs.get("parent_agent")
        result = switch_model_with_heuristic(
            overrides if overrides else None,
            config=cfg,
            save=True,
            exclude_current=exclude_current,
            exclude_providers=excl_provs,
            exclude_models=excl_models,
            agent=parent_agent,
        )
        if result is None:
            return json.dumps(
                {
                    "ok": False,
                    "error": "No matching candidate models found across the allowed providers.",
                    "existing_providers": discover_existing_providers(config=cfg),
                },
                indent=2,
            )

        return json.dumps({"ok": True, **result}, indent=2)
    except Exception as exc:
        logger.warning("fallback_heuristic_switch failed: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


def handle_configure(args: dict[str, Any], **kwargs: Any) -> str:
    """Handler for ``fallback_heuristic_configure`` tool."""
    try:
        action = str(args.get("action") or "status").strip().lower()
        cfg = load_hermes_config()

        if action == "on":
            settings = update_plugin_settings({"enabled": True}, config=cfg, save=True)
        elif action == "off":
            settings = update_plugin_settings({"enabled": False}, config=cfg, save=True)
        elif action == "clear_rules":
            cfg["fallback_heuristic_rules"] = []
            cfg["fallback_providers"] = [
                e
                for e in (cfg.get("fallback_providers") or [])
                if isinstance(e, dict)
                and not e.get("_managed_by_heuristic_plugin")
                and not e.get("_is_heuristic")
                and not e.get("heuristic")
                and not e.get("criteria")
            ]
            save_hermes_config(cfg)
            settings = get_plugin_settings(cfg)
        elif action == "add_rule":
            rule: dict[str, Any] = {}
            for k in (
                "provider",
                "providers",
                "exclude_providers",
                "allow_provider_switch",
                "prefer_current_provider",
                "sort_by",
                "free_only",
                "require_tools",
                "flash_only",
                "min_context",
                "max_context",
                "min_params",
                "max_params",
                "vendor",
                "name_filter",
                "reasoning_effort",
                "max_candidates",
            ):
                if args.get(k) is not None:
                    rule[k] = args[k]
            if not rule.get("provider") and not rule.get("providers"):
                rule["provider"] = "existing"
            if "sort_by" in rule and "heuristic" not in rule:
                rule["heuristic"] = rule["sort_by"]
            rules_list = cfg.setdefault("fallback_heuristic_rules", [])
            if isinstance(rules_list, list):
                rules_list.append(rule)
            cfg["fallback_heuristics"] = True
            sync_fallback_chain_to_config(cfg, save=True)
            settings = get_plugin_settings(cfg)
        elif action == "sync":
            sync_fallback_chain_to_config(
                cfg,
                save=True,
                exclude_providers=list(_SESSION_UNAVAILABLE_PROVIDERS),
                exclude_models=list(_SESSION_UNAVAILABLE_MODELS),
            )
            settings = get_plugin_settings(cfg)
        elif action in ("setup", "onboarding", "first_launch"):
            selected = parse_provider_list(args.get("providers") or args.get("provider"))
            settings = mark_first_launch_done(
                selected_providers=list(selected) if selected else None,
                config=cfg,
                save=True,
            )
            free_report = check_free_model_providers(config=cfg, allow_network=True)
            model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
            return json.dumps(
                {
                    "ok": True,
                    "action": action,
                    "first_launch_completed": True,
                    "selected_providers": list(selected),
                    "primary_provider": model_cfg.get("provider"),
                    "primary_model": model_cfg.get("default"),
                    "existing_providers": discover_existing_providers(config=cfg),
                    "free_model_providers": free_report,
                    "settings": settings,
                    "effective_fallback_chain": cfg.get("fallback_providers") or [],
                },
                indent=2,
            )
        elif action == "set" or any(
            k in args
            for k in (
                "enabled",
                "sort_by",
                "provider",
                "providers",
                "exclude_providers",
                "allow_provider_switch",
                "prefer_current_provider",
                "free_only",
                "require_tools",
                "flash_only",
                "min_context",
                "max_context",
                "min_params",
                "max_params",
                "vendor",
                "name_filter",
                "reasoning_effort",
                "max_candidates",
                "auto_switch_primary_when_unavailable",
            )
        ):
            updates = {k: v for k, v in args.items() if k != "action" and v is not None}
            settings = update_plugin_settings(updates, config=cfg, save=True)
        else:
            settings = get_plugin_settings(cfg)

        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        return json.dumps(
            {
                "ok": True,
                "action": action,
                "primary_provider": model_cfg.get("provider"),
                "primary_model": model_cfg.get("default"),
                "existing_providers": discover_existing_providers(config=cfg),
                "settings": settings,
                "stored_rules": get_stored_fallback_rules(cfg),
                "effective_fallback_chain": cfg.get("fallback_providers") or [],
            },
            indent=2,
        )
    except Exception as exc:
        logger.warning("fallback_heuristic_configure failed: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


def handle_providers(args: dict[str, Any], **kwargs: Any) -> str:
    """Handler for ``fallback_heuristic_providers`` tool."""
    try:
        cfg = load_hermes_config()
        include_counts = bool(args.get("include_catalog_counts", True))
        only_auth = bool(args.get("only_authenticated", False))
        check_free = bool(args.get("check_free_providers", True))
        providers_info = get_provider_status_list(
            config=cfg, include_catalog_counts=include_counts
        )
        if only_auth:
            providers_info = [p for p in providers_info if p.get("authenticated")]
        payload: dict[str, Any] = {
            "ok": True,
            "is_first_launch": is_first_launch(cfg),
            "existing_providers": discover_existing_providers(config=cfg),
            "providers": providers_info,
        }
        if check_free:
            payload["free_model_providers"] = check_free_model_providers(
                config=cfg, allow_network=True
            )
        return json.dumps(payload, indent=2)
    except Exception as exc:
        logger.warning("fallback_heuristic_providers failed: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


# ─── Lifecycle Hooks & Middleware ───────────────────────────────────────────


def on_session_start_hook(**kwargs: Any) -> Optional[str]:
    """Hook: ``on_session_start`` — ensures primary model and fallback_providers are synced before any LLM call,
    and on first launch runs the free-model provider check to guide the user."""
    global _FIRST_LAUNCH_BANNER_SHOWN
    try:
        reset_session_unavailability()
        cfg = load_hermes_config()
        settings = get_plugin_settings(cfg)
        if not settings.get("enabled", True) or not settings.get("auto_sync_config", True):
            return None
        ensure_available_model_or_fallback(config=cfg, save=True)
        if is_first_launch(cfg) and not _FIRST_LAUNCH_BANNER_SHOWN:
            _FIRST_LAUNCH_BANNER_SHOWN = True
            report = check_free_model_providers(config=cfg, allow_network=False)
            guide = format_first_launch_guide(report, config=cfg)
            logger.info("\n%s", guide)
            return guide
    except Exception as exc:
        logger.debug("on_session_start_hook skipped: %s", exc)
    return None


def pre_llm_call_hook(**kwargs: Any) -> None:
    """Hook: ``pre_llm_call`` — ensures fallback_providers are materialized prior to the turn's LLM call."""
    try:
        cfg = load_hermes_config()
        settings = get_plugin_settings(cfg)
        if not settings.get("enabled", True) or not settings.get("auto_sync_config", True):
            return
        fp = cfg.get("fallback_providers")
        if not fp:
            sync_fallback_chain_to_config(
                cfg,
                save=True,
                exclude_providers=list(_SESSION_UNAVAILABLE_PROVIDERS),
                exclude_models=list(_SESSION_UNAVAILABLE_MODELS),
            )
    except Exception as exc:
        logger.debug("pre_llm_call_hook skipped: %s", exc)


def api_request_error_hook(**kwargs: Any) -> None:
    """Hook: ``api_request_error`` — records failed route and refreshes heuristic fallback_providers."""
    try:
        reason = str(kwargs.get("reason") or "").lower()
        status_code = kwargs.get("status_code")
        api_kwargs = kwargs.get("api_kwargs")
        failed_model = ""
        if isinstance(api_kwargs, dict):
            failed_model = str(api_kwargs.get("model") or "").strip()

        cfg = load_hermes_config()
        settings = get_plugin_settings(cfg)
        if not settings.get("enabled", True):
            return

        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        active_provider = normalize_provider_name(
            kwargs.get("provider") or model_cfg.get("provider")
        )
        if not failed_model:
            failed_model = str(kwargs.get("model") or model_cfg.get("default") or "").strip()

        provider_wide_failure = reason in ("auth", "auth_permanent", "billing") or status_code in (
            401,
            402,
            403,
        )
        record_unavailable_route(
            provider=active_provider,
            model=failed_model,
            provider_wide=provider_wide_failure,
        )

        sync_fallback_chain_to_config(
            cfg,
            save=True,
            exclude_providers=list(_SESSION_UNAVAILABLE_PROVIDERS),
            exclude_models=list(_SESSION_UNAVAILABLE_MODELS),
        )
    except Exception as exc:
        logger.debug("api_request_error_hook skipped: %s", exc)


def llm_request_middleware(request: dict[str, Any], context: Any) -> dict[str, Any]:
    """Middleware: ``llm_request`` — normalizes reasoning config for heuristic fallback models."""
    if not isinstance(request, dict):
        return request
    try:
        model = str(getattr(context, "model", None) or request.get("model") or "").strip()
        if not model:
            return request
        cfg = load_hermes_config()
        fp_list = cfg.get("fallback_providers")
        if not isinstance(fp_list, list):
            return request

        matched_entry = None
        for entry in fp_list:
            if (
                isinstance(entry, dict)
                and str(entry.get("model") or "").strip().lower() == model.lower()
                and (entry.get("_is_heuristic") or entry.get("criteria_matched"))
            ):
                matched_entry = entry
                break

        if matched_entry is None:
            return request

        eff = matched_entry.get("reasoning_effort")
        if eff in ("none", "off", "false", False):
            updated = dict(request)
            updated.pop("reasoning_effort", None)
            extra_body = updated.get("extra_body")
            if isinstance(extra_body, dict) and "reasoning" in extra_body:
                eb = dict(extra_body)
                eb["reasoning"] = {"enabled": False}
                updated["extra_body"] = eb
            return updated
    except Exception as exc:
        logger.debug("llm_request_middleware skipped: %s", exc)
    return request


def llm_execution_middleware(
    request: dict[str, Any], call_next: Any, context: Any
) -> Any:
    """Middleware: ``llm_execution`` — catches unavailable model/provider failures and refreshes heuristic fallbacks."""
    try:
        return call_next(request)
    except Exception as exc:
        try:
            provider = str(getattr(context, "provider", "") or "").strip()
            model = str(
                getattr(context, "model", "")
                or (request.get("model") if isinstance(request, dict) else "")
                or ""
            ).strip()
            status_code = getattr(exc, "status_code", None)
            err_text = str(exc).lower()
            provider_wide = status_code in (401, 402, 403) or any(
                tok in err_text
                for tok in ("invalid api key", "insufficient_quota", "credit balance", "unauthorized")
            )
            record_unavailable_route(
                provider=provider, model=model, provider_wide=provider_wide
            )
            cfg = load_hermes_config()
            settings = get_plugin_settings(cfg)
            if settings.get("enabled", True):
                sync_fallback_chain_to_config(
                    cfg,
                    save=True,
                    exclude_providers=list(_SESSION_UNAVAILABLE_PROVIDERS),
                    exclude_models=list(_SESSION_UNAVAILABLE_MODELS),
                )
        except Exception as sync_err:
            logger.debug("llm_execution_middleware fallback sync failed: %s", sync_err)
        raise


# ─── In-Session Slash Command Handler (Zero LLM Calls) ──────────────────────


def _parse_key_value_tokens(tokens: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Split 'key=value' or 'key:value' tokens from positional arguments."""
    kv: dict[str, Any] = {}
    positional: list[str] = []
    known_keys = {
        "sort_by",
        "heuristic",
        "strategy",
        "provider",
        "model_provider",
        "providers",
        "exclude_providers",
        "exclude_provider",
        "allow_provider_switch",
        "switch_provider",
        "prefer_current_provider",
        "free",
        "free_only",
        "tools",
        "require_tools",
        "flash",
        "flash_only",
        "min_context",
        "min_ctx",
        "max_context",
        "max_ctx",
        "min_params",
        "max_params",
        "vendor",
        "lab",
        "filter",
        "name_filter",
        "pattern",
        "reasoning",
        "reasoning_effort",
        "thinking",
        "max_candidates",
        "limit",
        "enabled",
        "auto_sync_config",
        "auto_switch_primary_when_unavailable",
    }
    for tok in tokens:
        sep = "=" if "=" in tok else ":" if ":" in tok else None
        if sep:
            k, v = tok.split(sep, 1)
            k_clean = k.strip().lower().lstrip("-")
            if k_clean in known_keys:
                kv[k_clean] = v.strip()
                continue
        positional.append(tok)
    return kv, positional


def handle_slash_command(raw_args: str = "") -> str:
    """Handle ``/fallback`` and ``/fallback-heuristic`` in-session slash commands without an LLM call."""
    text = (raw_args or "").strip()
    try:
        tokens = shlex.split(text) if text else []
    except ValueError:
        tokens = text.split()

    subcmd = tokens[0].lower() if tokens else "status"
    rest = tokens[1:] if len(tokens) > 1 else []

    if subcmd in ("help", "-h", "--help"):
        return (
            "Model Fallback Heuristic Commands (no LLM call required):\n"
            "  /fallback status                       Show current settings, providers, and fallback chain\n"
            "  /fallback setup [providers...]         First-launch guide & live check of providers offering free models\n"
            "  /fallback free-providers               Run live check of all providers offering free ($0) models\n"
            "  /fallback on | off                     Enable or disable heuristic fallback routing\n"
            "  /fallback resolve [criteria...]        Preview top matching models across providers\n"
            "  /fallback switch [criteria...]         Switch active model/provider using heuristics\n"
            "  /fallback set <key=value ...>          Update heuristic settings (provider, sort_by, min_context, min_params, etc.)\n"
            "  /fallback providers                    List authenticated ('existing') providers and model counts\n"
            "  /fallback sync                         Re-materialize fallback_providers in config.yaml\n"
            "  /fallback clear                        Clear heuristic rules and materialized fallbacks\n"
            "\nExamples:\n"
            "  /fallback setup openrouter,gemini,nvidia\n"
            "  /fallback set provider=existing sort_by=largest_parameter_count min_context=128k\n"
            "  /fallback resolve provider=anthropic,openrouter,gemini sort_by=greatest_context\n"
            "  /fallback switch provider=existing sort_by=latest_flash"
        )

    cfg = load_hermes_config()

    if subcmd in ("free-providers", "free", "check-free"):
        report = check_free_model_providers(config=cfg, allow_network=True)
        return format_first_launch_guide(report, config=cfg)

    if subcmd in ("setup", "onboarding", "first-launch", "init"):
        report = check_free_model_providers(config=cfg, allow_network=True)
        if not rest:
            return format_first_launch_guide(report, config=cfg)
        joined = " ".join(rest).strip().lower()
        if joined in ("done", "skip", "dismiss"):
            mark_first_launch_done(config=cfg, save=True)
            return "✓ First-launch provider check marked complete. Run `/fallback setup` anytime to revisit."

        selected: list[str] = []
        for raw_tok in ",".join(rest).split(","):
            tok = raw_tok.strip()
            if not tok:
                continue
            if tok.isdigit():
                idx = int(tok) - 1
                if 0 <= idx < len(report):
                    selected.append(report[idx]["provider"])
            else:
                norm = normalize_provider_name(tok)
                if norm:
                    selected.append(norm)
        selected = list(dict.fromkeys(selected))
        if not selected:
            return format_first_launch_guide(report, config=cfg)

        settings = mark_first_launch_done(selected_providers=selected, config=cfg, save=True)
        chain = cfg.get("fallback_providers") or []
        out_lines = [
            f"✓ Saved preferred fallback providers: {', '.join(settings.get('providers', []))}",
            f"  Materialized {len(chain)} fallback route(s) in config.yaml.",
        ]
        missing = [
            p
            for p in report
            if p["provider"] in selected and not p.get("authenticated") and p.get("env_vars")
        ]
        if missing:
            out_lines.append("")
            out_lines.append("Next step — create & set free API keys for your unconfigured providers:")
            for mp in missing:
                env_name = mp["env_vars"][0]
                out_lines.append(
                    f"  • {mp['display_name']} (`{mp['provider']}`): {mp['signup_url']}  →  export {env_name}=\"...\""
                )
        return "\n".join(out_lines)

    if subcmd == "on":
        update_plugin_settings({"enabled": True}, config=cfg, save=True)
        chain = cfg.get("fallback_providers") or []
        return f"✓ Model fallback heuristics ENABLED ({len(chain)} fallback route(s) materialized in config.yaml)."

    if subcmd == "off":
        update_plugin_settings({"enabled": False}, config=cfg, save=True)
        return "✓ Model fallback heuristics DISABLED."

    if subcmd == "providers":
        infos = get_provider_status_list(config=cfg, include_catalog_counts=True)
        existing = [p for p in infos if p["authenticated"]]
        lines = [
            f"Existing (Authenticated/Configured) Providers ({len(existing)}):",
        ]
        if existing:
            for p in existing:
                prim = " [PRIMARY]" if p["is_primary"] else ""
                lines.append(
                    f"  • {p['provider']}{prim}: {p['candidate_count']} model(s) ({p['free_candidate_count']} free)"
                )
        else:
            lines.append("  (none detected — configure a provider or set an API key)")
        return "\n".join(lines)

    if subcmd == "set":
        kv, pos = _parse_key_value_tokens(rest)
        if pos and "sort_by" not in kv and "heuristic" not in kv:
            kv["sort_by"] = pos[0]
        if not kv:
            return "Usage: /fallback set <key=value ...> (e.g. /fallback set provider=existing sort_by=greatest_context min_context=128k)"
        settings = update_plugin_settings(kv, config=cfg, save=True)
        chain = cfg.get("fallback_providers") or []
        return (
            f"✓ Updated heuristic settings: sort_by={settings['sort_by']}, "
            f"provider={settings['provider']}, providers={settings['providers']}, "
            f"allow_provider_switch={settings['allow_provider_switch']}, "
            f"min_context={settings['min_context']}, min_params={settings['min_params']}, "
            f"free_only={settings['free_only']}\n"
            f"  Materialized {len(chain)} fallback route(s) in config.yaml."
        )

    if subcmd == "resolve":
        kv, pos = _parse_key_value_tokens(rest)
        if pos and "heuristic" not in kv and "sort_by" not in kv:
            kv["heuristic"] = " ".join(pos)
        payload = json.loads(handle_resolve(kv))
        if not payload.get("ok"):
            return f"✗ Failed to resolve candidates: {payload.get('error')}"
        cands = payload.get("candidates") or []
        lines = [
            f"Heuristic: {payload.get('criteria_summary')} (target providers: {', '.join(payload.get('target_providers') or [])})",
            f"Matched {payload.get('total_matching', 0)} candidate(s):",
        ]
        for c in cands:
            params_str = f"{c['param_size_b']:g}B" if c.get("param_size_b") else "?B"
            ctx_str = f"{c['context_window']:,}" if c.get("context_window") else "?"
            tier = "free" if c.get("is_free") else "paid"
            lines.append(
                f"  #{c['rank']}  {c['provider']}/{c['model']}  ({params_str}, ctx={ctx_str}, {tier})"
            )
        if not cands:
            lines.append("  (no matching models found)")
        return "\n".join(lines)

    if subcmd == "switch":
        kv, pos = _parse_key_value_tokens(rest)
        if pos and "heuristic" not in kv and "sort_by" not in kv:
            kv["heuristic"] = " ".join(pos)
        payload = json.loads(handle_switch(kv))
        if not payload.get("ok"):
            return f"✗ Could not switch model: {payload.get('error')}"
        return (
            f"✓ Switched active model to {payload['provider']}/{payload['model']} "
            f"(was {payload.get('previous_provider') or '?'}/{payload.get('previous_model') or '?'}) "
            f"[{payload.get('criteria_matched')}]."
        )

    if subcmd == "sync":
        chain = sync_fallback_chain_to_config(
            cfg,
            save=True,
            exclude_providers=list(_SESSION_UNAVAILABLE_PROVIDERS),
            exclude_models=list(_SESSION_UNAVAILABLE_MODELS),
        )
        lines = [f"✓ Synced {len(chain)} fallback route(s) to config.yaml:"]
        for idx, item in enumerate(chain, 1):
            lines.append(f"  {idx}. {item.get('provider')}/{item.get('model')}")
        return "\n".join(lines)

    if subcmd == "clear":
        handle_configure({"action": "clear_rules"})
        return "✓ Cleared heuristic fallback rules and materialized entries."

    # Default: status
    settings = get_plugin_settings(cfg)
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    existing = discover_existing_providers(config=cfg)
    chain = cfg.get("fallback_providers") or []
    lines = [
        f"Model Fallback Heuristic Status: {'ENABLED' if settings.get('enabled') else 'DISABLED'}",
        f"  Primary Route:       {model_cfg.get('provider') or '(unset)'} / {model_cfg.get('default') or '(unset)'}",
        f"  Existing Providers:  {', '.join(existing) if existing else '(none detected)'}",
        f"  Heuristic Settings:  sort_by={settings.get('sort_by')}, provider={settings.get('provider')}, "
        f"providers={settings.get('providers') or 'auto'}, switch_provider={settings.get('allow_provider_switch')}, "
        f"free_only={settings.get('free_only')}, min_ctx={settings.get('min_context')}, min_params={settings.get('min_params')}",
        f"  Materialized Chain ({len(chain)}):",
    ]
    if chain:
        for idx, entry in enumerate(chain, 1):
            if isinstance(entry, dict):
                tag = f" [{entry.get('criteria_matched')}]" if entry.get("criteria_matched") else ""
                lines.append(f"    {idx}. {entry.get('provider')}/{entry.get('model')}{tag}")
    else:
        lines.append("    (empty — run `/fallback sync` or `/fallback on` to materialize)")

    if is_first_launch(cfg):
        report = check_free_model_providers(config=cfg, allow_network=True)
        lines.append("")
        lines.append(format_first_launch_guide(report, config=cfg))

    return "\n".join(lines)
