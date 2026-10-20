"""CLI subcommand (`hermes fallback-heuristic` / `hermes fallback`) for the
Hermes Model Fallback Heuristic plugin.

All subcommands run locally and deterministically without requiring any LLM call:
  - `list` / `status`: Display primary model, heuristic settings, available providers, and fallback chain.
  - `on` / `off` / `heuristics [on|off]`: Enable or disable heuristic fallback resolution.
  - `set <key=value ...>`: Configure heuristic parameters (provider, min_params, min_context, sort_by, etc.).
  - `resolve [flags]`: Preview ranked fallback candidates across any or all providers.
  - `switch [flags]`: Immediately switch the active model & provider to the best available heuristic match.
  - `providers [--all]`: Show all model providers and their authentication/availability status.
  - `add [flags]`: Add a static or heuristic fallback entry (interactive picker or flags).
  - `remove [index]`: Remove an entry from the fallback chain.
  - `clear [--yes]`: Clear all entries from the fallback chain.
  - `sync`: Sync plugin heuristic settings into `config.yaml`'s `fallback_providers`.
"""

from __future__ import annotations

import argparse
import copy
import json
from typing import Any, Dict, List, Optional

try:
    from .config_sync import (
        build_criteria_from_settings,
        get_fallback_chain,
        get_plugin_settings,
        get_stored_fallback_rules,
        is_fallback_heuristics_enabled,
        load_hermes_config,
        resolve_best_available_model,
        save_hermes_config,
        switch_model_with_heuristic,
        sync_fallback_chain_to_config,
        update_plugin_settings,
    )
    from .heuristics import (
        FallbackCriteria,
        discover_existing_providers,
        get_provider_status_list,
        normalize_provider_name,
        parse_context_length,
        parse_fallback_criteria,
        parse_param_size,
        parse_provider_list,
        resolve_fallback_candidates,
    )
    from .onboarding import (
        check_free_model_providers,
        format_first_launch_guide,
        is_first_launch,
        mark_first_launch_done,
        run_interactive_first_launch_wizard,
    )
except ImportError:
    from config_sync import (  # type: ignore
        build_criteria_from_settings,
        get_fallback_chain,
        get_plugin_settings,
        get_stored_fallback_rules,
        is_fallback_heuristics_enabled,
        load_hermes_config,
        resolve_best_available_model,
        save_hermes_config,
        switch_model_with_heuristic,
        sync_fallback_chain_to_config,
        update_plugin_settings,
    )
    from heuristics import (  # type: ignore
        FallbackCriteria,
        discover_existing_providers,
        get_provider_status_list,
        normalize_provider_name,
        parse_context_length,
        parse_fallback_criteria,
        parse_param_size,
        parse_provider_list,
        resolve_fallback_candidates,
    )
    from onboarding import (  # type: ignore
        check_free_model_providers,
        format_first_launch_guide,
        is_first_launch,
        mark_first_launch_done,
        run_interactive_first_launch_wizard,
    )


# Declarative fallback chain (stores raw rules, preserves inactive heuristics); always fresh copy.
_read_chain = get_stored_fallback_rules

_MISSING_ACTIVE_PROVIDER = object()


def _identity(entry: Dict[str, Any]):
    """BackendIdentity for a ``{provider, model, base_url?}`` entry."""
    try:
        from agent.backend_identity import BackendIdentity

        return BackendIdentity.build(
            provider=entry.get("provider"),
            model=entry.get("model"),
            base_url=entry.get("base_url"),
        )
    except Exception:
        return (
            normalize_provider_name(str(entry.get("provider") or "")),
            str(entry.get("model") or "").strip().lower(),
            str(entry.get("base_url") or "").strip().rstrip("/").lower(),
        )


def _same_deployment(a: Any, b: Any) -> bool:
    try:
        from agent.backend_identity import same_deployment

        return same_deployment(a, b)
    except Exception:
        return a == b


def _write_chain(config: Dict[str, Any], chain: List[Dict[str, Any]]) -> None:
    """Write entries without _from_fallback_model to fallback_providers.
    Keep remaining entries with _from_fallback_model in fallback_model (or pop if none remain)."""
    fp_entries: list[dict[str, Any]] = []
    fm_entries: list[dict[str, Any]] = []
    for entry in chain:
        e = copy.deepcopy(entry)
        if e.pop("_from_fallback_model", False):
            fm_entries.append(e)
        else:
            fp_entries.append(e)

    config["fallback_providers"] = fp_entries
    if fm_entries:
        config["fallback_model"] = fm_entries
    else:
        config.pop("fallback_model", None)


def _format_entry(entry: Dict[str, Any]) -> str:
    """One-line human-readable rendering of a fallback entry."""
    base = entry.get("base_url")
    base_str = f"  [{base}]" if base else ""
    criteria_str = ""
    if entry.get("criteria_matched"):
        rank = f" #{entry.get('criteria_rank')}" if entry.get("criteria_rank") else ""
        criteria_str = f"  [heuristic: {entry['criteria_matched']}{rank}]"
    elif (
        entry.get("criteria")
        or entry.get("heuristic")
        or str(entry.get("model", "")).startswith(("auto:", "heuristic:", "criteria:"))
    ):
        try:
            crit = parse_fallback_criteria(entry)
            desc = crit.describe()
        except (ValueError, TypeError, KeyError):
            desc = "heuristic rule"
        criteria_str = f"  [heuristic: {desc}]"
    reasoning_str = ""
    if entry.get("reasoning_effort") and entry.get("reasoning_effort") != "default":
        reasoning_str = f"  [thinking: {entry['reasoning_effort']}]"
    model_name = entry.get("model")
    if not model_name or str(model_name).startswith(("auto:", "heuristic:", "criteria:")):
        model_name = "auto"
    return f"{model_name}  (via {entry.get('provider', '?')}){criteria_str}{reasoning_str}{base_str}"


def _extract_fallback_from_model_cfg(model_cfg: Any) -> Optional[Dict[str, Any]]:
    """Pull the ``{provider, model, base_url?, api_mode?}`` dict from a ``config["model"]`` snapshot."""
    if not isinstance(model_cfg, dict):
        return None
    provider = (model_cfg.get("provider") or "").strip()
    model = (model_cfg.get("default") or model_cfg.get("model") or "").strip()
    if not provider or not model:
        return None
    entry: Dict[str, Any] = {"provider": provider, "model": model}
    entry.update(
        {
            key: value
            for key in ("base_url", "api_mode")
            if (value := (model_cfg.get(key) or "").strip())
        }
    )
    return entry


def _snapshot_auth_active_provider() -> Any:
    """Return the current ``active_provider`` in auth.json."""
    from hermes_cli.auth import _auth_store_lock, _load_auth_store

    with _auth_store_lock():
        store = _load_auth_store()
        return store.get("active_provider", _MISSING_ACTIVE_PROVIDER)


def _restore_auth_active_provider(value: Any) -> None:
    """Write back a previously snapshotted ``active_provider`` value."""
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store

    with _auth_store_lock():
        store = _load_auth_store()
        if value is _MISSING_ACTIVE_PROVIDER:
            store.pop("active_provider", None)
        else:
            store["active_provider"] = value
        _save_auth_store(store)


def _restore_model_cfg(model_before: Any) -> None:
    """Restore ``config["model"]`` to a previously-captured snapshot."""
    cfg = load_hermes_config()
    cfg.pop("model", None)
    if model_before is not None:
        cfg["model"] = copy.deepcopy(model_before)
    save_hermes_config(cfg)


def _restore_primary_route(model_before: Any, active_provider_before: Any) -> None:
    """Attempt both halves of temporary picker-route restoration."""
    errors: list[BaseException] = []
    try:
        _restore_model_cfg(model_before)
    except BaseException as exc:
        errors.append(exc)
    try:
        _restore_auth_active_provider(active_provider_before)
    except BaseException as exc:
        errors.append(exc)
    if errors:
        details = "; ".join(str(exc) for exc in errors)
        raise RuntimeError(f"Could not fully restore the primary route: {details}") from errors[0]


def _entries(n: int) -> str:
    return f"{n} {'entry' if n == 1 else 'entries'}"


def _print_chain(heading: str, chain: List[Dict[str, Any]]) -> None:
    print(f"  {heading} ({_entries(len(chain))}):")
    print("".join(f"    {i}. {_format_entry(entry)}\n" for i, entry in enumerate(chain, 1)))


def _load_chain(empty_message: str):
    """Load config + chain; print ``empty_message`` block and return ``(config, None)`` when empty."""
    config = load_hermes_config()
    chain = _read_chain(config)
    if not chain:
        print(f"\n{empty_message}\n")
    return config, chain or None


def _describe_primary(config: Dict[str, Any]) -> Optional[str]:
    """One-line description of the primary model for display purposes."""
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        provider = (model_cfg.get("provider") or "?").strip() or "?"
        model = (model_cfg.get("default") or model_cfg.get("model") or "?").strip() or "?"
        return f"{model}  (via {provider})"
    return model_cfg.strip() or None if isinstance(model_cfg, str) else None


def cmd_fallback_list(args) -> None:  # noqa: ARG001
    """Print the current fallback chain, plugin heuristic settings, and available providers."""
    config = load_hermes_config()
    chain = _read_chain(config)
    settings = get_plugin_settings(config)
    criteria = build_criteria_from_settings(settings, config=config)
    available_providers = discover_existing_providers(config=config)

    print()
    if primary := _describe_primary(config):
        print(f"  Primary:             {primary}")
    heuristics_status = (
        "enabled"
        if (is_fallback_heuristics_enabled(config) or settings.get("enabled", True))
        else "disabled (default)"
    )
    print(f"  Heuristics:          {heuristics_status}")
    print(f"  Heuristic criteria:  {criteria.describe()}")
    print(
        f"  Provider switching:  {'enabled' if settings.get('allow_provider_switch', True) else 'disabled'} "
        f"(prefer current: {'yes' if settings.get('prefer_current_provider', True) else 'no'})"
    )
    print(
        f"  Available providers: {', '.join(available_providers) if available_providers else 'none detected'}\n"
    )

    if not chain:
        print("  No static/stored fallback entries in config.yaml.")
        print("  Run `hermes fallback-heuristic sync` to populate heuristic rules or `resolve` to preview.\n")
        if is_first_launch(config):
            report = check_free_model_providers(config=config, allow_network=True)
            print(format_first_launch_guide(report, config=config) + "\n")
        return

    _print_chain("Fallback chain", chain)
    print("  Tried in order when the primary fails (rate-limit, 5xx, connection/model errors).\n")
    if is_first_launch(config):
        report = check_free_model_providers(config=config, allow_network=True)
        print(format_first_launch_guide(report, config=config) + "\n")


def cmd_fallback_add(args) -> None:
    """Add a fallback entry either via CLI flags (no TTY required) or interactive model picker."""
    provider_arg = getattr(args, "provider", None)
    model_arg = getattr(args, "model", None)
    criteria_arg = getattr(args, "criteria", None)

    # Non-interactive path when any flag is given
    if provider_arg or model_arg or criteria_arg:
        cfg = load_hermes_config()
        chain = _read_chain(cfg)
        entry: Dict[str, Any] = {}
        entry["provider"] = normalize_provider_name(provider_arg or "auto")
        if model_arg:
            entry["model"] = str(model_arg).strip()
        if criteria_arg:
            entry["criteria"] = str(criteria_arg).strip()
        elif not model_arg:
            # Build heuristic entry from flags
            crit_dict: Dict[str, Any] = {"provider": entry["provider"]}
            for key in (
                "min_params",
                "max_params",
                "min_context",
                "max_context",
                "sort_by",
                "free_only",
                "require_tools",
                "flash_only",
                "vendor",
                "name_filter",
                "reasoning_effort",
                "max_candidates",
                "allow_provider_switch",
                "prefer_current_provider",
            ):
                val = getattr(args, key, None)
                if val is not None:
                    crit_dict[key] = val
            entry["criteria"] = crit_dict
        if getattr(args, "reasoning_effort", None):
            entry["reasoning_effort"] = getattr(args, "reasoning_effort")
        chain.append(entry)
        cfg["fallback_heuristics"] = True
        _write_chain(cfg, chain)
        save_hermes_config(cfg)
        print(f"\n  Added fallback: {_format_entry(entry)}")
        print(f"  Chain is now {_entries(len(chain))} long.\n")
        return

    # Interactive picker path (same as hermes model)
    from hermes_cli.main import _require_tty, select_provider_and_model

    _require_tty("fallback add")

    model_before = copy.deepcopy(load_hermes_config().get("model"))
    active_provider_before = _snapshot_auth_active_provider()
    print(
        "\n  Adding a fallback provider.  The picker below is the same one used by\n"
        "  `hermes model` — select the provider + model you want as a fallback.\n"
    )

    try:
        select_provider_and_model(args=args)
        after_cfg = load_hermes_config()
        model_after = after_cfg.get("model")
        new_entry = _extract_fallback_from_model_cfg(model_after)
    except BaseException as picker_error:
        try:
            _restore_primary_route(model_before, active_provider_before)
        except Exception as restore_error:
            picker_error.add_note(
                "Could not fully restore the primary route after fallback "
                f"selection failed: {restore_error}"
            )
        raise

    _restore_primary_route(model_before, active_provider_before)

    if not new_entry:
        print("\n  No fallback added.")
        return

    new_ident = _identity(new_entry)
    primary_entry = _extract_fallback_from_model_cfg(model_before)
    if primary_entry and _same_deployment(_identity(primary_entry), new_ident):
        print(f"\n  Selected model matches the current primary ({_format_entry(new_entry)}).")
        print("  A provider cannot be a fallback for itself — no change.")
        return

    final_cfg = load_hermes_config()
    chain = _read_chain(final_cfg)
    if any(_same_deployment(_identity(existing), new_ident) for existing in chain):
        print(f"\n  {_format_entry(new_entry)} is already in the fallback chain — skipped.")
        return
    chain.append(new_entry)
    _write_chain(final_cfg, chain)
    save_hermes_config(final_cfg)
    print(f"\n  Added fallback: {_format_entry(new_entry)}")
    print(f"  Chain is now {_entries(len(chain))} long.\n")
    print("  Run `hermes fallback-heuristic list` to view, or `hermes fallback-heuristic remove` to delete.")


def cmd_fallback_remove(args) -> None:
    """Remove an entry from the chain by 1-based index or interactive prompt."""
    config, chain = _load_chain("  No fallback providers configured — nothing to remove.")
    if chain is None:
        return

    index_arg = getattr(args, "index", None)
    if index_arg is not None:
        try:
            idx = int(index_arg) - 1
        except (TypeError, ValueError):
            print(f"\n  Invalid index: {index_arg!r}")
            return
        if idx < 0 or idx >= len(chain):
            print(f"\n  Index {index_arg} out of range (1..{len(chain)}).")
            return
    else:
        from hermes_cli.setup import _curses_prompt_choice

        idx = _curses_prompt_choice(
            "Select a fallback to remove:",
            [_format_entry(e) for e in chain] + ["Cancel"],
            0,
        )
        if idx is None or idx < 0 or idx >= len(chain):
            print("\n  Cancelled — no change.")
            return

    removed = chain.pop(idx)
    _write_chain(config, chain)
    save_hermes_config(config)
    print(f"\n  Removed fallback: {_format_entry(removed)}")
    print(
        f"  Chain is now {_entries(len(chain))} long.\n"
        if chain
        else "  Fallback chain is now empty.\n"
    )


def cmd_fallback_clear(args) -> None:
    """Remove all fallback entries (with confirmation unless --yes is passed)."""
    config, chain = _load_chain("  No fallback providers configured — nothing to clear.")
    if chain is None:
        return
    if not getattr(args, "yes", False):
        print()
        _print_chain("Current fallback chain", chain)
        try:
            resp = input("  Clear all entries? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n  Cancelled.")
            return
        if resp not in {"y", "yes"}:
            print("  Cancelled — no change.")
            return
    _write_chain(config, [])
    config.pop("fallback_model", None)
    save_hermes_config(config)
    print("\n  Fallback chain cleared.\n")


def cmd_fallback_heuristics(args) -> None:
    """Turn fallback heuristics on or off, or display current status."""
    action = getattr(args, "heuristics_action", None)
    if action is None:
        rem = getattr(args, "args", None)
        if rem and len(rem) > 0:
            action = rem[0]

    action_norm = str(action or "").strip().lower()
    cfg = load_hermes_config()

    if action_norm in ("on", "enable", "true", "1"):
        cfg["fallback_heuristics"] = True
        save_hermes_config(cfg)
        update_plugin_settings({"enabled": True}, sync_chain=True)
        print("\n  Fallback heuristics: ENABLED")
        print("  Dynamic heuristics in fallback_providers will now resolve across available providers.\n")
        return

    if action_norm in ("off", "disable", "false", "0"):
        cfg["fallback_heuristics"] = False
        save_hermes_config(cfg)
        update_plugin_settings({"enabled": False}, sync_chain=False)
        print("\n  Fallback heuristics: DISABLED (default)")
        print("  Only explicit static fallback models will be used.\n")
        return

    current = is_fallback_heuristics_enabled(cfg)
    status_str = "ENABLED" if current else "DISABLED (default)"
    print(f"\n  Fallback heuristics: {status_str}")
    print("  To enable:   hermes fallback-heuristic heuristics on")
    print("  To disable:  hermes fallback-heuristic heuristics off\n")


def _criteria_from_cli_args(args, config: Dict[str, Any]) -> FallbackCriteria:
    """Build FallbackCriteria merging saved plugin settings with CLI flags."""
    settings = get_plugin_settings(config)
    merged = dict(settings)

    if getattr(args, "criteria", None):
        parsed = parse_fallback_criteria(getattr(args, "criteria"))
        return parsed

    if getattr(args, "provider", None):
        merged["provider"] = args.provider
    if getattr(args, "providers", None):
        merged["providers"] = parse_provider_list(args.providers)
    if getattr(args, "exclude_providers", None):
        merged["exclude_providers"] = parse_provider_list(args.exclude_providers)
    if getattr(args, "min_params", None) is not None:
        merged["min_params"] = args.min_params
    if getattr(args, "max_params", None) is not None:
        merged["max_params"] = args.max_params
    if getattr(args, "min_context", None) is not None:
        merged["min_context"] = args.min_context
    if getattr(args, "max_context", None) is not None:
        merged["max_context"] = args.max_context
    if getattr(args, "sort_by", None):
        merged["sort_by"] = args.sort_by
    if getattr(args, "free_only", None) is not None:
        merged["free_only"] = args.free_only
    if getattr(args, "require_tools", None) is not None:
        merged["require_tools"] = args.require_tools
    if getattr(args, "flash_only", None) is not None:
        merged["flash_only"] = args.flash_only
    if getattr(args, "vendor", None) is not None:
        merged["vendor"] = args.vendor
    if getattr(args, "name_filter", None) is not None:
        merged["name_filter"] = args.name_filter
    if getattr(args, "reasoning_effort", None) is not None:
        merged["reasoning_effort"] = args.reasoning_effort
    if getattr(args, "max_candidates", None) is not None:
        merged["max_candidates"] = args.max_candidates
    if getattr(args, "allow_provider_switch", None) is not None:
        merged["allow_provider_switch"] = args.allow_provider_switch
    if getattr(args, "prefer_current_provider", None) is not None:
        merged["prefer_current_provider"] = args.prefer_current_provider

    return build_criteria_from_settings(merged, config=config)


def cmd_fallback_resolve(args) -> None:
    """Resolve and display ranked candidate models across existing providers without an LLM call."""
    config = load_hermes_config()
    criteria = _criteria_from_cli_args(args, config)
    candidates = resolve_best_available_model(criteria, config=config)

    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "criteria": criteria.describe(),
                    "count": len(candidates),
                    "candidates": candidates,
                },
                indent=2,
            )
        )
        return

    print(f"\n  Heuristic criteria: {criteria.describe()}")
    if not candidates:
        print("  No matching models found across available providers.\n")
        return

    print(f"  Resolved {len(candidates)} candidate(s):")
    for idx, c in enumerate(candidates, 1):
        thinking = (
            f"  [thinking: {c.get('reasoning_effort')}]"
            if c.get("reasoning_effort") and c.get("reasoning_effort") != "default"
            else ""
        )
        base = f"  [{c.get('base_url')}]" if c.get("base_url") else ""
        print(f"    {idx}. {c['model']}  (via {c['provider']}){thinking}{base}")
    print()


def cmd_fallback_switch(args) -> None:
    """Switch the active model (and provider if needed) to the best heuristic match."""
    config = load_hermes_config()
    criteria = _criteria_from_cli_args(args, config)
    persist = not getattr(args, "dry_run", False)
    result = switch_model_with_heuristic(criteria=criteria, config=config, persist=persist)

    if result is None:
        result = {"ok": False, "error": "No matching candidate models found across available providers."}

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
        return

    if not result.get("ok"):
        print(f"\n  Switch failed: {result.get('error')}\n")
        raise SystemExit(1)

    action_word = "Would switch" if getattr(args, "dry_run", False) else "Switched"
    print(f"\n  {action_word} model:")
    print(f"    Previous: {result['previous_model'] or '?'}  (via {result['previous_provider'] or '?'})")
    print(f"    Selected: {result['provider']}/{result['model']}")
    print(f"    Provider switched: {'yes' if result['provider_switched'] else 'no'}")
    print(f"    Matched criteria:  {result.get('criteria_matched')}\n")


def cmd_fallback_providers(args) -> None:
    """Display supported providers and whether each is currently configured/available."""
    config = load_hermes_config()
    statuses = get_provider_status_list(config=config)
    show_all = getattr(args, "all", False)
    filtered = statuses if show_all else [s for s in statuses if s.get("available")]

    if getattr(args, "json", False):
        print(json.dumps({"providers": statuses}, indent=2))
        return

    print("\n  Model Providers:")
    if not filtered:
        print("    No configured/available providers detected.")
        print("    Pass `--all` to view all supported providers and their required env vars.\n")
        return

    for s in filtered:
        badge = "AVAILABLE" if s.get("available") else "unconfigured"
        free_tag = " [free-tier defaults]" if s.get("free_tier_default") else ""
        env_tag = f" (env: {', '.join(s['env_vars'])})" if s.get("env_vars") else ""
        print(f"    - {s['provider']:<16} {badge:<13}{free_tag}{env_tag}")
    if not show_all:
        unavail_count = sum(1 for s in statuses if not s.get("available"))
        if unavail_count:
            print(f"\n  ({unavail_count} additional unconfigured providers hidden; pass `--all` to show)")
    print()


def _parse_kv_pairs(pairs: List[str]) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    bool_keys = {
        "enabled",
        "allow_provider_switch",
        "prefer_current_provider",
        "free_only",
        "require_tools",
        "flash_only",
        "auto_switch_on_unavailable",
    }
    list_keys = {"providers", "exclude_providers", "exclude_models"}

    for item in pairs:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        key = k.strip().lower().replace("-", "_")
        val_str = v.strip()
        if key in bool_keys:
            updates[key] = val_str.lower() in ("1", "true", "yes", "on")
        elif key in list_keys:
            if key == "exclude_models":
                updates[key] = [x.strip() for x in val_str.split(",") if x.strip()]
            else:
                updates[key] = parse_provider_list(val_str)
        elif key == "max_candidates":
            updates[key] = int(val_str)
        else:
            updates[key] = val_str
    return updates


def cmd_fallback_set(args) -> None:
    """Update heuristic settings without an LLM call and sync to config.yaml."""
    updates: Dict[str, Any] = {}
    kv_items = getattr(args, "settings", None) or []
    if kv_items:
        updates.update(_parse_kv_pairs(kv_items))

    for key in (
        "provider",
        "min_params",
        "max_params",
        "min_context",
        "max_context",
        "sort_by",
        "free_only",
        "require_tools",
        "flash_only",
        "vendor",
        "name_filter",
        "reasoning_effort",
        "max_candidates",
        "allow_provider_switch",
        "prefer_current_provider",
    ):
        val = getattr(args, key, None)
        if val is not None:
            updates[key] = val

    if getattr(args, "providers", None):
        updates["providers"] = parse_provider_list(args.providers)
    if getattr(args, "exclude_providers", None):
        updates["exclude_providers"] = parse_provider_list(args.exclude_providers)

    if not updates:
        print("\n  No settings specified. Example:")
        print("    hermes fallback-heuristic set provider=auto min_params=30B min_context=128k sort_by=params\n")
        return

    merged = update_plugin_settings(updates, sync_chain=True)
    criteria = build_criteria_from_settings(merged)
    print("\n  Updated fallback heuristic settings (synced to config.yaml):")
    print(f"    Active criteria: {criteria.describe()}\n")


def cmd_fallback_sync(args) -> None:  # noqa: ARG001
    """Sync plugin heuristic settings into config.yaml's fallback_providers."""
    entries = sync_fallback_chain_to_config()
    print(f"\n  Synced {len(entries)} fallback entry/entries into config.yaml.\n")


def cmd_fallback_free_providers(args) -> None:
    """Run a live/cached check across catalogs and display all providers offering free models."""
    config = load_hermes_config()
    report = check_free_model_providers(config=config, allow_network=True)
    if getattr(args, "json", False):
        print(json.dumps({"free_model_providers": report, "count": len(report)}, indent=2))
        return
    print("\n" + format_first_launch_guide(report, config=config) + "\n")


def cmd_fallback_setup(args) -> None:
    """Run first-launch onboarding: check all providers offering free models and let the user choose."""
    config = load_hermes_config()
    report = check_free_model_providers(config=config, allow_network=True)

    raw_provs = getattr(args, "providers", None) or getattr(args, "provider", None)
    if getattr(args, "done", False):
        mark_first_launch_done(config=config, save=True)
        if getattr(args, "json", False):
            print(json.dumps({"ok": True, "first_launch_completed": True}, indent=2))
        else:
            print("\n  ✓ First-launch provider check marked complete.\n")
        return

    if raw_provs:
        tokens = raw_provs if isinstance(raw_provs, list) else [str(raw_provs)]
        selected: List[str] = []
        for raw_item in ",".join(tokens).split(","):
            tok = raw_item.strip()
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
        settings = mark_first_launch_done(selected_providers=selected, config=config, save=True)
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "ok": True,
                        "first_launch_completed": True,
                        "selected_providers": settings.get("providers", []),
                        "free_model_providers": report,
                    },
                    indent=2,
                )
            )
            return
        print(f"\n  ✓ Saved preferred fallback providers: {', '.join(settings.get('providers', []))}")
        missing = [
            p
            for p in report
            if p["provider"] in selected and not p.get("authenticated") and p.get("env_vars")
        ]
        if missing:
            print("\n  Next step — create & set free API keys for your unconfigured providers:")
            for mp in missing:
                env_name = mp["env_vars"][0]
                print(f"    • {mp['display_name']} ({mp['provider']}): {mp['signup_url']}")
                print(f"      export {env_name}=\"your-api-key-here\"")
        print()
        return

    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "ok": True,
                    "is_first_launch": is_first_launch(config),
                    "free_model_providers": report,
                },
                indent=2,
            )
        )
        return

    run_interactive_first_launch_wizard(config=config, allow_network=True)


def _add_heuristic_filter_flags(parser: argparse.ArgumentParser) -> None:
    """Attach common heuristic filter flags to a subparser."""
    parser.add_argument(
        "--provider",
        "-p",
        help="Target provider ('auto' for any existing provider, or e.g. openrouter, anthropic, openai, gemini, deepseek, ollama, etc.)",
    )
    parser.add_argument(
        "--providers",
        help="Comma-separated ordered list of candidate providers (e.g. 'anthropic,openrouter,openai,gemini')",
    )
    parser.add_argument(
        "--exclude-providers",
        help="Comma-separated providers to exclude",
    )
    parser.add_argument("--min-params", help="Minimum parameter count (e.g. '30B', '70B')")
    parser.add_argument("--max-params", help="Maximum parameter count (e.g. '405B')")
    parser.add_argument("--min-context", help="Minimum context length (e.g. '64k', '128k', '1m')")
    parser.add_argument("--max-context", help="Maximum context length (e.g. '2m')")
    parser.add_argument(
        "--sort-by",
        choices=[
            "params",
            "largest_parameter_count",
            "context",
            "greatest_context",
            "recency",
            "latest",
            "latest_flash",
            "smallest",
            "provider_preference",
        ],
        help="Primary ranking dimension",
    )
    parser.add_argument(
        "--free-only",
        dest="free_only",
        action="store_true",
        default=None,
        help="Restrict to free-tier models",
    )
    parser.add_argument(
        "--include-paid",
        dest="free_only",
        action="store_false",
        help="Include both free and paid models",
    )
    parser.add_argument(
        "--require-tools",
        dest="require_tools",
        action="store_true",
        default=None,
        help="Require function/tool calling support",
    )
    parser.add_argument(
        "--no-require-tools",
        dest="require_tools",
        action="store_false",
        help="Allow models without tool calling support",
    )
    parser.add_argument(
        "--flash-only",
        dest="flash_only",
        action="store_true",
        default=None,
        help="Restrict to fast/flash/mini/haiku/turbo tier models",
    )
    parser.add_argument("--vendor", help="Filter by model vendor prefix (e.g. 'qwen', 'anthropic', 'google')")
    parser.add_argument("--name-filter", help="Substring or glob filter on model ID")
    parser.add_argument(
        "--reasoning-effort",
        choices=["default", "none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort to apply to resolved fallback models",
    )
    parser.add_argument("--max-candidates", type=int, help="Maximum candidates to return")
    parser.add_argument(
        "--allow-provider-switch",
        dest="allow_provider_switch",
        action="store_true",
        default=None,
        help="Allow switching to another configured provider when primary fails",
    )
    parser.add_argument(
        "--no-provider-switch",
        dest="allow_provider_switch",
        action="store_false",
        help="Restrict fallback to the current provider only",
    )
    parser.add_argument(
        "--prefer-current-provider",
        dest="prefer_current_provider",
        action="store_true",
        default=None,
        help="Try the active provider first before switching providers",
    )
    parser.add_argument(
        "--no-prefer-current-provider",
        dest="prefer_current_provider",
        action="store_false",
        help="Rank strictly by heuristic score across all providers",
    )
    parser.add_argument("--criteria", "-c", help="Shorthand criteria string (e.g. 'provider=auto,params>=30B,context>=128k')")


def setup_cli_parser(parser: argparse.ArgumentParser) -> None:
    """Configure argparse subparsers for ``hermes fallback-heuristic`` / ``hermes fallback``."""
    subparsers = parser.add_subparsers(dest="fallback_command")

    # list / status
    subparsers.add_parser("list", aliases=["ls", "status"], help="Show primary model, heuristic settings, and fallback chain")

    # resolve
    p_resolve = subparsers.add_parser(
        "resolve",
        help="Resolve and rank candidate models across existing providers without an LLM call",
    )
    _add_heuristic_filter_flags(p_resolve)
    p_resolve.add_argument("--json", action="store_true", help="Output JSON")

    # switch
    p_switch = subparsers.add_parser(
        "switch",
        help="Switch active model (and provider if needed) to the best available heuristic match",
    )
    _add_heuristic_filter_flags(p_switch)
    p_switch.add_argument("--dry-run", action="store_true", help="Preview switch without modifying config.yaml")
    p_switch.add_argument("--json", action="store_true", help="Output JSON")

    # providers
    p_providers = subparsers.add_parser(
        "providers",
        help="List model providers and whether they are configured/available",
    )
    p_providers.add_argument("--all", "-a", action="store_true", help="Show unconfigured providers too")
    p_providers.add_argument("--json", action="store_true", help="Output JSON")

    # set
    p_set = subparsers.add_parser(
        "set",
        aliases=["config", "configure"],
        help="Configure heuristic parameters without an LLM call",
    )
    p_set.add_argument("settings", nargs="*", help="key=value pairs (e.g. provider=auto min_params=30B)")
    _add_heuristic_filter_flags(p_set)

    # add
    p_add = subparsers.add_parser(
        "add",
        help="Add a static or heuristic fallback entry (via flags or interactive picker)",
    )
    p_add.add_argument("--model", "-m", help="Explicit model ID or 'auto:<criteria>'")
    _add_heuristic_filter_flags(p_add)

    # remove
    p_remove = subparsers.add_parser("remove", aliases=["rm"], help="Remove a fallback entry")
    p_remove.add_argument("index", nargs="?", type=int, help="1-based index of entry to remove")

    # clear
    p_clear = subparsers.add_parser("clear", help="Remove all fallback entries")
    p_clear.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")

    # heuristics / on / off
    p_heur = subparsers.add_parser("heuristics", aliases=["heuristic"], help="Enable or disable fallback heuristics")
    p_heur.add_argument("heuristics_action", nargs="?", choices=["on", "off", "status"], help="Action (on, off, status)")

    subparsers.add_parser("on", help="Enable fallback heuristics")
    subparsers.add_parser("off", help="Disable fallback heuristics")
    subparsers.add_parser("sync", help="Sync heuristic settings to config.yaml fallback_providers")

    # setup / onboarding / first-launch
    p_setup = subparsers.add_parser(
        "setup",
        aliases=["onboarding", "first-launch", "init"],
        help="First-launch guide: check all providers offering free models and choose which to use",
    )
    p_setup.add_argument(
        "providers",
        nargs="*",
        help="Optional provider IDs or numbers to select (e.g. openrouter,gemini,nvidia or 1,2,3)",
    )
    p_setup.add_argument("--done", action="store_true", help="Mark first-launch setup complete")
    p_setup.add_argument("--json", action="store_true", help="Output JSON")

    # free-providers / check-free
    p_free = subparsers.add_parser(
        "free-providers",
        aliases=["free", "check-free"],
        help="Run a live check of all providers offering free ($0) models",
    )
    p_free.add_argument("--json", action="store_true", help="Output JSON")


def _cmd_on(args) -> None:
    setattr(args, "heuristics_action", "on")
    cmd_fallback_heuristics(args)


def _cmd_off(args) -> None:
    setattr(args, "heuristics_action", "off")
    cmd_fallback_heuristics(args)


_SUBCOMMANDS = {
    **dict.fromkeys((None, "", "list", "ls", "status"), cmd_fallback_list),
    "add": cmd_fallback_add,
    **dict.fromkeys(("remove", "rm"), cmd_fallback_remove),
    "clear": cmd_fallback_clear,
    **dict.fromkeys(("heuristics", "heuristic"), cmd_fallback_heuristics),
    "on": _cmd_on,
    "off": _cmd_off,
    "resolve": cmd_fallback_resolve,
    "switch": cmd_fallback_switch,
    "providers": cmd_fallback_providers,
    **dict.fromkeys(("set", "config", "configure"), cmd_fallback_set),
    "sync": cmd_fallback_sync,
    **dict.fromkeys(("setup", "onboarding", "first-launch", "init"), cmd_fallback_setup),
    **dict.fromkeys(("free-providers", "free", "check-free"), cmd_fallback_free_providers),
}


def cmd_fallback(args) -> None:
    """Top-level dispatcher for ``hermes fallback-heuristic [subcommand]`` / ``hermes fallback [subcommand]``."""
    sub = getattr(args, "fallback_command", None)
    handler = _SUBCOMMANDS.get(sub)
    if handler is None:
        print(f"Unknown fallback subcommand: {sub}")
        print(
            "Use one of: list, status, resolve, switch, providers, set, add, remove, clear, heuristics, on, off, sync"
        )
        raise SystemExit(2)
    handler(args)


cmd_fallback_heuristic = cmd_fallback
