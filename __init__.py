"""Hermes Model Fallback Heuristic Plugin.

Dynamic multi-provider model fallback heuristic for Hermes Agent.
Resolves available models across any configured provider using criteria such as
provider preference, context window size, parameter count, release recency,
flash/fast tier, tool-calling support, free/paid tier, vendor, name filter,
and reasoning effort — working seamlessly with or without an LLM call.
"""

from __future__ import annotations

from typing import Any, Mapping

try:
    from .cli import (
        cmd_fallback,
        cmd_fallback_add,
        cmd_fallback_clear,
        cmd_fallback_free_providers,
        cmd_fallback_heuristic,
        cmd_fallback_heuristics,
        cmd_fallback_list,
        cmd_fallback_providers,
        cmd_fallback_remove,
        cmd_fallback_resolve,
        cmd_fallback_set,
        cmd_fallback_setup,
        cmd_fallback_switch,
        cmd_fallback_sync,
        setup_cli_parser,
    )
    from .config_sync import (
        DEFAULT_PLUGIN_SETTINGS,
        PLUGIN_ID,
        build_criteria_from_settings,
        ensure_available_model_or_fallback,
        get_fallback_chain,
        get_plugin_settings,
        get_stored_fallback_rules,
        is_fallback_heuristics_enabled,
        load_hermes_config,
        resolve_best_available_model,
        save_hermes_config,
        scoped_fallback_chain,
        switch_model_with_heuristic,
        sync_fallback_chain_to_config,
        update_plugin_settings,
    )
    from .handlers import (
        api_request_error_hook,
        handle_configure,
        handle_providers,
        handle_resolve,
        handle_slash_command,
        handle_switch,
        llm_execution_middleware,
        llm_request_middleware,
        on_session_start_hook,
        pre_llm_call_hook,
    )
    from .heuristics import (
        DEFAULT_PROVIDER_ORDER,
        PAID_FIRST_PROVIDERS,
        PROVIDER_ALIASES,
        PROVIDER_ENV_VARS,
        FallbackCriteria,
        ModelCandidateMetadata,
        discover_existing_providers,
        extract_param_billions,
        fetch_all_candidates,
        fetch_openrouter_candidates,
        fetch_provider_candidates,
        filter_and_rank_candidates,
        get_provider_status_list,
        is_flash_model,
        is_heuristic_fallback_entry,
        is_provider_available,
        normalize_provider_name,
        parse_context_length,
        parse_fallback_criteria,
        parse_param_size,
        parse_provider_list,
        resolve_fallback_candidates,
         resolve_heuristic_entry,
    )
    from .onboarding import (
        FREE_PROVIDER_CATALOG_META,
        check_free_model_providers,
        format_first_launch_guide,
        is_first_launch,
        mark_first_launch_done,
        run_interactive_first_launch_wizard,
        save_provider_api_key,
    )
    from .reasoning import (
        VALID_REASONING_EFFORTS,
        format_fallback_label,
        format_fallback_notice,
        parse_reasoning_effort,
        reresolve_fallback_reasoning_config,
        resolve_fallback_entry_reasoning_config,
        resolve_per_model_reasoning_effort,
    )
    from .schemas import (
        CONFIGURE_SCHEMA,
        PROVIDERS_SCHEMA,
        RESOLVE_SCHEMA,
        SWITCH_SCHEMA,
    )
except ImportError:
    from cli import (  # type: ignore
        cmd_fallback,
        cmd_fallback_add,
        cmd_fallback_clear,
        cmd_fallback_free_providers,
        cmd_fallback_heuristic,
        cmd_fallback_heuristics,
        cmd_fallback_list,
        cmd_fallback_providers,
        cmd_fallback_remove,
        cmd_fallback_resolve,
        cmd_fallback_set,
        cmd_fallback_setup,
        cmd_fallback_switch,
        cmd_fallback_sync,
        setup_cli_parser,
    )
    from config_sync import (  # type: ignore
        DEFAULT_PLUGIN_SETTINGS,
        PLUGIN_ID,
        build_criteria_from_settings,
        ensure_available_model_or_fallback,
        get_fallback_chain,
        get_plugin_settings,
        get_stored_fallback_rules,
        is_fallback_heuristics_enabled,
        load_hermes_config,
        resolve_best_available_model,
        save_hermes_config,
        scoped_fallback_chain,
        switch_model_with_heuristic,
        sync_fallback_chain_to_config,
        update_plugin_settings,
    )
    from handlers import (  # type: ignore
        api_request_error_hook,
        handle_configure,
        handle_providers,
        handle_resolve,
        handle_slash_command,
        handle_switch,
        llm_execution_middleware,
        llm_request_middleware,
        on_session_start_hook,
        pre_llm_call_hook,
    )
    from heuristics import (  # type: ignore
        DEFAULT_PROVIDER_ORDER,
        PAID_FIRST_PROVIDERS,
        PROVIDER_ALIASES,
        PROVIDER_ENV_VARS,
        FallbackCriteria,
        ModelCandidateMetadata,
        discover_existing_providers,
        extract_param_billions,
        fetch_all_candidates,
        fetch_openrouter_candidates,
        fetch_provider_candidates,
        filter_and_rank_candidates,
        get_provider_status_list,
        is_flash_model,
        is_heuristic_fallback_entry,
        is_provider_available,
        normalize_provider_name,
        parse_context_length,
        parse_fallback_criteria,
        parse_param_size,
        parse_provider_list,
        resolve_fallback_candidates,
        resolve_heuristic_entry,
    )
    from onboarding import (  # type: ignore
        FREE_PROVIDER_CATALOG_META,
        check_free_model_providers,
        format_first_launch_guide,
        is_first_launch,
        mark_first_launch_done,
        run_interactive_first_launch_wizard,
        save_provider_api_key,
    )
    from reasoning import (  # type: ignore
        VALID_REASONING_EFFORTS,
        format_fallback_label,
        format_fallback_notice,
        parse_reasoning_effort,
        reresolve_fallback_reasoning_config,
        resolve_fallback_entry_reasoning_config,
        resolve_per_model_reasoning_effort,
    )
    from schemas import (  # type: ignore
        CONFIGURE_SCHEMA,
        PROVIDERS_SCHEMA,
        RESOLVE_SCHEMA,
        SWITCH_SCHEMA,
    )


def _render_system_prompt_section(_session_info: Mapping[str, Any]) -> str:
    """Provide concise guidance on multi-provider model fallback heuristic tools."""
    first_launch_note = ""
    try:
        if is_first_launch():
            first_launch_note = (
                " First launch detected: use `fallback_heuristic_providers` or `/fallback setup` "
                "to show the user all providers offering free models (OpenRouter, Gemini, NVIDIA NIM, "
                "OpenCode Zen, KiloCode, Nous, etc.) and let them decide which API keys to configure."
            )
    except Exception:
        pass
    return (
        "Model Fallback Heuristic plugin is active. You can inspect available & free-model providers "
        "(`fallback_heuristic_providers`), resolve ranked fallback candidates across any "
        "configured provider by provider/context/params/recency/flash/tools/free tier "
        "(`fallback_heuristic_resolve`), switch active provider/model (`fallback_heuristic_switch`), "
        "or update persistent heuristic settings (`fallback_heuristic_configure`). "
        "Users can also manage heuristics without an LLM call via `/fallback` or `hermes fallback-heuristic`."
        + first_launch_note
    )


def register(ctx: Any) -> None:
    """Register tools, lifecycle hooks, middleware, slash commands, and CLI subcommands."""
    # 1. Tools (declared in plugin.yaml provides_tools)
    ctx.register_tool(
        name="fallback_heuristic_resolve",
        toolset="fallback_heuristic",
        schema=RESOLVE_SCHEMA,
        handler=handle_resolve,
        description="Resolve and rank candidate fallback models across any configured provider",
        emoji="🧭",
    )
    ctx.register_tool(
        name="fallback_heuristic_switch",
        toolset="fallback_heuristic",
        schema=SWITCH_SCHEMA,
        handler=handle_switch,
        description="Switch active model and provider using the fallback heuristic",
        emoji="🔀",
    )
    ctx.register_tool(
        name="fallback_heuristic_configure",
        toolset="fallback_heuristic",
        schema=CONFIGURE_SCHEMA,
        handler=handle_configure,
        description="View or update persistent fallback heuristic settings",
        emoji="⚙️",
    )
    ctx.register_tool(
        name="fallback_heuristic_providers",
        toolset="fallback_heuristic",
        schema=PROVIDERS_SCHEMA,
        handler=handle_providers,
        description="List model providers and their authentication/availability status",
        emoji="🔌",
    )

    # 2. Lifecycle hooks (declared in plugin.yaml provides_hooks)
    ctx.register_hook("on_session_start", on_session_start_hook)
    ctx.register_hook("pre_llm_call", pre_llm_call_hook)
    ctx.register_hook("api_request_error", api_request_error_hook)

    # 3. Middleware (declared in plugin.yaml provides_middleware)
    ctx.register_middleware("llm_request", llm_request_middleware)
    ctx.register_middleware("llm_execution", llm_execution_middleware)

    # 4. In-session slash commands (work without an LLM call)
    if hasattr(ctx, "register_command"):
        ctx.register_command(
            name="fallback",
            handler=handle_slash_command,
            description="Manage multi-provider model fallback heuristics (status, resolve, switch, set, providers)",
            args_hint="[status|on|off|set|resolve|switch|providers|sync]",
            argument_mode="text",
        )
        ctx.register_command(
            name="fallback-heuristic",
            handler=handle_slash_command,
            description="Manage multi-provider model fallback heuristics (status, resolve, switch, set, providers)",
            args_hint="[status|on|off|set|resolve|switch|providers|sync]",
            argument_mode="text",
        )

    # 5. CLI subcommands (work from terminal without an LLM call)
    if hasattr(ctx, "register_cli_command"):
        ctx.register_cli_command(
            name="fallback-heuristic",
            help="Multi-provider model fallback heuristic (resolve, switch, set, providers, list)",
            setup_fn=setup_cli_parser,
            handler_fn=cmd_fallback_heuristic,
            description="Configure and run dynamic multi-provider model fallback heuristics without an LLM call.",
        )
        ctx.register_cli_command(
            name="fallback",
            help="Manage the fallback provider chain and dynamic model heuristics",
            setup_fn=setup_cli_parser,
            handler_fn=cmd_fallback,
            description="Manage fallback providers and dynamic model heuristics.",
        )

    # 6. System prompt section
    if hasattr(ctx, "register_system_prompt_section"):
        try:
            ctx.register_system_prompt_section(
                id="model_fallback_heuristic",
                content=_render_system_prompt_section,
                position="after_memory",
                max_chars=600,
            )
        except Exception:
            pass


__all__ = [
    "CONFIGURE_SCHEMA",
    "DEFAULT_PLUGIN_SETTINGS",
    "DEFAULT_PROVIDER_ORDER",
    "FallbackCriteria",
    "ModelCandidateMetadata",
    "PAID_FIRST_PROVIDERS",
    "PLUGIN_ID",
    "PROVIDERS_SCHEMA",
    "PROVIDER_ALIASES",
    "PROVIDER_ENV_VARS",
    "RESOLVE_SCHEMA",
    "SWITCH_SCHEMA",
    "VALID_REASONING_EFFORTS",
    "api_request_error_hook",
    "build_criteria_from_settings",
    "cmd_fallback",
    "cmd_fallback_add",
    "cmd_fallback_clear",
    "cmd_fallback_heuristic",
    "cmd_fallback_heuristics",
    "cmd_fallback_list",
    "cmd_fallback_providers",
    "cmd_fallback_remove",
    "cmd_fallback_resolve",
    "cmd_fallback_set",
    "cmd_fallback_switch",
    "cmd_fallback_sync",
    "discover_existing_providers",
    "ensure_available_model_or_fallback",
    "extract_param_billions",
    "fetch_all_candidates",
    "fetch_openrouter_candidates",
    "fetch_provider_candidates",
    "filter_and_rank_candidates",
    "format_fallback_label",
    "format_fallback_notice",
    "get_fallback_chain",
    "get_plugin_settings",
    "get_provider_status_list",
    "get_stored_fallback_rules",
    "handle_configure",
    "handle_providers",
    "handle_resolve",
    "handle_slash_command",
    "handle_switch",
    "is_fallback_heuristics_enabled",
    "is_flash_model",
    "is_heuristic_fallback_entry",
    "is_provider_available",
    "llm_execution_middleware",
    "llm_request_middleware",
    "load_hermes_config",
    "normalize_provider_name",
    "on_session_start_hook",
    "parse_context_length",
    "parse_fallback_criteria",
    "parse_param_size",
    "parse_provider_list",
    "parse_reasoning_effort",
    "pre_llm_call_hook",
    "register",
    "reresolve_fallback_reasoning_config",
    "resolve_best_available_model",
    "resolve_fallback_candidates",
    "resolve_fallback_entry_reasoning_config",
    "resolve_heuristic_entry",
    "resolve_per_model_reasoning_effort",
    "save_hermes_config",
    "scoped_fallback_chain",
    "setup_cli_parser",
    "switch_model_with_heuristic",
    "sync_fallback_chain_to_config",
    "update_plugin_settings",
]
