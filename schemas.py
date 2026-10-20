"""OpenAI tool schemas for the Model Fallback Heuristic plugin."""

from __future__ import annotations

from typing import Any

FALLBACK_HEURISTIC_RESOLVE_SCHEMA: dict[str, Any] = {
    "name": "fallback_heuristic_resolve",
    "description": (
        "Resolve and rank available candidate models across one or more providers (or all "
        "existing authenticated providers) using heuristic parameters such as parameter count, "
        "context window size, release recency, flash tier, provider preference, tool support, "
        "free/paid tier, vendor, model filter, and reasoning effort. Works without making an LLM call."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "heuristic": {
                "type": "string",
                "description": (
                    "Optional shorthand heuristic string (e.g. 'largest_parameter_count', "
                    "'greatest_context', 'latest_flash', 'smallest', 'provider_preference', "
                    "'provider:anthropic,openrouter min_context:128k')."
                ),
            },
            "sort_by": {
                "type": "string",
                "enum": [
                    "largest_parameter_count",
                    "greatest_context",
                    "latest",
                    "latest_flash",
                    "smallest",
                    "provider_preference",
                ],
                "description": "Primary ranking strategy.",
            },
            "provider": {
                "type": "string",
                "description": (
                    "Target provider ID (e.g. 'openrouter', 'anthropic', 'openai', 'gemini', "
                    "'deepseek', 'zai', 'kimi-coding', 'minimax', 'xai', 'nous', 'ollama') "
                    "or 'existing'/'auto' to search across all authenticated providers."
                ),
            },
            "providers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered preference list of providers to search across.",
            },
            "exclude_providers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Providers to exclude from candidate resolution.",
            },
            "exclude_models": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific model IDs to exclude (e.g. a currently unavailable model).",
            },
            "allow_provider_switch": {
                "type": "boolean",
                "description": "Whether switching away from the current provider is allowed.",
            },
            "prefer_current_provider": {
                "type": "boolean",
                "description": "Whether to prioritize candidates from the current provider first.",
            },
            "min_context": {
                "type": "integer",
                "description": "Minimum context window in tokens (e.g. 64000, 128000, 200000).",
            },
            "max_context": {
                "type": "integer",
                "description": "Maximum context window in tokens (0 = no upper limit).",
            },
            "min_params": {
                "type": "number",
                "description": "Minimum parameter size in billions (e.g. 30, 70).",
            },
            "max_params": {
                "type": "number",
                "description": "Maximum parameter size in billions (0 = no upper limit).",
            },
            "free_only": {
                "type": "boolean",
                "description": "If true, restrict to free ($0) models only.",
            },
            "require_tools": {
                "type": "boolean",
                "description": "If true, require tool/function calling support.",
            },
            "flash_only": {
                "type": "boolean",
                "description": "If true, restrict to fast/flash tier models.",
            },
            "vendor": {
                "type": "string",
                "description": "Filter by model vendor/creator (e.g. 'anthropic', 'google', 'qwen', 'deepseek').",
            },
            "name_filter": {
                "type": "string",
                "description": "Substring or glob pattern filter on model ID/name.",
            },
            "reasoning_effort": {
                "type": "string",
                "description": "Reasoning effort for resolved fallback entries (none, low, medium, high, default).",
            },
            "max_candidates": {
                "type": "integer",
                "description": "Maximum number of top candidates to return (default 5).",
            },
        },
        "required": [],
    },
}

FALLBACK_HEURISTIC_SWITCH_SCHEMA: dict[str, Any] = {
    "name": "fallback_heuristic_switch",
    "description": (
        "Switch the active model and/or provider (and sync fallback_providers in config.yaml) "
        "to the best available model from the user's existing authenticated providers using "
        "the heuristic criteria. Use this when the chosen model is unavailable or when the "
        "user wants to switch models/providers by capability heuristic."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "heuristic": {
                "type": "string",
                "description": "Optional heuristic shorthand (e.g. 'largest_parameter_count', 'greatest_context', 'latest_flash').",
            },
            "sort_by": {
                "type": "string",
                "enum": [
                    "largest_parameter_count",
                    "greatest_context",
                    "latest",
                    "latest_flash",
                    "smallest",
                    "provider_preference",
                ],
                "description": "Primary ranking strategy.",
            },
            "provider": {
                "type": "string",
                "description": "Target provider or 'existing'/'auto' to pick from authenticated providers.",
            },
            "providers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered preference list of providers to consider.",
            },
            "exclude_providers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Providers to exclude (e.g. a provider that is currently down).",
            },
            "exclude_models": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Models to exclude (e.g. a model that returned 404/unavailable).",
            },
            "exclude_current": {
                "type": "boolean",
                "description": "If true, exclude the currently configured primary model so a different model is selected.",
            },
            "allow_provider_switch": {
                "type": "boolean",
                "description": "If true (default), allow switching to another authenticated provider.",
            },
            "prefer_current_provider": {
                "type": "boolean",
                "description": "If true, prefer candidates on the current provider before switching providers.",
            },
            "min_context": {
                "type": "integer",
                "description": "Minimum context window in tokens.",
            },
            "min_params": {
                "type": "number",
                "description": "Minimum parameter size in billions.",
            },
            "free_only": {
                "type": "boolean",
                "description": "If true, only switch to free ($0) models.",
            },
            "vendor": {
                "type": "string",
                "description": "Optional vendor filter.",
            },
            "name_filter": {
                "type": "string",
                "description": "Optional model name filter.",
            },
            "reasoning_effort": {
                "type": "string",
                "description": "Optional reasoning effort override.",
            },
        },
        "required": [],
    },
}

FALLBACK_HEURISTIC_CONFIGURE_SCHEMA: dict[str, Any] = {
    "name": "fallback_heuristic_configure",
    "description": (
        "Inspect or update persistent model fallback heuristic settings and rules in config.yaml, "
        "and materialize the effective fallback chain across existing providers."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "set", "on", "off", "sync", "add_rule", "clear_rules", "setup"],
                "description": "Configuration action to perform (default: 'status'; use 'setup' to complete first-launch free-provider selection).",
            },
            "enabled": {
                "type": "boolean",
                "description": "Enable or disable heuristic fallback resolution.",
            },
            "sort_by": {
                "type": "string",
                "enum": [
                    "largest_parameter_count",
                    "greatest_context",
                    "latest",
                    "latest_flash",
                    "smallest",
                    "provider_preference",
                ],
                "description": "Default heuristic ranking strategy.",
            },
            "provider": {
                "type": "string",
                "description": "Default target provider ('auto'/'existing' or a specific provider ID).",
            },
            "providers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered preference list of providers for heuristic fallback.",
            },
            "exclude_providers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Providers to exclude from heuristic selection.",
            },
            "allow_provider_switch": {
                "type": "boolean",
                "description": "Allow switching providers during fallback resolution.",
            },
            "prefer_current_provider": {
                "type": "boolean",
                "description": "Prefer the active primary provider before switching to other providers.",
            },
            "free_only": {
                "type": "boolean",
                "description": "Restrict heuristic selection to free models only.",
            },
            "require_tools": {
                "type": "boolean",
                "description": "Require tool calling support.",
            },
            "flash_only": {
                "type": "boolean",
                "description": "Restrict to flash/fast models.",
            },
            "min_context": {
                "type": "integer",
                "description": "Minimum context window in tokens.",
            },
            "max_context": {
                "type": "integer",
                "description": "Maximum context window in tokens.",
            },
            "min_params": {
                "type": "number",
                "description": "Minimum parameter size in billions.",
            },
            "max_params": {
                "type": "number",
                "description": "Maximum parameter size in billions.",
            },
            "vendor": {
                "type": "string",
                "description": "Vendor filter.",
            },
            "name_filter": {
                "type": "string",
                "description": "Model name substring/glob filter.",
            },
            "reasoning_effort": {
                "type": "string",
                "description": "Reasoning effort for heuristic fallback models.",
            },
            "max_candidates": {
                "type": "integer",
                "description": "Number of fallback candidates materialized per rule.",
            },
            "auto_switch_primary_when_unavailable": {
                "type": "boolean",
                "description": "Automatically switch primary model when the selected primary is unavailable.",
            },
        },
        "required": [],
    },
}

FALLBACK_HEURISTIC_PROVIDERS_SCHEMA: dict[str, Any] = {
    "name": "fallback_heuristic_providers",
    "description": (
        "List all discovered model providers, indicating which providers are authenticated/configured "
        "('existing') and how many candidate models are available per provider."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "include_catalog_counts": {
                "type": "boolean",
                "description": "Include candidate model counts for each provider (default true).",
            },
            "only_authenticated": {
                "type": "boolean",
                "description": "If true, only return currently authenticated/configured providers.",
            },
            "check_free_providers": {
                "type": "boolean",
                "description": "If true (default), run a live check of all providers offering free ($0) models and include signup URLs, env vars, and sample free models.",
            },
        },
        "required": [],
    },
}

RESOLVE_SCHEMA = FALLBACK_HEURISTIC_RESOLVE_SCHEMA
SWITCH_SCHEMA = FALLBACK_HEURISTIC_SWITCH_SCHEMA
CONFIGURE_SCHEMA = FALLBACK_HEURISTIC_CONFIGURE_SCHEMA
PROVIDERS_SCHEMA = FALLBACK_HEURISTIC_PROVIDERS_SCHEMA

