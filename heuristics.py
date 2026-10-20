"""Multi-provider model fallback heuristics and dynamic criteria resolution.

Repackaged and extended from the Hermes Core model-fallback-heuristic contribution to work
as a standalone Hermes plugin across **any** model provider (OpenRouter, Nous Portal,
Anthropic, OpenAI, OpenAI Codex, Google Gemini, DeepSeek, Z.ai/GLM, Kimi/Moonshot,
MiniMax, NVIDIA NIM, xAI/Grok, HuggingFace, GitHub Copilot, AWS Bedrock, Google Vertex AI,
OpenCode Zen/Go, Vercel AI Gateway, Alibaba/DashScope, Xiaomi MiMo, KiloCode, Ollama,
LM Studio, vLLM, and custom providers).

Key capabilities:
1. Works with any model provider and discovers the user's **existing** (authenticated/configured)
   providers without requiring an LLM call.
2. Treats ``provider`` (and ``providers``, ``exclude_providers``, ``allow_provider_switch``,
   ``prefer_current_provider``) as first-class parameters in the heuristic logic alongside:
   - parameter size (``largest_parameter_count``, ``smallest``, ``min_params``, ``max_params``)
   - context window size (``greatest_context``, ``min_context``, ``max_context``)
   - release recency & version (``latest``, ``latest_flash``)
   - provider preference order (``provider_preference``)
   - tool-calling support (``require_tools``), pricing tier (``free_only``), flash tier (``flash_only``)
   - vendor/family (``vendor``), model name pattern (``name_filter``), and ``reasoning_effort``
3. Multi-tier metadata enrichment:
   - Live/cached provider catalogs (OpenRouter, Nous pricing, models.dev, provider_models_cache)
   - Built-in cross-provider model metadata for offline / zero-network operation
   - High-speed regex name heuristics (MoE, active params, context tokens, dates, qualitative tiers)
"""

from __future__ import annotations

import datetime
import fnmatch
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _get_hermes_home() -> Path:
    """Resolve the active Hermes home directory without requiring hermes_constants."""
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        env_home = os.environ.get("HERMES_HOME", "").strip()
        if env_home:
            return Path(env_home).expanduser()
        return Path.home() / ".hermes"


def _get_hermes_home_key() -> str:
    """Resolve a profile-scoped cache key for the active Hermes home."""
    try:
        from hermes_constants import hermes_home_key

        return str(hermes_home_key())
    except Exception:
        try:
            return str(_get_hermes_home().resolve())
        except Exception:
            return str(_get_hermes_home())


# ─── Provider Registry, Aliases & Discovery ─────────────────────────────────

PROVIDER_ALIASES: dict[str, str] = {
    "nous": "nous",
    "nous-portal": "nous",
    "nous_portal": "nous",
    "nousresearch": "nous",
    "openrouter": "openrouter",
    "or": "openrouter",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "gpt": "openai",
    "openai-codex": "openai-codex",
    "codex": "openai-codex",
    "chatgpt": "openai-codex",
    "gemini": "gemini",
    "google": "gemini",
    "google-gemini": "gemini",
    "ai-studio": "gemini",
    "aistudio": "gemini",
    "deepseek": "deepseek",
    "zai": "zai",
    "z-ai": "zai",
    "z_ai": "zai",
    "glm": "zai",
    "zhipu": "zai",
    "bigmodel": "zai",
    "kimi-coding": "kimi-coding",
    "kimi": "kimi-coding",
    "moonshot": "kimi-coding",
    "minimax": "minimax",
    "minimax-global": "minimax",
    "minimax-cn": "minimax-cn",
    "minimax_cn": "minimax-cn",
    "nvidia": "nvidia",
    "nim": "nvidia",
    "nvidia-nim": "nvidia",
    "xai": "xai",
    "grok": "xai",
    "huggingface": "huggingface",
    "hf": "huggingface",
    "copilot": "copilot",
    "github-copilot": "copilot",
    "gh-copilot": "copilot",
    "copilot-acp": "copilot-acp",
    "bedrock": "bedrock",
    "aws-bedrock": "bedrock",
    "aws": "bedrock",
    "vertex": "vertex",
    "google-vertex": "vertex",
    "vertex-ai": "vertex",
    "opencode-zen": "opencode-zen",
    "zen": "opencode-zen",
    "opencode-go": "opencode-go",
    "opencode_go": "opencode-go",
    "ai-gateway": "ai-gateway",
    "vercel": "ai-gateway",
    "aigw": "ai-gateway",
    "alibaba": "alibaba",
    "dashscope": "alibaba",
    "qwen": "alibaba",
    "aliyun": "alibaba",
    "xiaomi": "xiaomi",
    "mimo": "xiaomi",
    "kilocode": "kilocode",
    "kilo": "kilocode",
    "ollama": "ollama",
    "lmstudio": "lmstudio",
    "lm-studio": "lmstudio",
    "vllm": "vllm",
}

MULTI_PROVIDER_TOKENS: frozenset[str] = frozenset({"auto", "existing", "any", "all", "*"})

# Environment variables that authenticate each provider (checked via os.environ.get / get_secret)
PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "openrouter": ("OPENROUTER_API_KEY",),
    "nous": ("NOUS_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY",),
    "openai-codex": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "zai": ("ZAI_API_KEY", "GLM_API_KEY", "Z_AI_API_KEY"),
    "kimi-coding": ("KIMI_API_KEY", "MOONSHOT_API_KEY"),
    "minimax": ("MINIMAX_API_KEY",),
    "minimax-cn": ("MINIMAX_CN_API_KEY",),
    "nvidia": ("NVIDIA_API_KEY", "NGC_API_KEY"),
    "xai": ("XAI_API_KEY",),
    "huggingface": ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN"),
    "copilot": ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"),
    "bedrock": ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_PROFILE"),
    "vertex": ("VERTEX_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"),
    "opencode-zen": ("OPENCODE_ZEN_API_KEY", "OPENCODE_API_KEY"),
    "opencode-go": ("OPENCODE_GO_API_KEY", "OPENCODE_API_KEY"),
    "ai-gateway": ("AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_API_KEY"),
    "alibaba": ("DASHSCOPE_API_KEY", "ALIBABA_API_KEY"),
    "xiaomi": ("XIAOMI_API_KEY", "MIMO_API_KEY"),
    "kilocode": ("KILOCODE_API_KEY",),
    "ollama": ("OLLAMA_API_KEY", "OLLAMA_HOST"),
    "lmstudio": ("LMSTUDIO_API_KEY", "LMSTUDIO_BASE_URL"),
}

# Default provider priority when searching across existing providers without an explicit preference list
DEFAULT_PROVIDER_ORDER: tuple[str, ...] = (
    "anthropic",
    "openai-codex",
    "openai",
    "openrouter",
    "nous",
    "gemini",
    "deepseek",
    "zai",
    "kimi-coding",
    "minimax",
    "xai",
    "nvidia",
    "copilot",
    "bedrock",
    "vertex",
    "alibaba",
    "huggingface",
    "opencode-zen",
    "opencode-go",
    "ai-gateway",
    "minimax-cn",
    "xiaomi",
    "kilocode",
    "ollama",
    "lmstudio",
)

# Providers that are primarily paid unless free_only is explicitly requested
PAID_FIRST_PROVIDERS: frozenset[str] = frozenset(
    {
        "anthropic",
        "openai",
        "openai-codex",
        "gemini",
        "deepseek",
        "zai",
        "kimi-coding",
        "minimax",
        "minimax-cn",
        "xai",
        "nvidia",
        "copilot",
        "copilot-acp",
        "bedrock",
        "vertex",
        "alibaba",
        "huggingface",
        "opencode-zen",
        "opencode-go",
        "ai-gateway",
        "xiaomi",
        "kilocode",
        "ollama",
        "lmstudio",
        "vllm",
    }
)


def normalize_provider_name(provider: str | None) -> str:
    """Normalize a provider name or alias to its canonical Hermes provider ID."""
    raw = str(provider or "").strip().lower()
    if not raw:
        return ""
    if raw.startswith("custom:"):
        return raw
    return PROVIDER_ALIASES.get(raw, raw)


def parse_provider_list(raw: Any) -> tuple[str, ...]:
    """Parse a string or sequence of provider names into normalized unique provider IDs."""
    if not raw:
        return ()
    items: list[str] = []
    if isinstance(raw, str):
        for token in re.split(r"[,|;\s]+", raw.strip()):
            tok = token.strip()
            if tok:
                items.append(tok)
    elif isinstance(raw, (list, tuple, set, frozenset)):
        for elem in raw:
            if isinstance(elem, str) and elem.strip():
                for sub in re.split(r"[,|;\s]+", elem.strip()):
                    if sub.strip():
                        items.append(sub.strip())
    result: list[str] = []
    for item in items:
        norm = normalize_provider_name(item)
        if norm and norm not in MULTI_PROVIDER_TOKENS and norm not in result:
            result.append(norm)
    return tuple(result)


def _read_secret_or_env(env_key: str) -> str:
    """Read an environment variable via Hermes secret_scope if available, else os.environ.get."""
    try:
        from agent.secret_scope import get_secret

        val = get_secret(env_key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        pass
    return (os.environ.get(env_key) or "").strip()


def _load_active_config(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Load the user's Hermes config dict without making any network/model calls."""
    if isinstance(config, dict):
        return config
    try:
        from hermes_cli.config import load_config

        loaded = load_config()
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    try:
        cfg_path = _get_hermes_home() / "config.yaml"
        if cfg_path.is_file():
            try:
                import hermes_yaml as yaml
            except ImportError:
                import yaml  # type: ignore[no-redef]
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
            if isinstance(raw, dict):
                return raw
    except Exception:
        pass
    return {}


def discover_existing_providers(
    config: Optional[dict[str, Any]] = None,
    include_unauthenticated_configured: bool = True,
) -> list[str]:
    """Discover all existing (authenticated or configured) model providers on this system.

    Works completely without a model call by inspecting:
    1. Hermes's ``list_authenticated_providers`` / ``get_auth_status`` (OAuth, credential pool, auth store)
    2. Environment variables for provider API keys (``OPENROUTER_API_KEY``, ``ANTHROPIC_API_KEY``, etc.)
    3. ``config.yaml`` (``model.provider``, ``providers``, ``custom_providers``, ``fallback_providers``)
    """
    cfg = _load_active_config(config)
    discovered: list[str] = []

    def _add(pid: str | None) -> None:
        norm = normalize_provider_name(pid)
        if norm and norm not in MULTI_PROVIDER_TOKENS and norm not in discovered:
            discovered.append(norm)

    # 1. Primary configured provider in config.yaml (if authenticated or configured)
    model_cfg = cfg.get("model")
    primary_provider = ""
    if isinstance(model_cfg, dict):
        primary_provider = normalize_provider_name(model_cfg.get("provider"))

    # 2. Query Hermes Core's authenticated provider discovery when available
    try:
        from hermes_cli.model_switch_providers import list_authenticated_providers

        for entry in list_authenticated_providers(cfg) or []:
            if isinstance(entry, dict) and entry.get("id"):
                _add(str(entry["id"]))
    except Exception as exc:
        logger.debug("list_authenticated_providers probe skipped: %s", exc)

    # 3. Check Hermes auth status for OAuth / portal providers (nous, openai-codex, copilot, anthropic)
    try:
        from hermes_cli.auth import get_auth_status

        for pid in ("nous", "openai-codex", "copilot", "anthropic"):
            try:
                st = get_auth_status(pid)
                if isinstance(st, dict) and st.get("logged_in"):
                    _add(pid)
            except Exception:
                pass
    except Exception:
        pass

    # 4. Check environment variables for provider API keys
    for pid, env_keys in PROVIDER_ENV_VARS.items():
        if any(_read_secret_or_env(k) for k in env_keys):
            _add(pid)

    # 5. Check config.yaml providers / custom_providers / fallback_providers blocks
    if include_unauthenticated_configured:
        if primary_provider:
            # Include primary provider if it has credentials or is a local/custom provider
            if (
                primary_provider in discovered
                or primary_provider in ("ollama", "lmstudio", "vllm")
                or primary_provider.startswith("custom:")
            ):
                _add(primary_provider)

        providers_block = cfg.get("providers")
        if isinstance(providers_block, dict):
            for pname, pconf in providers_block.items():
                if isinstance(pconf, dict):
                    has_cred = bool(
                        pconf.get("api_key")
                        or pconf.get("base_url")
                        or (pconf.get("key_env") and _read_secret_or_env(str(pconf["key_env"])))
                        or (pconf.get("api_key_env") and _read_secret_or_env(str(pconf["api_key_env"])))
                    )
                    if has_cred or pconf.get("enabled") is True:
                        _add(pname)

        custom_providers = cfg.get("custom_providers")
        if isinstance(custom_providers, list):
            for citem in custom_providers:
                if isinstance(citem, dict) and citem.get("name"):
                    cname = str(citem["name"]).strip()
                    if cname:
                        _add(f"custom:{cname}" if not cname.startswith("custom:") else cname)

        for fb_key in ("fallback_providers", "fallback_model"):
            fb_list = cfg.get(fb_key)
            if isinstance(fb_list, list):
                for fb_entry in fb_list:
                    if isinstance(fb_entry, dict):
                        fp = normalize_provider_name(
                            fb_entry.get("provider") or fb_entry.get("model_provider")
                        )
                        if fp and (
                            fb_entry.get("api_key")
                            or (fb_entry.get("key_env") and _read_secret_or_env(str(fb_entry["key_env"])))
                            or fp in ("ollama", "lmstudio", "vllm")
                        ):
                            _add(fp)

    # Place primary_provider at the front if it is in discovered
    if primary_provider and primary_provider in discovered:
        discovered.remove(primary_provider)
        discovered.insert(0, primary_provider)

    return discovered


def is_provider_available(
    provider: str,
    config: Optional[dict[str, Any]] = None,
) -> bool:
    """Return True if *provider* has credentials or configuration available without a model call."""
    norm = normalize_provider_name(provider)
    if not norm or norm in MULTI_PROVIDER_TOKENS:
        return bool(discover_existing_providers(config=config))
    if norm in ("ollama", "lmstudio", "vllm"):
        return True
    existing = discover_existing_providers(config=config)
    if norm in existing:
        return True
    # Also check custom:<name> match
    if f"custom:{norm}" in existing:
        return True
    return False


def get_provider_status_list(
    config: Optional[dict[str, Any]] = None,
    include_catalog_counts: bool = False,
) -> list[dict[str, Any]]:
    """Return structured status for all known and configured providers without a model call."""
    cfg = _load_active_config(config)
    existing = set(discover_existing_providers(config=cfg))
    primary_provider = ""
    if isinstance(cfg.get("model"), dict):
        primary_provider = normalize_provider_name(cfg["model"].get("provider"))

    all_providers: list[str] = []
    for p in (*existing, *DEFAULT_PROVIDER_ORDER):
        norm = normalize_provider_name(p)
        if norm and norm not in all_providers:
            all_providers.append(norm)

    results: list[dict[str, Any]] = []
    for pid in all_providers:
        env_keys = PROVIDER_ENV_VARS.get(pid, ())
        active_env = [k for k in env_keys if _read_secret_or_env(k)]
        is_auth = pid in existing
        info: dict[str, Any] = {
            "provider": pid,
            "authenticated": is_auth,
            "available": is_auth,
            "is_primary": pid == primary_provider,
            "env_vars": list(env_keys),
            "env_keys_configured": active_env,
            "free_tier_default": pid not in PAID_FIRST_PROVIDERS,
        }
        if include_catalog_counts:
            cands = get_candidate_models(provider=pid)
            info["candidate_count"] = len(cands)
            info["free_candidate_count"] = sum(1 for c in cands if c.is_free)
        results.append(info)
    return results


# ─── Regex & Name Heuristics ────────────────────────────────────────────────

_RE_MOE_PARAM = re.compile(r"(?:^|[-_/ ])(\d+)x(\d+(?:\.\d+)?)[bB](?:[-_/ :]|$)", re.IGNORECASE)
_RE_MOE_ACTIVE_PARAM = re.compile(r"(?:^|[-_/ ])(\d+(?:\.\d+)?)[bB]-a\d+[bB](?:[-_/ :]|$)", re.IGNORECASE)
_RE_B_PARAM = re.compile(r"(?:^|[-_/ ])(\d+(?:\.\d+)?)[bB](?:[-_/ :]|$)", re.IGNORECASE)
_RE_CTX_MILLION = re.compile(r"(?:^|[-_/ ])(\d+(?:\.\d+)?)[mM](?:[-_/ :]|$)", re.IGNORECASE)
_RE_CTX_THOUSAND = re.compile(r"(?:^|[-_/ ])(\d+)[kK](?:[-_/ :]|$)", re.IGNORECASE)
_RE_VERSION_MULTI = re.compile(r"(?:^|[-_/v])(\d+(?:\.\d+)+)(?:[-_/:]|$)")
_RE_VERSION_SINGLE = re.compile(r"(?:^|[-_/v])(\d+)(?:[-_/:]|$)")
_RE_DATE_FULL = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")
_RE_DATE_MONTH_DAY = re.compile(r"(?:^|[-_])(0[1-9]|1[0-2])([0-3]\d)(?:[-_:]|$)")
_RE_SHORTHAND_SPLIT = re.compile(r"[:_,\s-]+")
_RE_QUALITATIVE_TIERS = re.compile(
    r"\b(nano|xs|micro|mini|haiku|small|lite|medium|plus|sonnet|pro|large|max|opus|ultra|super)\b|(-s-)",
    re.IGNORECASE,
)

# Known qualitative size tiers mapped to estimated parameter size in Billions
_QUALITATIVE_TIER_DICT: dict[str, float] = {
    "nano": 0.5,
    "xs": 1.0,
    "micro": 1.0,
    "mini": 3.0,
    "haiku": 7.0,
    "small": 7.0,
    "-s-": 7.0,
    "lite": 8.0,
    "medium": 14.0,
    "plus": 50.0,
    "sonnet": 70.0,
    "pro": 70.0,
    "large": 70.0,
    "max": 100.0,
    "opus": 100.0,
    "ultra": 200.0,
    "super": 300.0,
}
_QUALITATIVE_SIZE_TIERS: list[tuple[str, float]] = list(_QUALITATIVE_TIER_DICT.items())

_DESCRIBE_SORT_BY: dict[str, str] = {
    "largest_parameter_count": "largest parameter count",
    "largest": "largest parameter count",
    "params": "largest parameter count",
    "greatest_context": "greatest context",
    "largest_context": "greatest context",
    "context": "greatest context",
    "smallest": "smallest model",
    "latest": "latest model",
    "recency": "latest model",
    "latest_flash": "latest flash model",
    "provider_preference": "provider preference",
    "provider": "provider preference",
}


# ─── Built-in Cross-Provider Catalog & Metadata ────────────────────────────

# Known model metadata for accurate offline enrichment across all providers:
# (param_size_b, context_window, supports_tools, is_flash, is_free, release_date, vendor)
_KNOWN_MODEL_SPECS: dict[str, dict[str, Any]] = {
    # Anthropic
    "claude-opus-4-6": {"param_size_b": 150.0, "context_window": 1000000, "supports_tools": True, "is_flash": False, "release_date": "2026-02-01", "vendor": "anthropic"},
    "claude-sonnet-4-6": {"param_size_b": 70.0, "context_window": 1000000, "supports_tools": True, "is_flash": False, "release_date": "2026-02-15", "vendor": "anthropic"},
    "claude-opus-4-5-20251101": {"param_size_b": 140.0, "context_window": 200000, "supports_tools": True, "is_flash": False, "release_date": "2025-11-01", "vendor": "anthropic"},
    "claude-sonnet-4-5-20250929": {"param_size_b": 70.0, "context_window": 200000, "supports_tools": True, "is_flash": False, "release_date": "2025-09-29", "vendor": "anthropic"},
    "claude-haiku-4-5-20251001": {"param_size_b": 12.0, "context_window": 200000, "supports_tools": True, "is_flash": True, "release_date": "2025-10-01", "vendor": "anthropic"},
    "claude-sonnet-4-20250514": {"param_size_b": 70.0, "context_window": 200000, "supports_tools": True, "is_flash": False, "release_date": "2025-05-14", "vendor": "anthropic"},
    "claude-opus-4-20250514": {"param_size_b": 130.0, "context_window": 200000, "supports_tools": True, "is_flash": False, "release_date": "2025-05-14", "vendor": "anthropic"},
    "claude-3-7-sonnet-20250219": {"param_size_b": 70.0, "context_window": 200000, "supports_tools": True, "is_flash": False, "release_date": "2025-02-19", "vendor": "anthropic"},
    # OpenAI / Codex
    "gpt-5.4": {"param_size_b": 200.0, "context_window": 1048576, "supports_tools": True, "is_flash": False, "release_date": "2026-03-01", "vendor": "openai"},
    "gpt-5.4-pro": {"param_size_b": 250.0, "context_window": 1048576, "supports_tools": True, "is_flash": False, "release_date": "2026-03-01", "vendor": "openai"},
    "gpt-5.4-mini": {"param_size_b": 20.0, "context_window": 400000, "supports_tools": True, "is_flash": True, "release_date": "2026-03-01", "vendor": "openai"},
    "gpt-5.4-nano": {"param_size_b": 4.0, "context_window": 400000, "supports_tools": True, "is_flash": True, "release_date": "2026-03-01", "vendor": "openai"},
    "gpt-5.3-codex": {"param_size_b": 180.0, "context_window": 400000, "supports_tools": True, "is_flash": False, "release_date": "2026-02-01", "vendor": "openai"},
    "gpt-5.3-codex-spark": {"param_size_b": 35.0, "context_window": 400000, "supports_tools": True, "is_flash": True, "release_date": "2026-02-10", "vendor": "openai"},
    "gpt-5.2-codex": {"param_size_b": 160.0, "context_window": 400000, "supports_tools": True, "is_flash": False, "release_date": "2026-01-15", "vendor": "openai"},
    "gpt-5.2": {"param_size_b": 160.0, "context_window": 400000, "supports_tools": True, "is_flash": False, "release_date": "2026-01-10", "vendor": "openai"},
    "gpt-5.1-codex-max": {"param_size_b": 175.0, "context_window": 400000, "supports_tools": True, "is_flash": False, "release_date": "2025-12-01", "vendor": "openai"},
    "gpt-5.1-codex-mini": {"param_size_b": 18.0, "context_window": 256000, "supports_tools": True, "is_flash": True, "release_date": "2025-12-01", "vendor": "openai"},
    "gpt-4.1": {"param_size_b": 120.0, "context_window": 1047576, "supports_tools": True, "is_flash": False, "release_date": "2025-04-14", "vendor": "openai"},
    "gpt-4.1-mini": {"param_size_b": 16.0, "context_window": 1047576, "supports_tools": True, "is_flash": True, "release_date": "2025-04-14", "vendor": "openai"},
    "gpt-4.1-nano": {"param_size_b": 3.0, "context_window": 1047576, "supports_tools": True, "is_flash": True, "release_date": "2025-04-14", "vendor": "openai"},
    "gpt-4o": {"param_size_b": 100.0, "context_window": 128000, "supports_tools": True, "is_flash": False, "release_date": "2024-11-20", "vendor": "openai"},
    "gpt-4o-mini": {"param_size_b": 8.0, "context_window": 128000, "supports_tools": True, "is_flash": True, "release_date": "2024-07-18", "vendor": "openai"},
    "o3": {"param_size_b": 150.0, "context_window": 200000, "supports_tools": True, "is_flash": False, "release_date": "2025-04-16", "vendor": "openai"},
    "o4-mini": {"param_size_b": 24.0, "context_window": 200000, "supports_tools": True, "is_flash": True, "release_date": "2025-04-16", "vendor": "openai"},
    # Google Gemini
    "gemini-3.1-pro-preview": {"param_size_b": 140.0, "context_window": 2097152, "supports_tools": True, "is_flash": False, "release_date": "2026-02-20", "vendor": "google"},
    "gemini-3-pro-preview": {"param_size_b": 130.0, "context_window": 1048576, "supports_tools": True, "is_flash": False, "release_date": "2025-12-15", "vendor": "google"},
    "gemini-3-flash-preview": {"param_size_b": 24.0, "context_window": 1048576, "supports_tools": True, "is_flash": True, "release_date": "2025-12-15", "vendor": "google"},
    "gemini-3.1-flash-lite-preview": {"param_size_b": 8.0, "context_window": 1048576, "supports_tools": True, "is_flash": True, "release_date": "2026-02-20", "vendor": "google"},
    "gemini-2.5-pro": {"param_size_b": 110.0, "context_window": 1048576, "supports_tools": True, "is_flash": False, "release_date": "2025-06-01", "vendor": "google"},
    "gemini-2.5-flash": {"param_size_b": 20.0, "context_window": 1048576, "supports_tools": True, "is_flash": True, "release_date": "2025-06-01", "vendor": "google"},
    "gemini-2.5-flash-lite": {"param_size_b": 7.0, "context_window": 1048576, "supports_tools": True, "is_flash": True, "release_date": "2025-06-15", "vendor": "google"},
    "gemini-2.0-flash": {"param_size_b": 15.0, "context_window": 1048576, "supports_tools": True, "is_flash": True, "release_date": "2025-02-05", "vendor": "google"},
    # DeepSeek
    "deepseek-chat": {"param_size_b": 671.0, "context_window": 128000, "supports_tools": True, "is_flash": False, "release_date": "2025-03-24", "vendor": "deepseek"},
    "deepseek-reasoner": {"param_size_b": 671.0, "context_window": 128000, "supports_tools": True, "is_flash": False, "release_date": "2025-05-28", "vendor": "deepseek"},
    "deepseek-v3.2": {"param_size_b": 671.0, "context_window": 163840, "supports_tools": True, "is_flash": False, "release_date": "2025-12-01", "vendor": "deepseek"},
    "deepseek-r1": {"param_size_b": 671.0, "context_window": 128000, "supports_tools": True, "is_flash": False, "release_date": "2025-05-28", "vendor": "deepseek"},
    # Z.ai / GLM
    "glm-5": {"param_size_b": 350.0, "context_window": 200000, "supports_tools": True, "is_flash": False, "release_date": "2026-01-20", "vendor": "z-ai"},
    "glm-5-turbo": {"param_size_b": 120.0, "context_window": 200000, "supports_tools": True, "is_flash": True, "release_date": "2026-02-01", "vendor": "z-ai"},
    "glm-4.7": {"param_size_b": 200.0, "context_window": 200000, "supports_tools": True, "is_flash": False, "release_date": "2025-11-10", "vendor": "z-ai"},
    "glm-4.7-flash": {"param_size_b": 18.0, "context_window": 200000, "supports_tools": True, "is_flash": True, "is_free": True, "release_date": "2025-11-15", "vendor": "z-ai"},
    "glm-4.5": {"param_size_b": 355.0, "context_window": 128000, "supports_tools": True, "is_flash": False, "release_date": "2025-07-28", "vendor": "z-ai"},
    "glm-4.5-air": {"param_size_b": 106.0, "context_window": 128000, "supports_tools": True, "is_flash": True, "release_date": "2025-07-28", "vendor": "z-ai"},
    "glm-4.5-flash": {"param_size_b": 12.0, "context_window": 128000, "supports_tools": True, "is_flash": True, "is_free": True, "release_date": "2025-08-01", "vendor": "z-ai"},
    # Kimi / Moonshot
    "kimi-k2.5": {"param_size_b": 1000.0, "context_window": 262144, "supports_tools": True, "is_flash": False, "release_date": "2026-01-25", "vendor": "moonshotai"},
    "kimi-k2-thinking": {"param_size_b": 1000.0, "context_window": 262144, "supports_tools": True, "is_flash": False, "release_date": "2025-11-06", "vendor": "moonshotai"},
    "kimi-k2-thinking-turbo": {"param_size_b": 1000.0, "context_window": 262144, "supports_tools": True, "is_flash": True, "release_date": "2025-11-10", "vendor": "moonshotai"},
    "kimi-k2-turbo-preview": {"param_size_b": 1000.0, "context_window": 262144, "supports_tools": True, "is_flash": True, "release_date": "2025-09-05", "vendor": "moonshotai"},
    "kimi-k2-0905-preview": {"param_size_b": 1000.0, "context_window": 262144, "supports_tools": True, "is_flash": False, "release_date": "2025-09-05", "vendor": "moonshotai"},
    # MiniMax
    "minimax-m2.7": {"param_size_b": 456.0, "context_window": 1000000, "supports_tools": True, "is_flash": False, "release_date": "2026-02-10", "vendor": "minimax"},
    "minimax-m2.7-highspeed": {"param_size_b": 456.0, "context_window": 1000000, "supports_tools": True, "is_flash": True, "release_date": "2026-02-10", "vendor": "minimax"},
    "minimax-m2.5": {"param_size_b": 456.0, "context_window": 1000000, "supports_tools": True, "is_flash": False, "release_date": "2025-12-01", "vendor": "minimax"},
    "minimax-m2.5-highspeed": {"param_size_b": 456.0, "context_window": 1000000, "supports_tools": True, "is_flash": True, "release_date": "2025-12-01", "vendor": "minimax"},
    "minimax-m2.1": {"param_size_b": 456.0, "context_window": 1000000, "supports_tools": True, "is_flash": False, "release_date": "2025-10-01", "vendor": "minimax"},
    # xAI / Grok
    "grok-4": {"param_size_b": 300.0, "context_window": 256000, "supports_tools": True, "is_flash": False, "release_date": "2025-07-09", "vendor": "xai"},
    "grok-4-fast": {"param_size_b": 70.0, "context_window": 2000000, "supports_tools": True, "is_flash": True, "release_date": "2025-09-19", "vendor": "xai"},
    "grok-code-fast-1": {"param_size_b": 40.0, "context_window": 256000, "supports_tools": True, "is_flash": True, "release_date": "2025-08-26", "vendor": "xai"},
    "grok-3": {"param_size_b": 200.0, "context_window": 131072, "supports_tools": True, "is_flash": False, "release_date": "2025-02-17", "vendor": "xai"},
    "grok-3-mini": {"param_size_b": 20.0, "context_window": 131072, "supports_tools": True, "is_flash": True, "release_date": "2025-02-17", "vendor": "xai"},
    # Alibaba / Qwen
    "qwen3-max": {"param_size_b": 400.0, "context_window": 262144, "supports_tools": True, "is_flash": False, "release_date": "2025-09-24", "vendor": "qwen"},
    "qwen3-coder-plus": {"param_size_b": 480.0, "context_window": 1000000, "supports_tools": True, "is_flash": False, "release_date": "2025-07-23", "vendor": "qwen"},
    "qwen3-coder-flash": {"param_size_b": 30.0, "context_window": 262144, "supports_tools": True, "is_flash": True, "release_date": "2025-08-01", "vendor": "qwen"},
    "qwen3.5-plus": {"param_size_b": 235.0, "context_window": 1000000, "supports_tools": True, "is_flash": False, "release_date": "2026-01-15", "vendor": "qwen"},
    "qwen3-235b-a22b": {"param_size_b": 235.0, "context_window": 131072, "supports_tools": True, "is_flash": False, "release_date": "2025-04-29", "vendor": "qwen"},
    # Xiaomi MiMo
    "mimo-v2-pro": {"param_size_b": 72.0, "context_window": 131072, "supports_tools": True, "is_flash": False, "release_date": "2026-01-10", "vendor": "xiaomi"},
    "mimo-v2-flash": {"param_size_b": 7.0, "context_window": 131072, "supports_tools": True, "is_flash": True, "release_date": "2026-01-10", "vendor": "xiaomi"},
}

_BUILTIN_PROVIDER_CATALOGS: dict[str, list[str]] = {
    "anthropic": [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
    ],
    "openai": [
        "gpt-5.4",
        "gpt-5.4-pro",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.3-codex",
        "gpt-5.2",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
        "o3",
        "o4-mini",
    ],
    "openai-codex": [
        "gpt-5.4",
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
        "gpt-5.2-codex",
        "gpt-5.2",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
    ],
    "gemini": [
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "zai": [
        "glm-5",
        "glm-5-turbo",
        "glm-4.7",
        "glm-4.7-flash",
        "glm-4.5",
        "glm-4.5-air",
        "glm-4.5-flash",
    ],
    "kimi-coding": [
        "kimi-k2.5",
        "kimi-k2-thinking",
        "kimi-k2-thinking-turbo",
        "kimi-k2-turbo-preview",
        "kimi-k2-0905-preview",
    ],
    "minimax": [
        "MiniMax-M2.7",
        "MiniMax-M2.7-highspeed",
        "MiniMax-M2.5",
        "MiniMax-M2.5-highspeed",
        "MiniMax-M2.1",
    ],
    "minimax-cn": [
        "MiniMax-M2.7",
        "MiniMax-M2.7-highspeed",
        "MiniMax-M2.5",
        "MiniMax-M2.5-highspeed",
        "MiniMax-M2.1",
    ],
    "nvidia": [
        "nvidia/llama-3.3-nemotron-super-49b-v1",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "deepseek-ai/deepseek-r1",
        "meta/llama-3.3-70b-instruct",
        "qwen/qwen2.5-coder-32b-instruct",
    ],
    "xai": [
        "grok-4",
        "grok-4-fast",
        "grok-code-fast-1",
        "grok-3",
        "grok-3-mini",
    ],
    "huggingface": [
        "Qwen/Qwen3-235B-A22B",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3",
        "meta-llama/Llama-3.3-70B-Instruct",
    ],
    "copilot": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex",
        "claude-opus-4.6",
        "claude-sonnet-4.6",
        "claude-sonnet-4.5",
        "claude-haiku-4.5",
        "gemini-2.5-pro",
        "gpt-4.1",
        "gpt-4o",
    ],
    "bedrock": [
        "us.anthropic.claude-opus-4-6-v1",
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.meta.llama3-3-70b-instruct-v1:0",
    ],
    "vertex": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    ],
    "alibaba": [
        "qwen3-max",
        "qwen3.5-plus",
        "qwen3-coder-plus",
        "qwen3-coder-flash",
        "qwen3-235b-a22b",
    ],
    "opencode-zen": [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "gpt-5.4",
        "gemini-3-pro",
        "glm-5",
        "kimi-k2.5",
        "minimax-m2.5",
    ],
    "opencode-go": [
        "glm-5",
        "kimi-k2.5",
        "minimax-m2.7",
        "minimax-m2.5",
    ],
    "ai-gateway": [
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
        "openai/gpt-5.4-mini",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
        "deepseek/deepseek-v3.2",
    ],
    "xiaomi": [
        "mimo-v2-pro",
        "mimo-v2-flash",
    ],
    "kilocode": [
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "google/gemini-3.1-pro-preview",
        "google/gemini-2.5-flash",
        "openai/gpt-5.4",
    ],
    "ollama": [
        "qwen2.5-coder:32b",
        "llama3.3:70b",
        "deepseek-r1:32b",
        "qwen2.5:14b",
        "llama3.2:8b",
    ],
    "lmstudio": [
        "qwen2.5-coder-32b-instruct",
        "llama-3.3-70b-instruct",
        "deepseek-r1-distill-qwen-32b",
        "qwen2.5-7b-instruct",
    ],
    "nous": [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "qwen/qwen3-235b-a22b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
        "qwen/qwen-2.5-coder-32b-instruct:free",
        "stepfun/step-3.7-flash:free",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
    ],
}

_BUILTIN_OPENROUTER_STATIC: list[tuple[str, str]] = [
    ("nvidia/nemotron-3-super-120b-a12b:free", "free"),
    ("qwen/qwen3-235b-a22b:free", "free"),
    ("deepseek/deepseek-chat-v3-0324:free", "free"),
    ("deepseek/deepseek-r1:free", "free"),
    ("meta-llama/llama-3.3-70b-instruct:free", "free"),
    ("google/gemini-2.0-flash-exp:free", "free"),
    ("qwen/qwen-2.5-coder-32b-instruct:free", "free"),
    ("qwen/qwen-2.5-72b-instruct:free", "free"),
    ("mistralai/mistral-small-3.1-24b-instruct:free", "free"),
    ("anthropic/claude-opus-4.6", "paid"),
    ("anthropic/claude-sonnet-4.6", "paid"),
    ("openai/gpt-5.4", "paid"),
    ("google/gemini-2.5-pro", "paid"),
    ("google/gemini-2.5-flash", "paid"),
    ("deepseek/deepseek-chat", "paid"),
]


# ─── Data Structures ────────────────────────────────────────────────────────


@dataclass
class FallbackCriteria:
    """Parsed criteria for dynamic fallback model resolution across one or more providers."""

    sort_by: str = "largest_parameter_count"
    provider: str = "nous"
    providers: tuple[str, ...] = ()
    exclude_providers: tuple[str, ...] = ()
    exclude_models: tuple[str, ...] = ()
    allow_provider_switch: bool = True
    prefer_current_provider: bool = False
    current_provider: Optional[str] = None
    free_only: bool = True
    require_tools: bool = True
    flash_only: bool = False
    name_filter: str = ""
    vendor: str = ""
    min_context: int = 0
    max_context: int = 0
    min_params: float = 0.0
    max_params: float = 0.0
    reasoning_effort: Optional[str] = None
    max_candidates: int = 3
    raw_heuristic: str = ""

    @property
    def model_provider(self) -> str:
        """Alias for provider."""
        return self.provider

    @model_provider.setter
    def model_provider(self, val: str) -> None:
        self.provider = val

    def describe(self) -> str:
        """User-friendly summary of the criteria."""
        parts = []
        if self.providers:
            parts.append(f"providers:{','.join(self.providers)}")
        elif self.provider and self.provider != "nous":
            parts.append(f"provider:{self.provider}")
        if self.exclude_providers:
            parts.append(f"exclude_providers:{','.join(self.exclude_providers)}")
        if not self.allow_provider_switch:
            parts.append("no_provider_switch")
        if self.flash_only:
            parts.append("flash")
        if self.vendor:
            parts.append(f"vendor:{self.vendor}")
        if self.name_filter:
            parts.append(f"filter:{self.name_filter}")
        if self.reasoning_effort and self.reasoning_effort != "default":
            parts.append(f"reasoning:{self.reasoning_effort}")

        sort_label = _DESCRIBE_SORT_BY.get(self.sort_by, self.sort_by)
        parts.append(sort_label)

        if self.free_only:
            parts.append("free")
        if self.min_context:
            parts.append(f"min_ctx:{self.min_context}")
        if self.max_context:
            parts.append(f"max_ctx:{self.max_context}")
        if self.min_params:
            parts.append(f"min_params:{self.min_params:g}b")
        if self.max_params:
            parts.append(f"max_params:{self.max_params:g}b")
        return " ".join(parts) or "default"

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict representation."""
        data = asdict(self)
        data["providers"] = list(self.providers)
        data["exclude_providers"] = list(self.exclude_providers)
        data["exclude_models"] = list(self.exclude_models)
        return data


@dataclass
class ModelCandidateMetadata:
    """Enriched metadata for a model candidate evaluated against fallback criteria."""

    id: str
    name: str = ""
    provider: str = ""
    is_free: bool = False
    context_window: int = 0
    param_size_b: Optional[float] = None
    release_date: str = ""
    created_ts: float = 0.0
    date_snapshot: float = 0.0
    version_tuple: tuple[int, ...] = (0,)
    supports_tools: bool = False
    is_flash: bool = False
    vendor: str = ""
    source: str = "heuristic"

    def __post_init__(self):
        if not self.name:
            self.name = self.id


# ─── Regex Extractors ───────────────────────────────────────────────────────


def is_free_model_slug(model_id: str) -> bool:
    """Detect if model identifier indicates a free tier model."""
    lower = model_id.lower()
    return (
        lower.endswith(":free")
        or "/free" in lower
        or "-free" in lower
        or ":free" in lower
        or lower.startswith("free-")
        or "free/" in lower
        or "stealth/union-alpha" in lower
    )


def is_flash_model(model_id: str, name: str = "") -> bool:
    """Detect if model is in the 'flash' / high-speed tier."""
    lower = f"{model_id} {name}".lower()
    return (
        "flash" in lower
        or "haiku" in lower
        or "speed" in lower
        or "turbo" in lower
        or "spark" in lower
    )


def extract_vendor(model_id: str) -> str:
    """Extract vendor/creator prefix from a model slug like 'google/gemini-2.5-flash'."""
    if "/" in model_id:
        return model_id.split("/", 1)[0].lower().strip()
    lower = model_id.lower()
    for prefix, vendor in (
        ("claude-", "anthropic"),
        ("gpt-", "openai"),
        ("o1-", "openai"),
        ("o3", "openai"),
        ("o4-", "openai"),
        ("gemini-", "google"),
        ("deepseek-", "deepseek"),
        ("glm-", "z-ai"),
        ("kimi-", "moonshotai"),
        ("minimax-", "minimax"),
        ("grok-", "xai"),
        ("qwen", "qwen"),
        ("llama", "meta-llama"),
        ("mistral", "mistralai"),
        ("mixtral", "mistralai"),
        ("nemotron", "nvidia"),
        ("mimo-", "xiaomi"),
    ):
        if lower.startswith(prefix):
            return vendor
    return ""


def extract_param_size_b(model_id: str, name: str = "") -> Optional[float]:
    """Extract parameter size in billions from model slug or display name."""
    text = f" {model_id} {name} "

    m = _RE_MOE_PARAM.search(text)
    if m:
        try:
            experts = int(m.group(1))
            size_per_expert = float(m.group(2))
            return round(experts * size_per_expert, 1)
        except (ValueError, TypeError):
            pass

    m = _RE_MOE_ACTIVE_PARAM.search(text)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass

    m = _RE_B_PARAM.search(text)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass

    m = _RE_QUALITATIVE_TIERS.search(text)
    if m:
        matched_term = (m.group(1) or m.group(2) or "").lower()
        if matched_term in _QUALITATIVE_TIER_DICT:
            return _QUALITATIVE_TIER_DICT[matched_term]

    return None


_K_POWER_OF_TWO: dict[int, int] = {
    4: 4096,
    8: 8192,
    16: 16384,
    32: 32768,
    64: 65536,
    128: 131072,
    256: 262144,
    512: 524288,
}


def extract_context_window_from_name(model_id: str, default: int = 0) -> int:
    """Extract context window in tokens from model slug."""
    text = f" {model_id} "
    m = _RE_CTX_MILLION.search(text)
    if m:
        try:
            val = float(m.group(1))
            if 0.1 <= val <= 32.0:
                return int(val * 1_000_000)
        except (ValueError, TypeError):
            pass

    m = _RE_CTX_THOUSAND.search(text)
    if m:
        try:
            val = int(m.group(1))
            if val in _K_POWER_OF_TWO:
                return _K_POWER_OF_TWO[val]
            if 4 <= val <= 2048:
                return val * 1000
        except (ValueError, TypeError):
            pass

    return default


def extract_version_tuple(model_id: str) -> tuple[int, ...]:
    """Extract version numbers (e.g. 3.3 or 2.5 or 1.5) from model ID."""
    m = _RE_VERSION_MULTI.search(model_id)
    if m:
        try:
            return tuple(int(p) for p in m.group(1).split("."))
        except (ValueError, TypeError):
            pass
    m = _RE_VERSION_SINGLE.search(model_id)
    if m:
        try:
            return (int(m.group(1)),)
        except (ValueError, TypeError):
            pass
    return (0,)


def extract_date_snapshot(model_id: str) -> float:
    """Extract date snapshot from model ID (e.g. 20241022 or 0813)."""
    m = _RE_DATE_FULL.search(model_id)
    if m:
        try:
            return float(m.group(1) + m.group(2) + m.group(3))
        except (ValueError, TypeError):
            pass
    m = _RE_DATE_MONTH_DAY.search(model_id)
    if m:
        try:
            return 20240000.0 + float(m.group(1) + m.group(2))
        except (ValueError, TypeError):
            pass
    return 0.0


def _parse_token_count_spec(raw_val: Any) -> int:
    """Parse context window values like 128000, '128k', '1m', '200000'."""
    if isinstance(raw_val, (int, float)):
        return max(0, int(raw_val))
    s = str(raw_val or "").strip().lower()
    if not s:
        return 0
    if s.endswith("m"):
        try:
            return int(float(s[:-1]) * 1_000_000)
        except ValueError:
            return 0
    if s.endswith("k"):
        try:
            val = float(s[:-1])
            if int(val) in _K_POWER_OF_TWO and val == int(val):
                return _K_POWER_OF_TWO[int(val)]
            return int(val * 1000)
        except ValueError:
            return 0
    try:
        return max(0, int(float(s)))
    except ValueError:
        return 0


def _parse_param_size_spec(raw_val: Any) -> float:
    """Parse parameter size values like 70, '70b', '0.5B', 32.0."""
    if isinstance(raw_val, (int, float)):
        return max(0.0, float(raw_val))
    s = str(raw_val or "").strip().lower()
    if not s:
        return 0.0
    if s.endswith("b"):
        s = s[:-1].strip()
    try:
        return max(0.0, float(s))
    except ValueError:
        return 0.0


# ─── Criteria Parsing ───────────────────────────────────────────────────────


def _apply_dict_fields(
    criteria: FallbackCriteria,
    raw: dict[str, Any],
    raw_dict: dict[str, Any],
    explicit_free_Ref: list[bool],
) -> None:
    """Apply dictionary fields to criteria object."""
    provider = (
        raw_dict.get("model_provider")
        or raw_dict.get("provider")
        or raw.get("model_provider")
        or raw.get("provider")
    )
    if provider:
        if isinstance(provider, (list, tuple)):
            parsed_provs = parse_provider_list(provider)
            if parsed_provs:
                criteria.providers = parsed_provs
                criteria.provider = parsed_provs[0]
        else:
            prov_str = str(provider).strip().lower()
            if "," in prov_str or "|" in prov_str:
                parsed_provs = parse_provider_list(prov_str)
                if len(parsed_provs) > 1:
                    criteria.providers = parsed_provs
                    criteria.provider = parsed_provs[0]
                elif len(parsed_provs) == 1:
                    criteria.provider = parsed_provs[0]
            else:
                criteria.provider = normalize_provider_name(prov_str) or prov_str

    providers_field = raw_dict.get("providers") or raw.get("providers")
    if providers_field:
        parsed_provs = parse_provider_list(providers_field)
        if parsed_provs:
            criteria.providers = parsed_provs
            if criteria.provider == "nous" and not provider:
                criteria.provider = parsed_provs[0]

    exclude_provs = (
        raw_dict.get("exclude_providers")
        or raw_dict.get("exclude_provider")
        or raw.get("exclude_providers")
        or raw.get("exclude_provider")
    )
    if exclude_provs:
        criteria.exclude_providers = parse_provider_list(exclude_provs)

    exclude_models = (
        raw_dict.get("exclude_models")
        or raw_dict.get("exclude_model")
        or raw.get("exclude_models")
        or raw.get("exclude_model")
    )
    if exclude_models:
        if isinstance(exclude_models, str):
            criteria.exclude_models = tuple(
                m.strip() for m in re.split(r"[,|;]+", exclude_models) if m.strip()
            )
        elif isinstance(exclude_models, (list, tuple, set)):
            criteria.exclude_models = tuple(str(m).strip() for m in exclude_models if str(m).strip())

    for switch_key in ("allow_provider_switch", "switch_provider", "provider_switch"):
        if switch_key in raw_dict:
            criteria.allow_provider_switch = bool(raw_dict[switch_key])
            break
        if switch_key in raw:
            criteria.allow_provider_switch = bool(raw[switch_key])
            break

    for pref_key in ("prefer_current_provider", "stay_on_current_provider"):
        if pref_key in raw_dict:
            criteria.prefer_current_provider = bool(raw_dict[pref_key])
            break
        if pref_key in raw:
            criteria.prefer_current_provider = bool(raw[pref_key])
            break

    cur_prov = raw_dict.get("current_provider") or raw.get("current_provider")
    if cur_prov:
        criteria.current_provider = normalize_provider_name(str(cur_prov))

    if "sort_by" in raw_dict:
        criteria.sort_by = str(raw_dict["sort_by"]).strip().lower()
    if "free" in raw_dict:
        criteria.free_only = bool(raw_dict["free"])
        explicit_free_Ref[0] = True
    elif "free_only" in raw_dict:
        criteria.free_only = bool(raw_dict["free_only"])
        explicit_free_Ref[0] = True
    elif "free" in raw:
        criteria.free_only = bool(raw["free"])
        explicit_free_Ref[0] = True
    elif "free_only" in raw:
        criteria.free_only = bool(raw["free_only"])
        explicit_free_Ref[0] = True

    if "require_tools" in raw_dict:
        criteria.require_tools = bool(raw_dict["require_tools"])
    elif "require_tools" in raw:
        criteria.require_tools = bool(raw["require_tools"])

    if "flash" in raw_dict:
        criteria.flash_only = bool(raw_dict["flash"])
    elif "flash_only" in raw_dict:
        criteria.flash_only = bool(raw_dict["flash_only"])

    name_filter = raw_dict.get("filter") or raw_dict.get("name_filter") or raw_dict.get("pattern")
    if name_filter:
        criteria.name_filter = str(name_filter).strip().lower()

    vendor = raw_dict.get("vendor") or raw_dict.get("lab")
    if vendor:
        criteria.vendor = str(vendor).strip().lower()

    for ctx_min_key in ("min_context", "min_ctx", "min_context_window"):
        if ctx_min_key in raw_dict:
            criteria.min_context = _parse_token_count_spec(raw_dict[ctx_min_key])
            break
        if ctx_min_key in raw:
            criteria.min_context = _parse_token_count_spec(raw[ctx_min_key])
            break

    for ctx_max_key in ("max_context", "max_ctx", "max_context_window"):
        if ctx_max_key in raw_dict:
            criteria.max_context = _parse_token_count_spec(raw_dict[ctx_max_key])
            break
        if ctx_max_key in raw:
            criteria.max_context = _parse_token_count_spec(raw[ctx_max_key])
            break

    for param_min_key in ("min_params", "min_params_b", "min_parameter_count"):
        if param_min_key in raw_dict:
            criteria.min_params = _parse_param_size_spec(raw_dict[param_min_key])
            break
        if param_min_key in raw:
            criteria.min_params = _parse_param_size_spec(raw[param_min_key])
            break

    for param_max_key in ("max_params", "max_params_b", "max_parameter_count"):
        if param_max_key in raw_dict:
            criteria.max_params = _parse_param_size_spec(raw_dict[param_max_key])
            break
        if param_max_key in raw:
            criteria.max_params = _parse_param_size_spec(raw[param_max_key])
            break

    if "max_candidates" in raw_dict or "limit" in raw_dict or "max_candidates" in raw:
        try:
            val = (
                raw_dict.get("max_candidates")
                if "max_candidates" in raw_dict
                else raw_dict.get("limit")
                if "limit" in raw_dict
                else raw.get("max_candidates")
            )
            if val is not None:
                criteria.max_candidates = max(1, int(val))
        except (ValueError, TypeError):
            pass

    _apply_dict_reasoning_effort(criteria, raw, raw_dict)


def _apply_dict_reasoning_effort(
    criteria: FallbackCriteria, raw: dict[str, Any], raw_dict: dict[str, Any]
) -> None:
    """Extract and assign reasoning effort setting from dictionary."""
    reasoning = None
    for key in ("reasoning_effort", "thinking", "thinking_effort", "reasoning"):
        if raw_dict.get(key) is not None:
            reasoning = raw_dict[key]
            break
        if isinstance(raw, dict) and raw.get(key) is not None:
            reasoning = raw[key]
            break
    if reasoning is not None:
        if isinstance(reasoning, bool):
            criteria.reasoning_effort = "none" if not reasoning else "default"
        elif isinstance(reasoning, str):
            criteria.reasoning_effort = reasoning.strip().lower()


def _apply_shorthand_tokens(
    criteria: FallbackCriteria, cleaned: str, explicit_free_ref: list[bool]
) -> None:
    """Parse shorthand tokens after key-value specifiers have been extracted."""
    parts = set(_RE_SHORTHAND_SPLIT.split(cleaned))

    if "free" in parts:
        criteria.free_only = True
        explicit_free_ref[0] = True
    elif "paid" in parts or "any_tier" in parts or "all_tiers" in parts:
        criteria.free_only = False
        explicit_free_ref[0] = True

    if "flash" in parts:
        criteria.flash_only = True

    if "tools" in parts or "require_tools" in parts:
        criteria.require_tools = True
    elif "no_tools" in parts:
        criteria.require_tools = False

    if any(
        k in cleaned
        for k in ("greatest_context", "largest_context", "max_context_window", "longest_context")
    ) or (
        "max_context" in cleaned and not re.search(r"max_context[:=]\d", cleaned)
    ):
        criteria.sort_by = "greatest_context"
    elif any(k in cleaned for k in ("smallest", "smallest_model", "fewest_parameters")) or (
        "min_params" in cleaned and not re.search(r"min_params[:=]\d", cleaned)
    ):
        criteria.sort_by = "smallest"
    elif any(k in cleaned for k in ("latest_flash", "newest_flash")):
        criteria.sort_by = "latest_flash"
        criteria.flash_only = True
    elif any(k in cleaned for k in ("latest", "newest", "recent", "most_recent")):
        criteria.sort_by = "latest"
    elif any(k in cleaned for k in ("provider_preference", "preferred_provider", "by_provider")):
        criteria.sort_by = "provider_preference"
    elif any(
        k in cleaned
        for k in (
            "largest_parameter_count",
            "largest_parameters",
            "largest_params",
            "largest_param",
            "largest",
            "largest_model",
            "largest_size",
        )
    ) or ("max_params" in cleaned and not re.search(r"max_params[:=]\d", cleaned)):
        criteria.sort_by = "largest_parameter_count"
    elif criteria.flash_only:
        criteria.sort_by = "latest_flash"


def _apply_shorthand_string(
    criteria: FallbackCriteria, token: str, explicit_free_ref: list[bool]
) -> None:
    """Parse shorthand strings like 'largest_parameter_count_free', 'provider:anthropic,openrouter greatest_context'."""
    cleaned = token.lower().strip()

    m = re.search(r"(?:model_providers|providers)[:=]([a-zA-Z0-9_,\-|]+)", cleaned)
    if m:
        provs = parse_provider_list(m.group(1))
        if provs:
            criteria.providers = provs
            criteria.provider = provs[0]

    m = re.search(r"(?<![a-z_])(?:model_provider|provider)[:=]([a-zA-Z0-9_,\-|:]+)", cleaned)
    if m:
        raw_p = m.group(1).strip().rstrip(",|")
        if "," in raw_p or "|" in raw_p:
            provs = parse_provider_list(raw_p)
            if len(provs) > 1:
                criteria.providers = provs
                criteria.provider = provs[0]
            elif len(provs) == 1:
                criteria.provider = provs[0]
        else:
            criteria.provider = normalize_provider_name(raw_p) or raw_p

    m = re.search(r"(?:exclude_providers|exclude_provider)[:=]([a-zA-Z0-9_,\-|]+)", cleaned)
    if m:
        criteria.exclude_providers = parse_provider_list(m.group(1))

    m = re.search(r"(?:switch_provider|allow_provider_switch)[:=](true|false|yes|no|1|0)", cleaned)
    if m:
        criteria.allow_provider_switch = m.group(1) in ("true", "yes", "1")

    m = re.search(r"(?:prefer_current_provider)[:=](true|false|yes|no|1|0)", cleaned)
    if m:
        criteria.prefer_current_provider = m.group(1) in ("true", "yes", "1")

    m = re.search(r"(?:free_only|free)[:=](true|false|yes|no|1|0)", cleaned)
    if m:
        criteria.free_only = m.group(1) in ("true", "yes", "1")
        explicit_free_ref[0] = True
        cleaned = cleaned[: m.start()] + " " + cleaned[m.end() :]

    m = re.search(r"(?:require_tools|tools)[:=](true|false|yes|no|1|0)", cleaned)
    if m:
        criteria.require_tools = m.group(1) in ("true", "yes", "1")
        cleaned = cleaned[: m.start()] + " " + cleaned[m.end() :]

    m = re.search(r"(?:vendor|lab)[:=]([a-zA-Z0-9_\-]+)", cleaned)
    if m:
        criteria.vendor = m.group(1).lower().strip()

    m = re.search(r"(?:filter|pattern|name_filter)[:=]([a-zA-Z0-9_\-\.\*]+)", cleaned)
    if m:
        criteria.name_filter = m.group(1).lower().strip()

    m = re.search(r"(?:min_ctx|min_context)[:=]([0-9]+(?:\.[0-9]+)?[kmKM]?)", cleaned)
    if m:
        criteria.min_context = _parse_token_count_spec(m.group(1))

    m = re.search(r"(?:max_ctx|max_context)[:=]([0-9]+(?:\.[0-9]+)?[kmKM]?)", cleaned)
    if m:
        criteria.max_context = _parse_token_count_spec(m.group(1))

    m = re.search(r"(?:min_params|min_param)[:=]([0-9]+(?:\.[0-9]+)?[bB]?)", cleaned)
    if m:
        criteria.min_params = _parse_param_size_spec(m.group(1))

    m = re.search(r"(?:max_params|max_param)[:=]([0-9]+(?:\.[0-9]+)?[bB]?)", cleaned)
    if m:
        criteria.max_params = _parse_param_size_spec(m.group(1))

    m = re.search(r"(?:reasoning_effort|reasoning|thinking)[:=]([a-zA-Z0-9_\-]+)", cleaned)
    if m:
        criteria.reasoning_effort = m.group(1).lower().strip()

    # Check leading '<provider>:' prefix (e.g. 'nous:latest_flash_free', 'anthropic:largest', 'existing:latest')
    prefix_match = re.match(r"^([a-z0-9\-]+)[:_](.+)$", cleaned)
    if prefix_match:
        cand_prefix = prefix_match.group(1)
        if cand_prefix in PROVIDER_ALIASES or cand_prefix in ("existing", "any"):
            criteria.provider = normalize_provider_name(cand_prefix) if cand_prefix in PROVIDER_ALIASES else cand_prefix
            cleaned = prefix_match.group(2).strip()

    _apply_shorthand_tokens(criteria, cleaned, explicit_free_ref)


def parse_fallback_criteria(raw: Any) -> FallbackCriteria:
    """Parse FallbackCriteria from dict, string, or entry.

    Supports:
    - Dict: {"criteria": {"sort_by": "largest_parameter_count", "free": True, ...}}
    - Top-level keys in dict: {"provider": "openrouter", "heuristic": "largest_parameter_count_free"}
    - Multi-provider dict: {"provider": "existing", "providers": ["anthropic", "openrouter"], "heuristic": "greatest_context"}
    - Model string tokens: "auto:free:largest_parameter_count", "heuristic:latest_flash", "criteria:..."
    - Shorthand string: "largest_parameter_count_free", "greatest_context", "smallest", "latest_flash"
    """
    if isinstance(raw, FallbackCriteria):
        return raw

    criteria = FallbackCriteria()
    explicit_free_ref = [False]

    if isinstance(raw, dict):
        raw_dict = raw.get("criteria") if isinstance(raw.get("criteria"), dict) else raw
        raw_heuristic = str(
            raw.get("heuristic")
            or (raw.get("criteria") if isinstance(raw.get("criteria"), str) else "")
            or raw_dict.get("sort_by")
            or ""
        ).strip()
        criteria.raw_heuristic = raw_heuristic

        if raw_heuristic and isinstance(raw_heuristic, str):
            _apply_shorthand_string(criteria, raw_heuristic, explicit_free_ref)

        _apply_dict_fields(criteria, raw, raw_dict, explicit_free_ref)

        model_str = str(raw.get("model") or "").strip()
        if model_str.startswith(("auto:", "heuristic:", "criteria:")):
            _apply_shorthand_string(criteria, model_str, explicit_free_ref)

        # If provider is a paid-first provider or multi-provider token ('existing', 'any', 'auto')
        # and free_only was NOT explicitly requested via 'free' token or free=True, allow paid models too.
        if not explicit_free_ref[0]:
            norm_prov = normalize_provider_name(criteria.provider)
            if (
                norm_prov in PAID_FIRST_PROVIDERS
                or norm_prov in MULTI_PROVIDER_TOKENS
                or bool(criteria.providers)
            ):
                criteria.free_only = False

        return criteria

    if isinstance(raw, str):
        criteria.raw_heuristic = raw
        _apply_shorthand_string(criteria, raw, explicit_free_ref)
        if not explicit_free_ref[0]:
            norm_prov = normalize_provider_name(criteria.provider)
            if (
                norm_prov in PAID_FIRST_PROVIDERS
                or norm_prov in MULTI_PROVIDER_TOKENS
                or bool(criteria.providers)
            ):
                criteria.free_only = False
        return criteria

    return criteria


# ─── Metadata Enrichment ───────────────────────────────────────────────────


def _enrich_from_known_specs(meta: ModelCandidateMetadata, model_id: str) -> None:
    """Enrich candidate metadata from built-in cross-provider model specifications."""
    slug_lower = model_id.lower().strip()
    bare_slug = slug_lower.split("/", 1)[-1]
    if bare_slug.endswith(":free"):
        bare_slug = bare_slug[:-5]
    # Also handle bedrock prefixes like us.anthropic.claude-opus-4-6-v1
    cleaned_bare = re.sub(r"^(?:us|eu|ap|global)\.(?:anthropic|meta|amazon|google)\.", "", bare_slug)
    cleaned_bare = re.sub(r"-v\d+(?::\d+)?$", "", cleaned_bare)

    spec = (
        _KNOWN_MODEL_SPECS.get(slug_lower)
        or _KNOWN_MODEL_SPECS.get(bare_slug)
        or _KNOWN_MODEL_SPECS.get(cleaned_bare)
        or _KNOWN_MODEL_SPECS.get(bare_slug.replace(".", "-"))
    )
    if not isinstance(spec, dict):
        return

    if spec.get("param_size_b") and not meta.param_size_b:
        meta.param_size_b = float(spec["param_size_b"])
    if spec.get("context_window") and spec["context_window"] > meta.context_window:
        meta.context_window = int(spec["context_window"])
    if spec.get("supports_tools") is not None:
        meta.supports_tools = bool(spec["supports_tools"])
    if spec.get("is_flash") is not None and spec["is_flash"]:
        meta.is_flash = True
    if spec.get("is_free") is not None and spec["is_free"]:
        meta.is_free = True
    if spec.get("release_date") and not meta.release_date:
        meta.release_date = str(spec["release_date"])
        try:
            parts = [int(p) for p in meta.release_date.split("-") if p.isdigit()]
            if len(parts) >= 3:
                dt = datetime.datetime(parts[0], parts[1], parts[2], tzinfo=datetime.timezone.utc)
                meta.created_ts = dt.timestamp()
        except (ValueError, TypeError):
            pass
    if spec.get("vendor") and (not meta.vendor or meta.vendor == meta.provider):
        meta.vendor = str(spec["vendor"])
    if meta.source == "heuristic":
        meta.source = "builtin_spec"


def _enrich_from_raw_item(meta: ModelCandidateMetadata, model_id: str, raw_item: dict[str, Any]) -> None:
    """Extract metadata from raw OpenRouter API payload."""
    meta.name = str(raw_item.get("name") or model_id)
    if raw_item.get("context_length"):
        try:
            meta.context_window = int(raw_item["context_length"])
        except (ValueError, TypeError):
            pass
    if raw_item.get("created"):
        try:
            meta.created_ts = float(raw_item["created"])
        except (ValueError, TypeError):
            pass
    pricing = raw_item.get("pricing")
    if isinstance(pricing, dict):
        try:
            prompt_cost = float(pricing.get("prompt", 0))
            comp_cost = float(pricing.get("completion", 0))
            if prompt_cost == 0 and comp_cost == 0:
                meta.is_free = True
        except (ValueError, TypeError):
            pass
    params = raw_item.get("supported_parameters")
    if isinstance(params, list):
        meta.supports_tools = "tools" in params
    if is_flash_model(model_id, meta.name):
        meta.is_flash = True
    if not meta.param_size_b:
        meta.param_size_b = extract_param_size_b(model_id, meta.name)
    meta.source = "openrouter_api"


def _enrich_from_openrouter_cache(meta: ModelCandidateMetadata, model_id: str) -> None:
    """Query cached OpenRouter metadata catalog."""
    try:
        from agent.model_metadata import fetch_model_metadata

        catalog = fetch_model_metadata()
        entry = catalog.get(model_id) or catalog.get(model_id.split("/", 1)[-1])
        if isinstance(entry, dict):
            meta.name = str(entry.get("name") or model_id)
            if entry.get("context_length"):
                try:
                    meta.context_window = int(entry["context_length"])
                except (ValueError, TypeError):
                    pass
            pricing = entry.get("pricing")
            if isinstance(pricing, dict):
                try:
                    prompt_cost = float(pricing.get("prompt", 0) or 0)
                    comp_cost = float(pricing.get("completion", 0) or 0)
                    if prompt_cost == 0 and comp_cost == 0:
                        meta.is_free = True
                except (ValueError, TypeError):
                    pass
            params = entry.get("supported_parameters")
            if isinstance(params, list):
                meta.supports_tools = "tools" in params
            elif entry.get("tools") is not None:
                meta.supports_tools = bool(entry.get("tools"))
            meta.source = "openrouter_cache"
    except Exception as exc:
        logger.debug("OpenRouter cache lookup failed for %s: %s", model_id, exc, exc_info=True)


def _enrich_from_nous_pricing(meta: ModelCandidateMetadata, model_id: str) -> None:
    """Query Nous Portal pricing tables."""
    try:
        from hermes_cli.models import _is_model_free
        from hermes_cli import models_pricing as mp

        pricing = mp.get_pricing_for_provider("nous") or {}
        entry = pricing.get(model_id)
        if isinstance(entry, dict):
            if (
                entry.get("tools") is not False
                and not entry.get("generation")
                and not model_id.startswith(
                    ("voyageai/", "sentence-transformers/", "thenlper/", "baai/", "intfloat/")
                )
            ):
                meta.supports_tools = True
            meta.source = "nous_pricing"
        if _is_model_free(model_id, pricing):
            meta.is_free = True
    except Exception as exc:
        logger.debug("Nous pricing lookup failed for %s: %s", model_id, exc, exc_info=True)


def _enrich_from_models_dev(meta: ModelCandidateMetadata, model_id: str, provider: str) -> None:
    """Query models.dev static registry for context and release date."""
    try:
        from agent.models_dev import get_model_info

        m_info = get_model_info(provider, model_id)
        if m_info is not None:
            if m_info.name:
                meta.name = m_info.name
            if m_info.context_window and m_info.context_window > meta.context_window:
                meta.context_window = m_info.context_window
            if m_info.release_date:
                meta.release_date = m_info.release_date
                try:
                    parts = [int(p) for p in m_info.release_date.split("-") if p.isdigit()]
                    if len(parts) >= 3:
                        dt = datetime.datetime(
                            parts[0], parts[1], parts[2], tzinfo=datetime.timezone.utc
                        )
                        meta.created_ts = dt.timestamp()
                except (ValueError, TypeError):
                    pass
            if m_info.has_cost_data() and m_info.cost_input == 0 and m_info.cost_output == 0:
                meta.is_free = True
            if m_info.tool_call is not None:
                meta.supports_tools = m_info.tool_call
            if meta.source == "heuristic":
                meta.source = "models_dev"
    except Exception as exc:
        logger.debug("models.dev lookup failed for %s: %s", model_id, exc, exc_info=True)


def enrich_model_metadata(
    model_id: str,
    provider: str = "openrouter",
    raw_item: Optional[dict[str, Any]] = None,
    base_url: str = "",
) -> ModelCandidateMetadata:
    """Gather multi-tier metadata for a model candidate from API, registry, specs, and heuristics."""
    provider_norm = normalize_provider_name(provider) or provider
    meta = ModelCandidateMetadata(id=model_id, provider=provider_norm)

    meta.vendor = extract_vendor(model_id) or provider_norm
    meta.is_flash = is_flash_model(model_id)
    meta.param_size_b = extract_param_size_b(model_id)
    meta.context_window = extract_context_window_from_name(model_id, default=0)
    meta.version_tuple = extract_version_tuple(model_id)
    meta.date_snapshot = extract_date_snapshot(model_id)
    meta.is_free = is_free_model_slug(model_id)

    if isinstance(raw_item, dict):
        _enrich_from_raw_item(meta, model_id, raw_item)
        _enrich_from_known_specs(meta, model_id)
        return meta

    if provider_norm == "openrouter":
        _enrich_from_openrouter_cache(meta, model_id)
    elif provider_norm in ("nous", "nousresearch", "nous-portal"):
        _enrich_from_nous_pricing(meta, model_id)
        _enrich_from_openrouter_cache(meta, model_id)

    _enrich_from_models_dev(meta, model_id, provider_norm)
    _enrich_from_known_specs(meta, model_id)

    # Local providers (Ollama, LM Studio, vLLM) are free to run locally and support tool calling on modern models
    if provider_norm in ("ollama", "lmstudio", "vllm"):
        meta.is_free = True
        if not meta.supports_tools and any(
            fam in model_id.lower()
            for fam in ("qwen", "llama", "deepseek", "mistral", "glm", "hermes", "function", "tool")
        ):
            meta.supports_tools = True

    if meta.context_window <= 0:
        try:
            from agent.model_metadata import (
                DEFAULT_CONTEXT_LENGTHS,
                _config_override_context_length,
                _longest_key_match,
                _resolve_endpoint_context_length,
                _resolve_provider_aware_context_length,
            )

            ctx = _config_override_context_length(model_id, base_url, provider_norm, None)
            if ctx is None and base_url:
                ctx = _resolve_endpoint_context_length(model_id, base_url)
            if ctx is None:
                ctx = _resolve_provider_aware_context_length(
                    model_id, base_url, "", provider_norm, provider_norm
                )
            if ctx is None:
                hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model_id.lower())
                if hit:
                    ctx = hit[1]
            if ctx and ctx > 0:
                meta.context_window = ctx
        except Exception as exc:
            logger.debug(
                "Context length positive resolution failed for %s (%s): %s",
                model_id,
                provider_norm,
                exc,
                exc_info=True,
            )

    return meta


# ─── Candidate Gathering and Caching ────────────────────────────────────────


_CANDIDATE_CACHE: dict[tuple[str, str, str], tuple[float, list[ModelCandidateMetadata]]] = {}
_CANDIDATE_CACHE_TTL = 300.0  # 5 minutes
_CANDIDATE_NEGATIVE_CACHE_TTL = 30.0  # 30 seconds for empty results


def clear_candidate_cache() -> None:
    """Clear the candidate models in-memory cache (for testing or manual refresh)."""
    _CANDIDATE_CACHE.clear()


def _gather_nous_candidates(
    provider_norm: str, force_refresh: bool = False
) -> dict[str, ModelCandidateMetadata]:
    """Gather candidates for Nous Research Portal."""
    candidates_by_id: dict[str, ModelCandidateMetadata] = {}
    try:
        from hermes_cli import models_pricing as mp
        from hermes_cli.models import _is_model_free, provider_model_ids

        mids = list(provider_model_ids("nous", force_refresh=force_refresh) or [])
        pricing = mp.get_pricing_for_provider("nous") or {}
        for mid in pricing:
            if mid not in mids and _is_model_free(mid, pricing):
                mids.append(mid)
        for mid in mids:
            cand = enrich_model_metadata(mid, provider="nous")
            if _is_model_free(mid, pricing):
                cand.is_free = True
            candidates_by_id[mid] = cand
    except Exception as exc:
        logger.debug("Nous catalog probe failed in fallback heuristics: %s", exc, exc_info=True)

    if not candidates_by_id:
        for mid in _BUILTIN_PROVIDER_CATALOGS.get("nous", ()):
            cand = enrich_model_metadata(mid, provider="nous")
            if is_free_model_slug(mid):
                cand.is_free = True
                cand.supports_tools = True
            candidates_by_id[mid] = cand

    return candidates_by_id


def _load_openrouter_live_catalog(provider_norm: str) -> dict[str, ModelCandidateMetadata]:
    candidates_by_id: dict[str, ModelCandidateMetadata] = {}
    try:
        from hermes_cli.models import (
            _OPENROUTER_CATALOG_URL,
            _fetch_live_catalog_index,
            _urlopen_model_catalog_request,
        )

        live = _fetch_live_catalog_index(
            _OPENROUTER_CATALOG_URL, 6.0, _urlopen_model_catalog_request
        )
        if live is not None:
            live_items, _ = live
            for item in live_items:
                if isinstance(item, dict):
                    mid = str(item.get("id") or "").strip()
                    if mid:
                        candidates_by_id[mid] = enrich_model_metadata(
                            mid, provider=provider_norm, raw_item=item
                        )
    except Exception as exc:
        logger.debug("Live OpenRouter catalog probe failed: %s", exc, exc_info=True)
    return candidates_by_id


def _load_openrouter_static_catalog(provider_norm: str) -> dict[str, ModelCandidateMetadata]:
    candidates_by_id: dict[str, ModelCandidateMetadata] = {}
    try:
        from hermes_cli.models_catalog_static import OPENROUTER_MODELS

        source_models = list(OPENROUTER_MODELS)
    except Exception:
        source_models = list(_BUILTIN_OPENROUTER_STATIC)

    for mid, desc in source_models:
        cand = enrich_model_metadata(mid, provider=provider_norm)
        if desc == "free" or mid.endswith(":free"):
            cand.is_free = True
            if not cand.supports_tools:
                cand.supports_tools = True
        candidates_by_id[mid] = cand
    return candidates_by_id


def _load_openrouter_disk_catalog(provider_norm: str) -> dict[str, ModelCandidateMetadata]:
    candidates_by_id: dict[str, ModelCandidateMetadata] = {}
    try:
        for p in (
            Path(__file__).resolve().parent.parent / "website" / "static" / "api" / "model-catalog.json",
            _get_hermes_home() / "cache" / "model_catalog.json",
        ):
            if not p.is_file():
                continue
            with open(p, "r", encoding="utf-8-sig") as fh:
                data = json.load(fh)
            m_list = data.get("providers", {}).get("openrouter", {}).get("models", [])
            for item in m_list:
                mid = str(item.get("id") or "").strip()
                if mid and mid not in candidates_by_id:
                    cand = enrich_model_metadata(mid, provider=provider_norm)
                    if ":free" in mid or "free" in str(item.get("description") or "").lower():
                        cand.is_free = True
                    candidates_by_id[mid] = cand
            if candidates_by_id:
                break
    except Exception as exc:
        logger.debug("OpenRouter disk catalog load failed: %s", exc, exc_info=True)
    return candidates_by_id


def _gather_openrouter_candidates(
    provider_norm: str, force_refresh: bool = False
) -> dict[str, ModelCandidateMetadata]:
    """Gather candidates for OpenRouter through live API, static catalog, or disk cache."""
    candidates_by_id = _load_openrouter_live_catalog(provider_norm)
    if not candidates_by_id:
        candidates_by_id = _load_openrouter_static_catalog(provider_norm)
    if not candidates_by_id:
        candidates_by_id = _load_openrouter_disk_catalog(provider_norm)

    return candidates_by_id


def _gather_custom_provider_candidates(
    provider_norm: str, base_url: str = "", config: Optional[dict[str, Any]] = None
) -> dict[str, ModelCandidateMetadata]:
    """Gather models configured under custom_providers or providers.<name> in config.yaml."""
    candidates_by_id: dict[str, ModelCandidateMetadata] = {}
    cfg = _load_active_config(config)
    clean_name = provider_norm[7:] if provider_norm.startswith("custom:") else provider_norm

    # Check providers.<name>
    providers_block = cfg.get("providers")
    if isinstance(providers_block, dict):
        pblock = providers_block.get(clean_name) or providers_block.get(provider_norm)
        if isinstance(pblock, dict):
            eff_url = str(pblock.get("base_url") or base_url or "").strip()
            models_val = pblock.get("models") or pblock.get("model")
            if isinstance(models_val, str) and models_val.strip():
                models_val = [models_val.strip()]
            if isinstance(models_val, list):
                for item in models_val:
                    mid = str(item.get("id") if isinstance(item, dict) else item).strip()
                    if mid:
                        cand = enrich_model_metadata(mid, provider=provider_norm, base_url=eff_url)
                        if not cand.supports_tools:
                            cand.supports_tools = True
                        candidates_by_id[mid] = cand

    # Check custom_providers list
    custom_list = cfg.get("custom_providers")
    if isinstance(custom_list, list):
        for citem in custom_list:
            if not isinstance(citem, dict):
                continue
            cname = str(citem.get("name") or "").strip().lower()
            if cname in (clean_name, provider_norm):
                eff_url = str(citem.get("base_url") or base_url or "").strip()
                models_val = citem.get("models") or citem.get("model")
                if isinstance(models_val, str) and models_val.strip():
                    models_val = [models_val.strip()]
                if isinstance(models_val, list):
                    for item in models_val:
                        mid = str(item.get("id") if isinstance(item, dict) else item).strip()
                        if mid:
                            cand = enrich_model_metadata(mid, provider=provider_norm, base_url=eff_url)
                            if not cand.supports_tools:
                                cand.supports_tools = True
                            candidates_by_id[mid] = cand

    return candidates_by_id


def _gather_generic_candidates(
    provider_norm: str, base_url: str = "", force_refresh: bool = False
) -> dict[str, ModelCandidateMetadata]:
    """Gather candidate models for any provider (Anthropic, OpenAI, Gemini, DeepSeek, Z.ai, Kimi, MiniMax, xAI, Ollama, custom, etc.)."""
    candidates_by_id: dict[str, ModelCandidateMetadata] = {}

    # 1. Custom provider entries in config.yaml
    if provider_norm.startswith("custom:"):
        candidates_by_id.update(_gather_custom_provider_candidates(provider_norm, base_url=base_url))

    # 2. Hermes Core provider_model_ids (queries live/cached provider endpoints & models.dev)
    try:
        from hermes_cli.models import provider_model_ids

        model_ids = provider_model_ids(provider_norm, force_refresh=force_refresh) or []
        for mid in model_ids:
            if mid and mid not in candidates_by_id:
                candidates_by_id[mid] = enrich_model_metadata(
                    mid, provider=provider_norm, base_url=base_url
                )
    except Exception as exc:
        logger.debug("Generic provider model probe failed: %s", exc, exc_info=True)

    # 3. Hermes static _PROVIDER_MODELS table
    if not candidates_by_id:
        try:
            from hermes_cli.models_catalog_static import _PROVIDER_MODELS

            static_list = _PROVIDER_MODELS.get(provider_norm, [])
            for mid in static_list:
                if mid and mid not in candidates_by_id:
                    candidates_by_id[mid] = enrich_model_metadata(
                        mid, provider=provider_norm, base_url=base_url
                    )
        except Exception as exc:
            logger.debug("Static provider model fallback failed: %s", exc, exc_info=True)

    # 4. Built-in cross-provider catalog (works 100% offline / outside Hermes checkout)
    if not candidates_by_id and provider_norm in _BUILTIN_PROVIDER_CATALOGS:
        for mid in _BUILTIN_PROVIDER_CATALOGS[provider_norm]:
            if mid and mid not in candidates_by_id:
                candidates_by_id[mid] = enrich_model_metadata(
                    mid, provider=provider_norm, base_url=base_url
                )

    # 5. Also check custom config definitions for non-prefixed named providers
    if not candidates_by_id:
        candidates_by_id.update(_gather_custom_provider_candidates(provider_norm, base_url=base_url))

    return candidates_by_id


def get_candidate_models(
    provider: str = "nous",
    base_url: str | None = "",
    force_refresh: bool = False,
    config: Optional[dict[str, Any]] = None,
    providers: Optional[ tuple[str, ...] | list[str] ] = None,
) -> list[ModelCandidateMetadata]:
    """Gather candidate models for one or more providers, enriched with metadata.

    - When ``provider`` is a specific provider (e.g. ``"nous"``, ``"openrouter"``, ``"anthropic"``),
      returns candidates for that provider.
    - When ``provider`` is ``"auto"``, ``"existing"``, or ``"any"`` (or ``providers`` is given),
      gathers candidates across all target/existing providers.
    """
    provider_raw = (provider or "nous").strip().lower()
    provider_norm = normalize_provider_name(provider_raw) or provider_raw

    # Multi-provider gathering when 'providers' list or 'existing'/'any'/'auto' is requested
    target_providers: list[str] = []
    if providers:
        target_providers = list(parse_provider_list(providers))
    elif "," in provider_raw or "|" in provider_raw:
        target_providers = list(parse_provider_list(provider_raw))
    elif provider_norm in MULTI_PROVIDER_TOKENS:
        target_providers = discover_existing_providers(config=config)
        if not target_providers:
            target_providers = ["nous", "openrouter"]

    if len(target_providers) > 1 or (target_providers and provider_norm in MULTI_PROVIDER_TOKENS):
        combined: list[ModelCandidateMetadata] = []
        for tprov in target_providers:
            combined.extend(
                get_candidate_models(
                    provider=tprov,
                    base_url=base_url,
                    force_refresh=force_refresh,
                    config=config,
                )
            )
        return combined

    if len(target_providers) == 1:
        provider_norm = target_providers[0]

    home_key = _get_hermes_home_key()
    base_url_norm = (base_url or "").strip()
    cache_key = (home_key, provider_norm, base_url_norm)
    now = time.monotonic()

    if not force_refresh and cache_key in _CANDIDATE_CACHE:
        cached_ts, cached_candidates = _CANDIDATE_CACHE[cache_key]
        ttl = _CANDIDATE_CACHE_TTL if cached_candidates else _CANDIDATE_NEGATIVE_CACHE_TTL
        if now - cached_ts < ttl:
            return cached_candidates

    candidates_by_id: dict[str, ModelCandidateMetadata] = {}

    if provider_norm in ("nous", "nousresearch", "nous-portal"):
        candidates_by_id.update(_gather_nous_candidates(provider_norm, force_refresh))
    elif provider_norm == "openrouter":
        candidates_by_id.update(_gather_openrouter_candidates(provider_norm, force_refresh))
    else:
        candidates_by_id.update(_gather_generic_candidates(provider_norm, base_url_norm, force_refresh))

    res = list(candidates_by_id.values())
    _CANDIDATE_CACHE[cache_key] = (now, res)
    return res


# ─── Candidate Ranking and Sorting ──────────────────────────────────────────


def _key_largest_params(c: ModelCandidateMetadata) -> tuple[Any, ...]:
    size_val = c.param_size_b if c.param_size_b is not None else 1.0
    return (
        size_val,
        c.context_window,
        c.created_ts,
        c.date_snapshot,
        c.version_tuple,
    )


def _key_greatest_context(c: ModelCandidateMetadata) -> tuple[Any, ...]:
    return (
        c.context_window,
        c.param_size_b if c.param_size_b is not None else 1.0,
        c.created_ts,
        c.date_snapshot,
        c.version_tuple,
    )


def _key_smallest(c: ModelCandidateMetadata) -> tuple[Any, ...]:
    return (
        -(c.param_size_b if c.param_size_b is not None and c.param_size_b > 0 else 999999.0),
        -c.context_window,
    )


def _key_latest(c: ModelCandidateMetadata) -> tuple[Any, ...]:
    has_date = bool(c.created_ts or c.date_snapshot)
    return (
        c.created_ts,
        c.date_snapshot,
        c.version_tuple if has_date else (0,),
        c.param_size_b if c.param_size_b is not None else 1.0,
        c.context_window,
    )


def _key_latest_flash(c: ModelCandidateMetadata) -> tuple[Any, ...]:
    has_date = bool(c.created_ts or c.date_snapshot)
    return (
        1 if c.is_flash else 0,
        c.created_ts,
        c.date_snapshot,
        c.version_tuple if has_date else (0,),
        c.param_size_b if c.param_size_b is not None else 1.0,
        c.context_window,
    )


_SORT_KEY_DISPATCH: dict[str, Callable[[ModelCandidateMetadata], tuple[Any, ...]]] = {
    "largest_parameter_count": _key_largest_params,
    "largest_parameters": _key_largest_params,
    "largest_params": _key_largest_params,
    "largest": _key_largest_params,
    "largest_size": _key_largest_params,
    "max_params": _key_largest_params,
    "params": _key_largest_params,
    "greatest_context": _key_greatest_context,
    "largest_context": _key_greatest_context,
    "max_context": _key_greatest_context,
    "longest_context": _key_greatest_context,
    "context": _key_greatest_context,
    "smallest": _key_smallest,
    "smallest_parameters": _key_smallest,
    "smallest_size": _key_smallest,
    "min_params": _key_smallest,
    "latest": _key_latest,
    "newest": _key_latest,
    "most_recent": _key_latest,
    "recency": _key_latest,
    "latest_flash": _key_latest_flash,
    "provider_preference": _key_largest_params,
    "provider": _key_largest_params,
    "preferred_provider": _key_largest_params,
}


def _provider_preference_score(cand: ModelCandidateMetadata, criteria: FallbackCriteria) -> int:
    """Compute a higher-is-better score for a candidate's provider based on criteria.providers or default order."""
    c_prov = normalize_provider_name(cand.provider)
    if not c_prov:
        return 0
    order = criteria.providers if criteria.providers else DEFAULT_PROVIDER_ORDER
    if c_prov in order:
        return len(order) - order.index(c_prov)
    return 0


def filter_and_rank_candidates(
    candidates: list[ModelCandidateMetadata],
    criteria: FallbackCriteria,
) -> list[ModelCandidateMetadata]:
    """Filter candidate models and sort them by the specified heuristic (including provider parameters)."""
    filtered: list[ModelCandidateMetadata] = []
    allowed_provs = set(criteria.providers) if criteria.providers else set()
    excluded_provs = set(criteria.exclude_providers) if criteria.exclude_providers else set()
    excluded_models = {m.lower() for m in criteria.exclude_models} if criteria.exclude_models else set()
    current_prov_norm = normalize_provider_name(criteria.current_provider)

    for cand in candidates:
        cand_prov = normalize_provider_name(cand.provider)
        if cand_prov:
            if excluded_provs and cand_prov in excluded_provs:
                continue
            if not criteria.allow_provider_switch and current_prov_norm and cand_prov != current_prov_norm:
                continue
            if allowed_provs and cand_prov not in allowed_provs:
                continue

        if excluded_models and (
            cand.id.lower() in excluded_models
            or f"{cand_prov}/{cand.id}".lower() in excluded_models
        ):
            continue

        if criteria.free_only and not cand.is_free:
            continue
        if criteria.require_tools and not cand.supports_tools:
            continue
        if criteria.flash_only and not cand.is_flash:
            continue
        if criteria.vendor and criteria.vendor.lower() != cand.vendor.lower():
            continue
        if criteria.name_filter:
            pattern = criteria.name_filter.lower()
            name_lower = f"{cand.id} {cand.name}".lower()
            if not fnmatch.fnmatch(name_lower, f"*{pattern}*"):
                continue
        if criteria.min_context > 0 and cand.context_window < criteria.min_context:
            continue
        if criteria.max_context > 0 and cand.context_window > criteria.max_context:
            continue
        if criteria.min_params > 0:
            if cand.param_size_b is None or cand.param_size_b < criteria.min_params:
                continue
        if criteria.max_params > 0:
            if cand.param_size_b is None or cand.param_size_b > criteria.max_params:
                continue

        filtered.append(cand)

    if not filtered:
        return []

    sort_by = criteria.sort_by.lower().strip()
    base_sort_fn = _SORT_KEY_DISPATCH.get(sort_by, _key_largest_params)

    def _composite_key(c: ModelCandidateMetadata) -> tuple[Any, ...]:
        c_prov = normalize_provider_name(c.provider)
        is_current = 1 if (criteria.prefer_current_provider and current_prov_norm and c_prov == current_prov_norm) else 0
        prov_score = _provider_preference_score(c, criteria)
        base_key = base_sort_fn(c)
        if sort_by in ("provider_preference", "provider", "preferred_provider"):
            return (is_current, prov_score, *base_key)
        return (is_current, *base_key, prov_score)

    filtered.sort(key=_composite_key, reverse=True)
    return filtered


# ─── Public Resolver ────────────────────────────────────────────────────────


def _resolve_target_providers(
    criteria: FallbackCriteria,
    explicit_provider: str = "",
    config: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Resolve the list of providers to query for a heuristic criteria object."""
    if criteria.providers:
        provs = list(criteria.providers)
    else:
        raw_p = (explicit_provider or criteria.provider or "nous").strip().lower()
        if "," in raw_p or "|" in raw_p:
            provs = list(parse_provider_list(raw_p))
        else:
            norm_p = normalize_provider_name(raw_p) or raw_p
            if norm_p in MULTI_PROVIDER_TOKENS:
                provs = discover_existing_providers(config=config)
                if not provs:
                    provs = ["nous", "openrouter"]
            else:
                provs = [norm_p]

    if criteria.exclude_providers:
        excluded = set(criteria.exclude_providers)
        provs = [p for p in provs if p not in excluded]

    if not criteria.allow_provider_switch and criteria.current_provider:
        cur = normalize_provider_name(criteria.current_provider)
        if cur:
            provs = [p for p in provs if p == cur]

    return provs


def resolve_fallback_candidates(
    provider: str,
    criteria: FallbackCriteria | dict[str, Any] | str,
    base_url: str = "",
    max_candidates: Optional[int] = None,
    config: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Resolve model IDs matching the criteria for one or more providers."""
    parsed_criteria = parse_fallback_criteria(criteria)
    entry: dict[str, Any] = {
        "provider": provider,
        "base_url": base_url,
        "criteria": parsed_criteria.to_dict(),
    }
    resolved = resolve_fallback_entry(entry, max_candidates=max_candidates, config=config)
    return [r["model"] for r in resolved if r.get("model")]


def resolve_fallback_entry(
    entry: dict[str, Any],
    max_candidates: Optional[int] = None,
    config: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Resolve a raw fallback entry into one or more concrete fallback entries.

    - If the entry already has a static model and no criteria/heuristic, returns ``[entry]``.
    - If the entry specifies ``criteria`` or ``heuristic``, resolves the top matching models
      across the target provider(s) (supporting single provider, explicit ``providers`` list,
      or ``provider: "existing"`` / ``"auto"`` across all authenticated providers) and returns
      concrete ``{"provider": ..., "model": ...}`` entries for the fallback chain.
    """
    if not isinstance(entry, dict):
        return []

    provider_raw = str(entry.get("provider") or entry.get("model_provider") or "").strip()
    model = str(entry.get("model") or "").strip()
    has_criteria = bool(entry.get("criteria") or entry.get("heuristic") or entry.get("providers")) or (
        model.startswith(("auto:", "heuristic:", "criteria:"))
    )

    if not has_criteria and model:
        return [entry]

    criteria = parse_fallback_criteria(entry)
    target_providers = _resolve_target_providers(
        criteria, explicit_provider=provider_raw, config=config
    )
    if not target_providers:
        return []

    base_url = str(entry.get("base_url") or "").strip()
    limit = max_candidates or criteria.max_candidates or 3

    if len(target_providers) == 1:
        if config is None:
            candidates = get_candidate_models(provider=target_providers[0], base_url=base_url)
        else:
            candidates = get_candidate_models(
                provider=target_providers[0], base_url=base_url, config=config
            )
    else:
        candidates = get_candidate_models(
            provider="existing",
            base_url=base_url,
            config=config,
            providers=target_providers,
        )
        if not criteria.providers:
            criteria.providers = tuple(target_providers)

    ranked = filter_and_rank_candidates(candidates, criteria)

    if not ranked:
        logger.warning(
            "No models found matching criteria %s for provider(s) %s",
            criteria.describe(),
            ",".join(target_providers),
        )
        return []

    entry_effort = None
    for key in ("reasoning_effort", "thinking_effort", "thinking", "reasoning"):
        if entry.get(key) is not None:
            entry_effort = entry[key]
            break
    if entry_effort is None and criteria.reasoning_effort is not None:
        entry_effort = criteria.reasoning_effort

    resolved: list[dict[str, Any]] = []
    for rank, cand in enumerate(ranked[:limit]):
        cand_provider = normalize_provider_name(cand.provider) or target_providers[0]
        resolved_entry = {
            **entry,
            "provider": cand_provider,
            "model": cand.id,
            "criteria_matched": criteria.describe(),
            "criteria_rank": rank + 1,
            "criteria_total_matched": len(ranked),
            "_is_heuristic": True,
            "_resolved_criteria": criteria.to_dict(),
        }
        resolved_entry.pop("criteria", None)
        resolved_entry.pop("heuristic", None)
        resolved_entry.pop("providers", None)
        resolved_entry.pop("exclude_providers", None)
        resolved_entry.pop("exclude_models", None)
        if entry_effort is not None:
            resolved_entry["reasoning_effort"] = entry_effort
        elif "reasoning_effort" in resolved_entry:
            del resolved_entry["reasoning_effort"]

        resolved.append(resolved_entry)

    logger.info(
        "Resolved heuristic fallback for %s (%s): %s",
        ",".join(target_providers),
        criteria.describe(),
        " -> ".join(f"{r['provider']}/{r['model']} (#{r['criteria_rank']})" for r in resolved),
    )
    return resolved


# Public convenience aliases
parse_context_length = _parse_token_count_spec
parse_param_size = _parse_param_size_spec
extract_param_billions = extract_param_size_b
fetch_provider_candidates = get_candidate_models
fetch_all_candidates = get_candidate_models
fetch_openrouter_candidates = _gather_openrouter_candidates
resolve_heuristic_entry = resolve_fallback_entry


def is_heuristic_fallback_entry(entry: Any) -> bool:
    """Return True if an entry in fallback_providers is a declarative heuristic rule."""
    if not isinstance(entry, dict):
        return False
    model = str(entry.get("model") or "").strip()
    return bool(
        entry.get("criteria")
        or entry.get("heuristic")
        or entry.get("providers")
        or model.startswith(("auto:", "heuristic:", "criteria:"))
    )

