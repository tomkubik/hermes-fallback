"""First-launch onboarding and live free-model provider discovery for the
Hermes Model Fallback Heuristic plugin.

On first launch of the plugin (or whenever invoked via `/fallback setup`,
`/fallback free-providers`, or `hermes fallback-heuristic setup`), this module:
1. Runs a check across live/cached provider catalogs (`models.dev`, OpenRouter,
   Nous Portal, OpenCode, KiloCode, Vercel AI Gateway, and Hermes provider
   registries) to discover every provider offering free ($0) models.
2. Presents the full list of free-model providers to the user — highlighting
   OpenRouter and other top free-tier providers — along with free model counts,
   sample models, API key signup URLs, and environment variable names.
3. Lets the user decide which providers to configure and include in their
   heuristic fallback chain.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional

try:
    from .config_sync import (
        PLUGIN_ID,
        get_plugin_settings,
        load_hermes_config,
        save_hermes_config,
        update_plugin_settings,
    )
    from .heuristics import (
        PROVIDER_ENV_VARS,
        _get_hermes_home,
        _read_secret_or_env,
        discover_existing_providers,
        get_candidate_models,
        normalize_provider_name,
        parse_provider_list,
    )
except ImportError:
    from config_sync import (  # type: ignore
        PLUGIN_ID,
        get_plugin_settings,
        load_hermes_config,
        save_hermes_config,
        update_plugin_settings,
    )
    from heuristics import (  # type: ignore
        PROVIDER_ENV_VARS,
        _get_hermes_home,
        _read_secret_or_env,
        discover_existing_providers,
        get_candidate_models,
        normalize_provider_name,
        parse_provider_list,
    )

logger = logging.getLogger(__name__)


# Known metadata for Hermes providers that offer free ($0) models or a free API tier.
# The live check in `check_free_model_providers()` enriches this with live catalog counts
# and dynamically adds any additional providers found in `models.dev` with $0 models.
FREE_PROVIDER_CATALOG_META: Dict[str, Dict[str, Any]] = {
    "openrouter": {
        "display_name": "OpenRouter",
        "recommended": True,
        "signup_url": "https://openrouter.ai/keys",
        "env_vars": ["OPENROUTER_API_KEY"],
        "free_tier_type": "free_models_catalog",
        "summary": "60+ free `:free` models (Llama, Qwen, Nemotron, DeepSeek, MiniMax, Gemma) with one API key.",
        "default_free_samples": [
            "nvidia/nemotron-3-super-120b-a12b:free",
            "minimax/minimax-m3.1-flash:free",
            "qwen/qwen3-coder:free",
            "deepseek/deepseek-r1:free",
            "meta-llama/llama-3.3-70b-instruct:free",
        ],
    },
    "gemini": {
        "display_name": "Google AI Studio (Gemini)",
        "recommended": True,
        "signup_url": "https://aistudio.google.com/apikey",
        "env_vars": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "free_tier_type": "free_api_tier",
        "summary": "Generous free API tier on Google AI Studio (1M+ token context windows, tool calling, Flash & Pro).",
        "default_free_samples": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-3.7-flash",
            "gemini-flash-lite-latest",
        ],
    },
    "nvidia": {
        "display_name": "NVIDIA NIM (build.nvidia.com)",
        "recommended": True,
        "signup_url": "https://build.nvidia.com",
        "env_vars": ["NVIDIA_API_KEY"],
        "free_tier_type": "free_models_catalog",
        "summary": "100+ zero-cost / free developer tier endpoints (Nemotron, Llama 3.3 70B, Mistral, Mixtral).",
        "default_free_samples": [
            "nvidia/llama-3.3-70b-instruct",
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "mistralai/mistral-nemotron",
            "mistralai/mixtral-8x7b-instruct",
        ],
    },
    "opencode-zen": {
        "display_name": "OpenCode Zen",
        "recommended": True,
        "signup_url": "https://opencode.ai/zen",
        "env_vars": ["OPENCODE_ZEN_API_KEY", "OPENCODE_API_KEY"],
        "free_tier_type": "free_models_catalog",
        "summary": "40+ zero-cost `-free` models in the OpenCode Zen gateway.",
        "default_free_samples": [
            "nemotron-3-ultra-free",
            "nemotron-3-super-free",
            "nemotron-3.5-super-vl-free",
            "ling-3.0-flash-fin-free",
            "ling-3.0-tiny-free",
        ],
    },
    "kilocode": {
        "display_name": "Kilo Code Gateway",
        "recommended": True,
        "signup_url": "https://kilocode.ai",
        "env_vars": ["KILOCODE_API_KEY"],
        "free_tier_type": "free_models_catalog",
        "summary": "16–23+ zero-cost `:free` coding and general models via Kilo Gateway.",
        "default_free_samples": [
            "minimax/minimax-m3.1-flash:free",
            "cohere/north-mini-code:free",
            "xingchen-agi/xing4.0-29b-a4b:free",
            "thinkingmachines/inkling-small:free",
            "openrouter/free",
        ],
    },
    "nous": {
        "display_name": "Nous Research Portal",
        "recommended": True,
        "signup_url": "https://portal.nousresearch.com",
        "env_vars": ["NOUS_API_KEY"],
        "free_tier_type": "free_models_catalog",
        "summary": "Free recommended `:free` models via Nous Portal API key or `hermes login` OAuth.",
        "default_free_samples": [
            "stepfun/step-3.7-flash:free",
            "poolside/laguna-s-2.1:free",
            "poolside/laguna-xs-2.1:free",
            "upstage/solar-mini4:free",
            "meituan/longcat-2.5-preview:free",
        ],
    },
    "ai-gateway": {
        "display_name": "Vercel AI Gateway",
        "recommended": False,
        "signup_url": "https://vercel.com/ai-gateway",
        "env_vars": ["AI_GATEWAY_API_KEY"],
        "free_tier_type": "free_models_catalog",
        "summary": "9 zero-cost models (Llama 3.3 70B, Llama 4 Maverick/Scout, MiniMax M3.1 Flash).",
        "default_free_samples": [
            "meta/llama-3.3-70b",
            "meta/llama-4-maverick",
            "meta/llama-4-scout",
            "minimax/minimax-m3.1-flash",
            "reflection/beam-free",
        ],
    },
    "opencode-go": {
        "display_name": "OpenCode Go",
        "recommended": False,
        "signup_url": "https://opencode.ai",
        "env_vars": ["OPENCODE_GO_API_KEY"],
        "free_tier_type": "free_models_catalog",
        "summary": "5 zero-cost `-free` models (MiniMax M3.1 Flash, Step 5 Preview, LongCat 2.5).",
        "default_free_samples": [
            "minimax-m3.1-flash-free",
            "step-5-preview-free",
            "longcat-2.5-preview-free",
            "cat-jam-free",
            "beam-free",
        ],
    },
    "zai": {
        "display_name": "Z.AI / Zhipu GLM",
        "recommended": False,
        "signup_url": "https://chat.z.ai",
        "env_vars": ["GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"],
        "free_tier_type": "free_models_catalog",
        "summary": "3 permanently free GLM Flash models (`glm-4.7-flash`, `glm-4.6v-flash`, `glm-4.5-flash`).",
        "default_free_samples": [
            "glm-4.7-flash",
            "glm-4.6v-flash",
            "glm-4.5-flash",
        ],
    },
    "huggingface": {
        "display_name": "Hugging Face Inference",
        "recommended": False,
        "signup_url": "https://huggingface.co/settings/tokens",
        "env_vars": ["HF_TOKEN", "HUGGINGFACE_API_KEY"],
        "free_tier_type": "free_api_tier",
        "summary": "Free serverless inference tier with a free Hugging Face account token.",
        "default_free_samples": [
            "zai-org/GLM-4.7-Flash",
            "Qwen/Qwen3.5-72B-Instruct",
            "deepseek-ai/DeepSeek-V3.2",
        ],
    },
    "alibaba": {
        "display_name": "Alibaba Cloud DashScope / Qwen",
        "recommended": False,
        "signup_url": "https://modelstudio.console.alibabacloud.com",
        "env_vars": ["DASHSCOPE_API_KEY"],
        "free_tier_type": "free_api_tier",
        "summary": "Free token quota on Model Studio + free Qwen OAuth (`qwen-oauth`) and distilled R1 models.",
        "default_free_samples": [
            "qwen3.8-flash",
            "qwen3.6-flash",
            "deepseek-r1-distill-llama-8b",
            "deepseek-r1-distill-qwen-1-5b",
        ],
    },
    "tencent-tokenhub": {
        "display_name": "Tencent TokenHub",
        "recommended": False,
        "signup_url": "https://tokenhub.tencentmaas.com",
        "env_vars": ["TENCENT_TOKENHUB_API_KEY"],
        "free_tier_type": "free_models_catalog",
        "summary": "Free preview models (`hy3-preview`, `hy3`) on Tencent TokenHub.",
        "default_free_samples": [
            "hy3-preview",
            "hy3",
        ],
    },
    "commandcode": {
        "display_name": "CommandCode",
        "recommended": False,
        "signup_url": "https://commandcode.ai",
        "env_vars": ["COMMANDCODE_API_KEY"],
        "free_tier_type": "free_models_catalog",
        "summary": "Offers free Flash coding model (`inclusionai/ling-3.1-flash:free`).",
        "default_free_samples": [
            "inclusionai/ling-3.1-flash:free",
        ],
    },
    "novita": {
        "display_name": "Novita AI",
        "recommended": False,
        "signup_url": "https://novita.ai/settings/key-management",
        "env_vars": ["NOVITA_API_KEY"],
        "free_tier_type": "free_models_catalog",
        "summary": "Offers free mini model (`apodex/apodex-1.1-mini`) and open-weights catalog.",
        "default_free_samples": [
            "apodex/apodex-1.1-mini",
        ],
    },
    "groq": {
        "display_name": "GroqCloud",
        "recommended": False,
        "signup_url": "https://console.groq.com/keys",
        "env_vars": ["GROQ_API_KEY"],
        "free_tier_type": "free_api_tier",
        "summary": "Free developer API tier + $0 model (`allam-2-7b`, Llama 3.3 70B, Qwen).",
        "default_free_samples": [
            "allam-2-7b",
            "llama-3.3-70b-versatile",
        ],
    },
    "mistral": {
        "display_name": "Mistral AI (La Plateforme)",
        "recommended": False,
        "signup_url": "https://console.mistral.ai/api-keys",
        "env_vars": ["MISTRAL_API_KEY"],
        "free_tier_type": "free_models_catalog",
        "summary": "Free Experiment plan + $0 Labs models (`labs-devstral-small-2512`, `labs-leanstral-1-5-1`).",
        "default_free_samples": [
            "labs-devstral-small-2512",
            "labs-leanstral-1-5-1",
        ],
    },
    "cohere": {
        "display_name": "Cohere",
        "recommended": False,
        "signup_url": "https://dashboard.cohere.com/api-keys",
        "env_vars": ["COHERE_API_KEY"],
        "free_tier_type": "free_models_catalog",
        "summary": "Free Trial API key + $0 models (`north-mini-code-1-0`, `north-small-translate-09-2026`).",
        "default_free_samples": [
            "north-mini-code-1-0",
            "north-small-translate-09-2026",
        ],
    },
    "ollama": {
        "display_name": "Ollama / LM Studio (Local)",
        "recommended": False,
        "signup_url": "https://ollama.com",
        "env_vars": [],
        "free_tier_type": "local_free",
        "summary": "100% free offline/local inference on your machine (no API key required).",
        "default_free_samples": [
            "qwen3-coder:30b",
            "llama3.3:70b",
            "gpt-oss:20b",
        ],
    },
}


def check_free_model_providers(
    config: Optional[Dict[str, Any]] = None,
    *,
    allow_network: bool = True,
) -> List[Dict[str, Any]]:
    """Run a live/cached check across provider catalogs to find all providers offering free models.

    Inspects:
    1. ``models.dev`` registry (`cost.input == 0 and cost.output == 0` or `:free`/`-free` models)
    2. Live/cached OpenRouter model metadata (`fetch_model_metadata`)
    3. Nous Portal `freeRecommendedModels` (`fetch_nous_recommended_models`)
    4. Hermes `provider_model_ids` for `openrouter`, `opencode-zen`, `kilocode`, `opencode-go`, etc.
    5. Built-in heuristic candidate catalogs (`get_candidate_models`)

    Returns a sorted list of provider dicts (recommended providers first, with OpenRouter #1).
    """
    cfg = config if isinstance(config, dict) else load_hermes_config()
    existing_set = set(discover_existing_providers(config=cfg))

    discovered_free_models: Dict[str, List[str]] = {
        pid: [] for pid in FREE_PROVIDER_CATALOG_META
    }

    def _add_free_model(pid: str, model_id: str) -> None:
        if not pid or not model_id:
            return
        bucket = discovered_free_models.setdefault(pid, [])
        if model_id not in bucket:
            bucket.append(model_id)

    # 1. Probe models.dev registry for $0 cost models across Hermes providers
    try:
        import agent.models_dev as md

        reg = md.fetch_models_dev(allow_network=allow_network)
        if isinstance(reg, dict):
            for mdev_id, pdata in reg.items():
                if not isinstance(pdata, dict):
                    continue
                hermes_ids = md._models_dev_to_hermes_ids(mdev_id)
                if not hermes_ids and mdev_id in FREE_PROVIDER_CATALOG_META:
                    hermes_ids = [mdev_id]
                if not hermes_ids:
                    continue
                models_dict = pdata.get("models") or {}
                if not isinstance(models_dict, dict):
                    continue
                for mid, mdata in models_dict.items():
                    if not isinstance(mdata, dict):
                        continue
                    cost = mdata.get("cost")
                    is_zero = (
                        isinstance(cost, dict)
                        and cost.get("input") == 0
                        and cost.get("output") == 0
                    ) or "free" in str(mid).lower()
                    if is_zero:
                        for hid in hermes_ids:
                            canon = normalize_provider_name(hid) or hid
                            _add_free_model(canon, str(mid))
    except Exception as exc:
        logger.debug("models.dev free provider scan skipped: %s", exc)

    # 2. Probe OpenRouter live/cached metadata for :free and $0 models
    try:
        from agent.model_metadata import fetch_model_metadata

        or_meta = fetch_model_metadata(force_refresh=False) if allow_network else {}
        if isinstance(or_meta, dict):
            for mid, info in or_meta.items():
                if not isinstance(info, dict):
                    continue
                pricing = info.get("pricing")
                is_zero = str(mid).endswith(":free") or (
                    isinstance(pricing, dict)
                    and str(pricing.get("prompt")) in ("0", "0.0")
                    and str(pricing.get("completion")) in ("0", "0.0")
                )
                if is_zero and "/" in str(mid):
                    _add_free_model("openrouter", str(mid))
    except Exception as exc:
        logger.debug("OpenRouter metadata scan skipped: %s", exc)

    # 3. Probe Nous Portal freeRecommendedModels & Hermes provider_model_ids
    try:
        import hermes_cli.models as hm

        if allow_network:
            try:
                nous_rec = hm.fetch_nous_recommended_models()
                for entry in (nous_rec or {}).get("freeRecommendedModels", []):
                    name = hm._extract_model_name(entry)
                    if name:
                        _add_free_model("nous", name)
            except Exception:
                pass

        for slug in ("openrouter", "opencode-zen", "opencode-go", "kilocode"):
            try:
                for mid in hm.provider_model_ids(slug):
                    if "free" in str(mid).lower():
                        _add_free_model(slug, str(mid))
            except Exception:
                pass
    except Exception as exc:
        logger.debug("hermes_cli.models free scan skipped: %s", exc)

    # 4. Also check our plugin's own candidate resolver & fallback samples so offline mode works 100%
    for pid, meta in FREE_PROVIDER_CATALOG_META.items():
        try:
            for cand in get_candidate_models(provider=pid):
                if cand.is_free or pid == "gemini":
                    _add_free_model(pid, cand.id)
        except Exception:
            pass
        for sample in meta.get("default_free_samples", []):
            _add_free_model(pid, sample)

    # Assemble structured provider report
    priority_order = [
        "openrouter",
        "gemini",
        "nvidia",
        "opencode-zen",
        "kilocode",
        "nous",
        "ai-gateway",
        "zai",
        "huggingface",
        "opencode-go",
        "alibaba",
        "tencent-tokenhub",
        "commandcode",
        "novita",
        "ollama",
    ]

    all_pids = list(dict.fromkeys([*priority_order, *discovered_free_models.keys()]))
    results: List[Dict[str, Any]] = []

    for pid in all_pids:
        models = discovered_free_models.get(pid, [])
        meta = FREE_PROVIDER_CATALOG_META.get(pid, {})
        if not models and not meta:
            continue
        env_vars = list(meta.get("env_vars") or PROVIDER_ENV_VARS.get(pid, ()))
        configured_envs = [k for k in env_vars if _read_secret_or_env(k)]
        is_auth = pid in existing_set or bool(configured_envs)
        results.append(
            {
                "provider": pid,
                "display_name": meta.get("display_name") or pid.title(),
                "recommended": bool(meta.get("recommended", False)),
                "authenticated": is_auth,
                "available": is_auth,
                "free_model_count": len(models),
                "sample_free_models": models[:6],
                "signup_url": meta.get("signup_url") or "",
                "env_vars": env_vars,
                "env_keys_configured": configured_envs,
                "free_tier_type": meta.get("free_tier_type") or "free_models_catalog",
                "summary": meta.get("summary") or f"{len(models)} free model(s) detected.",
            }
        )

    return results


def is_first_launch(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True if the user has not yet completed or dismissed the first-launch provider setup."""
    cfg = config if isinstance(config, dict) else load_hermes_config()
    plugins_cfg = cfg.get("plugins")
    if isinstance(plugins_cfg, dict):
        entries = plugins_cfg.get("entries")
        if isinstance(entries, dict):
            p_entry = entries.get(PLUGIN_ID)
            if isinstance(p_entry, dict):
                for sub in ("settings", "config"):
                    sdict = p_entry.get(sub)
                    if isinstance(sdict, dict) and sdict.get("first_launch_completed") is True:
                        return False
    return True


def mark_first_launch_done(
    selected_providers: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    save: bool = True,
) -> Dict[str, Any]:
    """Record that first-launch onboarding has been completed and optionally persist chosen providers."""
    updates: Dict[str, Any] = {"first_launch_completed": True}
    if selected_providers:
        provs = list(parse_provider_list(selected_providers))
        if provs:
            updates["providers"] = provs
            updates["provider"] = "existing"
            updates["allow_provider_switch"] = True
    return update_plugin_settings(updates, config=config, save=save)


def format_first_launch_guide(
    providers_report: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the user-facing first-launch guide showing all providers offering free models."""
    report = (
        providers_report
        if providers_report is not None
        else check_free_model_providers(config=config, allow_network=True)
    )
    configured = [p for p in report if p.get("authenticated")]
    recommended = [p for p in report if p.get("recommended")]
    others = [p for p in report if not p.get("recommended")]

    lines: List[str] = [
        "╭──────────────────────────────────────────────────────────────────────────────╮",
        "│  🧭 Model Fallback Heuristic — First-Launch Free Provider Setup Guide        │",
        "╰──────────────────────────────────────────────────────────────────────────────╯",
        "To guarantee your agent always has a working fallback when your primary model is",
        "unavailable, we recommend creating free API keys with 2–3 providers below.",
        "",
        f"Currently configured free-capable providers ({len(configured)}): "
        + (", ".join(p["provider"] for p in configured) if configured else "none yet"),
        "",
        "★ Top Recommended Providers Offering Free Models (Live Catalog Check):",
    ]

    for idx, p in enumerate(recommended, 1):
        status = "✓ CONFIGURED" if p.get("authenticated") else "○ Needs API key"
        env_str = ", ".join(p.get("env_vars") or []) or "(none needed)"
        samples = ", ".join(p.get("sample_free_models", [])[:3])
        lines.append(
            f"  {idx}. {p['display_name']} (`{p['provider']}`) [{status}] — {p['free_model_count']} free/tier models"
        )
        lines.append(f"     • {p['summary']}")
        lines.append(f"     • Sample models: {samples}")
        lines.append(f"     • Get free key:  {p['signup_url']}  (env: `{env_str}`)")

    if others:
        lines.append("")
        lines.append("Additional Providers Offering Free Models / Free Tiers:")
        for idx, p in enumerate(others, len(recommended) + 1):
            status = "✓ CONFIGURED" if p.get("authenticated") else "○ Needs API key"
            env_str = ", ".join(p.get("env_vars") or []) or "local / no key"
            samples = ", ".join(p.get("sample_free_models", [])[:2])
            lines.append(
                f"  {idx}. {p['display_name']} (`{p['provider']}`) [{status}] — "
                f"{p['free_model_count']} free model(s) ({samples})"
            )
            lines.append(f"     • Key URL: {p['signup_url']}  (env: `{env_str}`)")

    lines.extend(
        [
            "",
            "How to choose & activate your providers (no LLM call required):",
            "  • In chat:     `/fallback setup openrouter,gemini,nvidia`",
            "                 (or `/fallback setup` to view this check anytime)",
            "  • In terminal: `hermes fallback-heuristic setup` (interactive picker)",
            "                 `hermes fallback-heuristic set providers=openrouter,gemini,nvidia`",
        ]
    )
    return "\n".join(lines)


def save_provider_api_key(env_var: str, api_key: str) -> bool:
    """Save an API key to ``~/.hermes/.env`` and ``os.environ``."""
    import os

    key_name = str(env_var or "").strip()
    key_val = str(api_key or "").strip()
    if not key_name or not key_val:
        return False

    os.environ[key_name] = key_val
    try:
        from hermes_cli.config import save_env_value

        save_env_value(key_name, key_val)
        return True
    except Exception:
        pass

    try:
        env_path = _get_hermes_home() / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        existing_lines: List[str] = []
        if env_path.is_file():
            existing_lines = env_path.read_text(encoding="utf-8").splitlines()
        updated = False
        new_lines: List[str] = []
        for line in existing_lines:
            stripped = line.strip()
            if stripped.startswith(f"{key_name}=") or stripped.startswith(f"export {key_name}="):
                new_lines.append(f"{key_name}={key_val}")
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            new_lines.append(f"{key_name}={key_val}")
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return True
    except Exception as exc:
        logger.debug("Could not write %s to .env: %s", key_name, exc)
        return False


def run_interactive_first_launch_wizard(
    config: Optional[Dict[str, Any]] = None,
    *,
    allow_network: bool = True,
    prompt_for_keys: bool = True,
) -> Dict[str, Any]:
    """Run an interactive CLI wizard that checks all free-model providers and lets the user decide."""
    cfg = config if isinstance(config, dict) else load_hermes_config()
    report = check_free_model_providers(config=cfg, allow_network=allow_network)
    print("\n" + format_first_launch_guide(report, config=cfg) + "\n")

    if not sys.stdin.isatty():
        return {"ok": True, "interactive": False, "providers_checked": len(report), "free_providers": report}

    try:
        choice = input(
            "  Enter the numbers or IDs of the providers you want in your fallback chain\n"
            "  (e.g. `1,2,3` or `openrouter,gemini,nvidia`, or press Enter to keep current): "
        ).strip()
    except (KeyboardInterrupt, EOFError):
        print("\n  Setup skipped.\n")
        return {"ok": False, "cancelled": True}

    selected: List[str] = []
    if choice:
        for token in choice.replace(" ", ",").split(","):
            tok = token.strip()
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

    saved_keys: List[str] = []
    if selected:
        missing = [
            p
            for p in report
            if p["provider"] in selected and not p.get("authenticated") and p.get("env_vars")
        ]
        if missing and prompt_for_keys:
            print("\n  Optional: Paste API keys now to save to ~/.hermes/.env (or press Enter to set later):")
            for mp in missing:
                env_name = mp["env_vars"][0]
                print(f"    • {mp['display_name']} — get a free key at: {mp['signup_url']}")
                try:
                    entered_key = input(f"      {env_name} (press Enter to skip): ").strip()
                except (KeyboardInterrupt, EOFError):
                    entered_key = ""
                if entered_key:
                    if save_provider_api_key(env_name, entered_key):
                        saved_keys.append(env_name)
                        mp["authenticated"] = True
                        print(f"      ✓ Saved {env_name} to ~/.hermes/.env")

        settings = mark_first_launch_done(selected_providers=selected, config=cfg, save=True)
        print(f"\n  ✓ Saved preferred fallback providers: {', '.join(settings.get('providers', []))}")
        still_missing = [
            p
            for p in missing
            if not p.get("authenticated")
        ]
        if still_missing:
            print("\n  Whenever you're ready, set your API keys for the remaining providers:")
            for mp in still_missing:
                env_name = mp["env_vars"][0]
                print(f"    • {mp['display_name']}: create key at {mp['signup_url']}")
                print(f"      export {env_name}=\"your-api-key-here\"")
        print()
    else:
        mark_first_launch_done(config=cfg, save=True)
        print("\n  ✓ First-launch check marked complete. Run `hermes fallback-heuristic setup` anytime.\n")

    return {
        "ok": True,
        "interactive": True,
        "selected_providers": selected,
        "saved_env_keys": saved_keys,
        "providers_checked": len(report),
        "free_providers": report,
    }
