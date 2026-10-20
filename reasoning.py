"""Reasoning effort resolution and notice helpers for model fallback routes."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_VALID_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
)
VALID_REASONING_EFFORTS = _VALID_REASONING_EFFORTS


def parse_reasoning_effort(raw: Any) -> Optional[Dict[str, Any]]:
    """Parse a reasoning effort value into a normalized reasoning_config dict."""
    try:
        from hermes_constants import parse_reasoning_effort as _core_parse

        res = _core_parse(raw)
        if res is not None or raw in (None, "", "default", "auto", "native"):
            return res
    except Exception:
        pass

    if raw is None:
        return None
    if isinstance(raw, dict):
        if raw.get("native") is True:
            return {"native": True}
        if "enabled" in raw:
            return dict(raw)
        return None
    if isinstance(raw, bool):
        return {"enabled": False} if not raw else None
    val = str(raw).strip().lower()
    if not val or val in ("default", "auto", "native"):
        return None
    if val in ("none", "off", "false", "disable", "disabled", "0"):
        return {"enabled": False}
    if val in _VALID_REASONING_EFFORTS:
        return {"enabled": True, "effort": val}
    return None


def resolve_per_model_reasoning_effort(
    model: str, overrides: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Resolve per-model reasoning effort override from agent.reasoning_overrides."""
    try:
        from hermes_constants import resolve_per_model_reasoning_effort as _core_resolve

        return _core_resolve(model, overrides)
    except Exception:
        pass

    if not model or not isinstance(overrides, dict) or not overrides:
        return None
    m_lower = model.lower().strip()
    bare = m_lower.split("/", 1)[-1]
    raw_val = None
    for k, v in overrides.items():
        k_lower = str(k).lower().strip()
        if k_lower == m_lower or k_lower == bare:
            raw_val = v
            break
    if raw_val is None:
        return None
    if isinstance(raw_val, str) and raw_val.strip().lower() in ("default", "auto", "native"):
        return {"native": True}
    return parse_reasoning_effort(raw_val)


def _extract_entry_reasoning_effort(fb: Dict[str, Any]) -> Any:
    """Extract reasoning effort from fallback entry or its criteria, preserving False and None."""
    for key in ("reasoning_effort", "thinking_effort", "thinking", "reasoning"):
        if fb.get(key) is not None:
            return fb[key]
    criteria = fb.get("criteria")
    if isinstance(criteria, dict):
        for key in ("reasoning_effort", "thinking_effort", "thinking", "reasoning"):
            if criteria.get(key) is not None:
                return criteria[key]
    return None


def resolve_fallback_entry_reasoning_config(
    cfg: Optional[Dict[str, Any]],
    model: str,
    fallback_entry: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve reasoning_config for a fallback route.

    Precedence:
    1. Explicit reasoning_effort in fallback_entry or its criteria (e.g. "default", "none", "low", False).
       "default"/"auto"/"native" maps to {"native": True} so the model's native default applies.
       False/"none" maps to {"enabled": False}.
    2. Per-model override from config.yaml (agent.reasoning_overrides) for this specific fallback model.
    3. Provenance-based fallback:
       - Heuristic route: defaults to provider/model native default ({"native": True}), avoiding
         forcing a primary model's global high effort onto an arbitrary fallback.
       - Static route: preserves base Hermes behavior, inheriting global agent.reasoning_effort.
    """
    fb = fallback_entry or {}
    entry_effort = _extract_entry_reasoning_effort(fb)
    if entry_effort is not None:
        val_str = str(entry_effort).strip().lower()
        if val_str in ("default", "auto", "native"):
            return {"native": True}
        parsed = parse_reasoning_effort(entry_effort)
        if parsed is not None:
            return parsed

    cfg_dict = cfg if isinstance(cfg, dict) else {}
    agent_cfg = cfg_dict.get("agent") if isinstance(cfg_dict.get("agent"), dict) else {}
    overrides = agent_cfg.get("reasoning_overrides") or {}
    per_model = resolve_per_model_reasoning_effort(model, overrides)
    if per_model is not None:
        return per_model

    is_heuristic = bool(fb.get("_is_heuristic") or fb.get("criteria_matched"))
    if is_heuristic:
        return {"native": True}

    global_effort = agent_cfg.get("reasoning_effort", "")
    return parse_reasoning_effort(global_effort)


def reresolve_fallback_reasoning_config(
    agent: Any, fallback_entry: Optional[Dict[str, Any]] = None
) -> None:
    """Resolve and update reasoning_config on agent for an activated fallback model."""
    agent._reasoning_effort_rejected = False
    agent._reasoning_disable_rejected = False
    agent._reasoning_floor_required = False

    try:
        try:
            from hermes_cli.config import load_config

            cfg = load_config() or {}
        except Exception:
            cfg = {}
        resolved = resolve_fallback_entry_reasoning_config(cfg, agent.model, fallback_entry)
        agent.reasoning_config = resolved
        logger.info("Fallback %s: reasoning_config resolved: %s", agent.model, resolved)
    except Exception as _reasoning_err:
        logger.debug(
            "Failed to resolve reasoning_config for fallback %s; keeping current: %s",
            agent.model,
            _reasoning_err,
            exc_info=True,
        )


def format_fallback_label(entry: Dict[str, Any]) -> str:
    """Format a fallback chain entry for user-facing startup logs."""
    provider = entry.get("provider") or "?"
    model = entry.get("model") or "?"
    lbl = f"{model} ({provider})"
    matched = entry.get("criteria_matched")
    rank = entry.get("criteria_rank")
    if matched and rank:
        lbl += f" [#{rank} {matched}]"
    elif matched:
        lbl += f" [{matched}]"
    return lbl


def format_fallback_notice(
    old_model: str,
    old_provider: str,
    fb_model: str,
    fb_provider: str,
    reason: Any,
    fb: Dict[str, Any],
    cooldown_seconds: Optional[int] = None,
    remaining_seconds: int = 0,
) -> str:
    """Format user-visible one-shot notice when falling back during a turn."""
    try:
        from agent.chat_completion_helpers import _fallback_reason_text

        reason_str = _fallback_reason_text(reason)
    except Exception:
        reason_str = str(getattr(reason, "value", None) or reason or "provider failure").replace(
            "_", " "
        )
    criteria_info = ""
    if fb.get("criteria_matched"):
        rank_str = f" #{fb['criteria_rank']}" if fb.get("criteria_rank") else ""
        criteria_info = f" [matched heuristic: {fb['criteria_matched']}{rank_str}]"
    notice = (
        f"⚠️ Model fallback: {old_model} via {old_provider} unavailable ({reason_str}); "
        f"using {fb_model} via {fb_provider}{criteria_info}."
    )
    if cooldown_seconds is not None:
        notice += f" Primary retry eligible in ~{remaining_seconds} s; recovery is not guaranteed."
    return notice
