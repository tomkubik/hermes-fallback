"""Comprehensive unit and integration tests for the Hermes Model Fallback Heuristic Plugin."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import cli  # noqa: E402
import config_sync  # noqa: E402
import handlers  # noqa: E402
import heuristics  # noqa: E402
import reasoning  # noqa: E402
from __init__ import register  # noqa: E402
from cli import (  # noqa: E402
    _write_chain,
    cmd_fallback,
    cmd_fallback_clear,
    cmd_fallback_heuristics,
    cmd_fallback_remove,
    setup_cli_parser,
)
from config_sync import (  # noqa: E402
    _iter_fallback_entries,
    build_criteria_from_settings,
    ensure_available_model_or_fallback,
    get_fallback_chain,
    get_plugin_settings,
    get_stored_fallback_rules,
    is_fallback_heuristics_enabled,
    resolve_best_available_model,
    scoped_fallback_chain,
    switch_model_with_heuristic,
    sync_fallback_chain_to_config,
    update_plugin_settings,
)
from handlers import (  # noqa: E402
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
    reset_session_unavailability,
)
from heuristics import (  # noqa: E402
    FallbackCriteria,
    ModelCandidateMetadata,
    _CANDIDATE_CACHE,
    _gather_openrouter_candidates,
    _key_greatest_context,
    _key_largest_params,
    _key_latest,
    clear_candidate_cache,
    discover_existing_providers,
    enrich_model_metadata,
    extract_context_window_from_name,
    extract_date_snapshot,
    extract_param_size_b,
    extract_vendor,
    extract_version_tuple,
    filter_and_rank_candidates,
    get_candidate_models,
    get_provider_status_list,
    is_flash_model,
    is_free_model_slug,
    is_provider_available,
    normalize_provider_name,
    parse_fallback_criteria,
    parse_provider_list,
    resolve_fallback_candidates,
    resolve_fallback_entry,
)
from reasoning import (  # noqa: E402
    parse_reasoning_effort,
    reresolve_fallback_reasoning_config,
    resolve_fallback_entry_reasoning_config,
)


class TestParseFallbackCriteria(unittest.TestCase):
    def test_shorthand_strings(self):
        c1 = parse_fallback_criteria("largest_parameter_count_free")
        self.assertEqual(c1.sort_by, "largest_parameter_count")
        self.assertTrue(c1.free_only)
        self.assertTrue(c1.require_tools)

        c1_alias = parse_fallback_criteria("largest_free")
        self.assertEqual(c1_alias.sort_by, "largest_parameter_count")

        c2 = parse_fallback_criteria("greatest_context_free")
        self.assertEqual(c2.sort_by, "greatest_context")
        self.assertTrue(c2.free_only)

        c3 = parse_fallback_criteria("smallest_free")
        self.assertEqual(c3.sort_by, "smallest")
        self.assertTrue(c3.free_only)

        c4 = parse_fallback_criteria("latest_flash_free")
        self.assertEqual(c4.sort_by, "latest_flash")
        self.assertTrue(c4.flash_only)
        self.assertTrue(c4.free_only)

        c5 = parse_fallback_criteria("latest")
        self.assertEqual(c5.sort_by, "latest")

    def test_auto_and_heuristic_prefixes(self):
        c1 = parse_fallback_criteria("auto:free:largest_parameter_count")
        self.assertEqual(c1.sort_by, "largest_parameter_count")
        self.assertTrue(c1.free_only)

        c1_alias = parse_fallback_criteria("auto:free:largest")
        self.assertEqual(c1_alias.sort_by, "largest_parameter_count")

        c2 = parse_fallback_criteria("heuristic:greatest_context")
        self.assertEqual(c2.sort_by, "greatest_context")

    def test_dict_criteria_with_provider_context_and_params(self):
        data = {
            "criteria": {
                "sort_by": "largest_parameter_count",
                "provider": "anthropic,openrouter,gemini",
                "exclude_providers": ["openai"],
                "allow_provider_switch": True,
                "prefer_current_provider": True,
                "free": False,
                "require_tools": True,
                "flash": True,
                "vendor": "google",
                "filter": "flash",
                "min_context": "128k",
                "max_context": "2m",
                "min_params": "30b",
                "max_params": "405b",
                "max_candidates": 5,
            }
        }
        c = parse_fallback_criteria(data)
        self.assertEqual(c.sort_by, "largest_parameter_count")
        self.assertEqual(c.provider, "anthropic")
        self.assertEqual(c.providers, ("anthropic", "openrouter", "gemini"))
        self.assertEqual(c.exclude_providers, ("openai",))
        self.assertTrue(c.allow_provider_switch)
        self.assertTrue(c.prefer_current_provider)
        self.assertFalse(c.free_only)
        self.assertTrue(c.require_tools)
        self.assertTrue(c.flash_only)
        self.assertEqual(c.vendor, "google")
        self.assertEqual(c.name_filter, "flash")
        self.assertEqual(c.min_context, 131072)
        self.assertEqual(c.max_context, 2000000)
        self.assertEqual(c.min_params, 30.0)
        self.assertEqual(c.max_params, 405.0)
        self.assertEqual(c.max_candidates, 5)

    def test_shorthand_with_provider_context_and_params(self):
        c = parse_fallback_criteria(
            "provider:anthropic,openrouter,gemini min_context:128k min_params:70b greatest_context"
        )
        self.assertEqual(c.providers, ("anthropic", "openrouter", "gemini"))
        self.assertEqual(c.provider, "anthropic")
        self.assertEqual(c.min_context, 131072)
        self.assertEqual(c.min_params, 70.0)
        self.assertEqual(c.sort_by, "greatest_context")
        self.assertFalse(c.free_only)

    def test_describe(self):
        c = FallbackCriteria(
            sort_by="largest_parameter_count",
            providers=("anthropic", "openrouter"),
            min_context=131072,
            min_params=70.0,
            free_only=False,
            flash_only=True,
        )
        desc = c.describe()
        self.assertIn("flash", desc)
        self.assertIn("largest parameter count", desc)
        self.assertIn("providers:anthropic,openrouter", desc)
        self.assertIn("min_ctx:131072", desc)
        self.assertIn("min_params:70b", desc)


class TestHeuristicExtractors(unittest.TestCase):
    def test_extract_param_size_b(self):
        self.assertEqual(extract_param_size_b("meta-llama/llama-3.3-70b-instruct"), 70.0)
        self.assertEqual(extract_param_size_b("openai/gpt-405b"), 405.0)
        self.assertEqual(extract_param_size_b("qwen/qwen-2.5-0.5b"), 0.5)
        self.assertEqual(extract_param_size_b("google/gemini-1.5b"), 1.5)
        self.assertEqual(extract_param_size_b("qwen/qwen-2.5-coder-32b-instruct:free"), 32.0)
        self.assertEqual(extract_param_size_b("mistralai/mixtral-8x7b-instruct"), 56.0)
        self.assertEqual(extract_param_size_b("mistralai/mixtral-8x22b-instruct"), 176.0)
        self.assertEqual(extract_param_size_b("nvidia/nemotron-3-super-120b-a12b:free"), 120.0)
        self.assertEqual(extract_param_size_b("thinkingmachines/inkling-small:free"), 7.0)
        self.assertEqual(extract_param_size_b("poolside/laguna-xs-2.1:free"), 1.0)
        self.assertEqual(extract_param_size_b("anthropic/claude-opus-5"), 100.0)
        self.assertEqual(extract_param_size_b("galaxy-axis-70b"), 70.0)
        self.assertEqual(extract_param_size_b("prompt-model-mini"), 3.0)

    def test_extract_context_window(self):
        self.assertEqual(extract_context_window_from_name("llama-3-70b-128k"), 131072)
        self.assertEqual(extract_context_window_from_name("model-32k"), 32768)
        self.assertEqual(extract_context_window_from_name("model-200k"), 200000)
        self.assertEqual(extract_context_window_from_name("gemini-1m"), 1000000)
        self.assertEqual(extract_context_window_from_name("gemini-2m"), 2000000)
        self.assertEqual(extract_context_window_from_name("unknown-model", default=8192), 8192)

    def test_is_flash_model(self):
        self.assertTrue(is_flash_model("google/gemini-2.0-flash-exp:free"))
        self.assertTrue(is_flash_model("deepseek/deepseek-v4.1-flash"))
        self.assertTrue(is_flash_model("claude-haiku-4-5-20251001"))
        self.assertFalse(is_flash_model("anthropic/claude-opus-5"))
        self.assertFalse(is_flash_model("meta-llama/llama-3.3-70b-instruct:free"))

    def test_is_free_model_slug(self):
        self.assertTrue(is_free_model_slug("meta-llama/llama-3.3-70b-instruct:free"))
        self.assertTrue(is_free_model_slug("stealth/union-alpha"))
        self.assertFalse(is_free_model_slug("anthropic/claude-opus-5"))

    def test_extract_vendor(self):
        self.assertEqual(extract_vendor("google/gemini-2.5-flash"), "google")
        self.assertEqual(extract_vendor("claude-sonnet-4-5-20250929"), "anthropic")
        self.assertEqual(extract_vendor("gpt-4.1"), "openai")
        self.assertEqual(extract_vendor("gemini-2.5-pro"), "google")
        self.assertEqual(extract_vendor("deepseek-chat"), "deepseek")
        self.assertEqual(extract_vendor("bare-model"), "")

    def test_extract_version_and_date(self):
        self.assertEqual(extract_version_tuple("gemini-3.8-flash"), (3, 8))
        self.assertEqual(extract_version_tuple("llama-3.3-70b"), (3, 3))
        self.assertEqual(extract_date_snapshot("deepseek-v4-flash-0731"), 20240731.0)
        self.assertEqual(extract_date_snapshot("model-20241022"), 20241022.0)


class TestMultiProviderDiscoveryAndHeuristics(unittest.TestCase):
    def setUp(self):
        clear_candidate_cache()
        reset_session_unavailability()

    def tearDown(self):
        clear_candidate_cache()
        reset_session_unavailability()

    def test_normalize_provider_aliases(self):
        self.assertEqual(normalize_provider_name("claude"), "anthropic")
        self.assertEqual(normalize_provider_name("google"), "gemini")
        self.assertEqual(normalize_provider_name("or"), "openrouter")
        self.assertEqual(normalize_provider_name("glm"), "zai")
        self.assertEqual(normalize_provider_name("moonshot"), "kimi-coding")
        self.assertEqual(normalize_provider_name("grok"), "xai")
        self.assertEqual(normalize_provider_name("nous-portal"), "nous")
        self.assertEqual(normalize_provider_name("any"), "any")

    def test_discover_existing_providers_from_config_and_env(self):
        cfg = {
            "model": {"provider": "anthropic", "default": "claude-sonnet-4-5-20250929"},
            "custom_providers": [{"name": "my-vllm", "base_url": "http://localhost:8000/v1"}],
        }
        fake_env = {
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "OPENROUTER_API_KEY": "sk-or-test",
            "GEMINI_API_KEY": "gem-test",
            "DEEPSEEK_API_KEY": "ds-test",
            "NOUS_API_KEY": "nous-test",
        }
        with patch.dict(os.environ, fake_env, clear=True):
            provs = discover_existing_providers(config=cfg)
            self.assertEqual(provs[0], "anthropic")
            self.assertIn("openrouter", provs)
            self.assertIn("gemini", provs)
            self.assertIn("deepseek", provs)
            self.assertIn("nous", provs)
            self.assertIn("custom:my-vllm", provs)

            self.assertTrue(is_provider_available("anthropic", config=cfg))
            self.assertTrue(is_provider_available("google", config=cfg))  # alias -> gemini
            self.assertFalse(is_provider_available("xai", config=cfg))

    def test_get_candidate_models_for_any_provider(self):
        for provider_id in (
            "anthropic",
            "openai",
            "gemini",
            "deepseek",
            "zai",
            "kimi-coding",
            "minimax",
            "xai",
            "openrouter",
            "nous",
        ):
            cands = get_candidate_models(provider=provider_id, force_refresh=True)
            self.assertTrue(
                len(cands) > 0, f"Expected builtin/known candidates for provider {provider_id}"
            )
            self.assertTrue(all(c.provider == provider_id for c in cands))

    def test_multi_provider_ranking_and_provider_preference(self):
        cands = [
            ModelCandidateMetadata(
                id="claude-sonnet-4-5-20250929",
                provider="anthropic",
                is_free=False,
                param_size_b=70.0,
                context_window=200000,
                supports_tools=True,
                vendor="anthropic",
            ),
            ModelCandidateMetadata(
                id="gemini-2.5-pro",
                provider="gemini",
                is_free=False,
                param_size_b=120.0,
                context_window=1048576,
                supports_tools=True,
                vendor="google",
            ),
            ModelCandidateMetadata(
                id="qwen/qwen3-coder:free",
                provider="openrouter",
                is_free=True,
                param_size_b=480.0,
                context_window=262144,
                supports_tools=True,
                vendor="qwen",
            ),
        ]

        # 1. Rank by greatest_context across all providers
        c_ctx = FallbackCriteria(
            sort_by="greatest_context",
            providers=("anthropic", "gemini", "openrouter"),
            free_only=False,
        )
        ranked_ctx = filter_and_rank_candidates(cands, c_ctx)
        self.assertEqual(ranked_ctx[0].provider, "gemini")
        self.assertEqual(ranked_ctx[0].id, "gemini-2.5-pro")

        # 2. Rank by largest_parameter_count across all providers
        c_params = FallbackCriteria(
            sort_by="largest_parameter_count",
            providers=("anthropic", "gemini", "openrouter"),
            free_only=False,
        )
        ranked_params = filter_and_rank_candidates(cands, c_params)
        self.assertEqual(ranked_params[0].provider, "openrouter")
        self.assertEqual(ranked_params[0].id, "qwen/qwen3-coder:free")

        # 3. Rank by provider_preference (anthropic > gemini > openrouter)
        c_pref = FallbackCriteria(
            sort_by="provider_preference",
            providers=("anthropic", "gemini", "openrouter"),
            free_only=False,
        )
        ranked_pref = filter_and_rank_candidates(cands, c_pref)
        self.assertEqual(
            [r.provider for r in ranked_pref], ["anthropic", "gemini", "openrouter"]
        )

        # 4. Prefer current provider first when prefer_current_provider=True
        c_cur = FallbackCriteria(
            sort_by="largest_parameter_count",
            providers=("anthropic", "gemini", "openrouter"),
            current_provider="anthropic",
            prefer_current_provider=True,
            free_only=False,
        )
        ranked_cur = filter_and_rank_candidates(cands, c_cur)
        self.assertEqual(ranked_cur[0].provider, "anthropic")

        # 5. Disallow provider switching when allow_provider_switch=False
        c_no_switch = FallbackCriteria(
            sort_by="largest_parameter_count",
            current_provider="anthropic",
            allow_provider_switch=False,
            free_only=False,
        )
        ranked_no_switch = filter_and_rank_candidates(cands, c_no_switch)
        self.assertEqual(len(ranked_no_switch), 1)
        self.assertEqual(ranked_no_switch[0].provider, "anthropic")

        # 6. Exclude provider and min_context / min_params filters
        c_filtered = FallbackCriteria(
            sort_by="largest_parameter_count",
            exclude_providers=("openrouter",),
            min_context=500000,
            min_params=100.0,
            free_only=False,
        )
        ranked_filtered = filter_and_rank_candidates(cands, c_filtered)
        self.assertEqual(len(ranked_filtered), 1)
        self.assertEqual(ranked_filtered[0].id, "gemini-2.5-pro")


class TestZeroLLMCallSettingsSyncAndSwitch(unittest.TestCase):
    def setUp(self):
        clear_candidate_cache()
        reset_session_unavailability()

    def tearDown(self):
        clear_candidate_cache()
        reset_session_unavailability()

    def test_update_plugin_settings_and_sync_chain_without_llm_call(self):
        cfg = {
            "model": {"provider": "anthropic", "default": "claude-opus-4-6"},
        }
        saved_holder = []

        with patch("config_sync.load_hermes_config", return_value=cfg), patch(
            "config_sync.save_hermes_config", side_effect=lambda c: saved_holder.append(c)
        ), patch(
            "config_sync.discover_existing_providers",
            return_value=["anthropic", "gemini", "openrouter"],
        ), patch(
            "heuristics.discover_existing_providers",
            return_value=["anthropic", "gemini", "openrouter"],
        ):
            settings = update_plugin_settings(
                {
                    "provider": "existing",
                    "providers": ["anthropic", "gemini", "openrouter"],
                    "sort_by": "greatest_context",
                    "min_context": "128k",
                    "min_params": "30b",
                    "free_only": False,
                    "max_candidates": 4,
                },
                config=cfg,
                save=True,
            )
            self.assertEqual(settings["sort_by"], "greatest_context")
            self.assertEqual(settings["min_context"], 131072)
            self.assertEqual(settings["min_params"], 30.0)
            self.assertEqual(settings["providers"], ["anthropic", "gemini", "openrouter"])
            self.assertTrue(len(saved_holder) >= 1)

            chain = cfg.get("fallback_providers") or []
            self.assertTrue(len(chain) >= 1)
            # Active primary (anthropic/claude-opus-4-6) must not be duplicated in fallback_providers
            for item in chain:
                self.assertFalse(
                    item["provider"] == "anthropic" and item["model"] == "claude-opus-4-6"
                )

    def test_switch_model_with_heuristic_switches_provider_when_primary_unavailable(self):
        cfg = {
            "model": {"provider": "openai", "default": "gpt-unavailable-999"},
        }
        with patch("config_sync.save_hermes_config"), patch(
            "config_sync.discover_existing_providers", return_value=["gemini", "openrouter"]
        ), patch(
            "heuristics.discover_existing_providers", return_value=["gemini", "openrouter"]
        ), patch(
            "config_sync.is_provider_available",
            side_effect=lambda p, config=None: p in ("gemini", "openrouter"),
        ):
            res = ensure_available_model_or_fallback(config=cfg, save=False)
            self.assertEqual(res["status"], "switched_primary")
            self.assertTrue(res["primary_was_unavailable"])
            self.assertIn(cfg["model"]["provider"], ("gemini", "openrouter"))
            self.assertNotEqual(cfg["model"]["default"], "gpt-unavailable-999")
            self.assertTrue(len(cfg.get("fallback_providers") or []) >= 1)

    def test_slash_commands_work_without_llm_call(self):
        cfg = {
            "model": {"provider": "anthropic", "default": "claude-sonnet-4-5-20250929"},
        }
        with patch("handlers.load_hermes_config", return_value=cfg), patch(
            "config_sync.load_hermes_config", return_value=cfg
        ), patch("config_sync.save_hermes_config"), patch(
            "handlers.discover_existing_providers", return_value=["anthropic", "gemini"]
        ), patch(
            "heuristics.discover_existing_providers", return_value=["anthropic", "gemini"]
        ):
            out_status = handle_slash_command("status")
            self.assertIn("Model Fallback Heuristic Status", out_status)

            out_set = handle_slash_command(
                "set provider=existing sort_by=greatest_context min_context=128k"
            )
            self.assertIn("Updated heuristic settings", out_set)

            out_resolve = handle_slash_command(
                "resolve provider=anthropic,gemini sort_by=greatest_context"
            )
            self.assertIn("Matched", out_resolve)
            self.assertIn("gemini/gemini-2.5-pro", out_resolve)

            out_switch = handle_slash_command("switch provider=gemini sort_by=greatest_context")
            self.assertIn("Switched active model to gemini/", out_switch)
            self.assertEqual(cfg["model"]["provider"], "gemini")

    def test_cli_subcommands_work_without_llm_call(self):
        cfg = {
            "model": {"provider": "anthropic", "default": "claude-sonnet-4-5-20250929"},
        }
        parser = argparse.ArgumentParser(prog="hermes fallback-heuristic")
        setup_cli_parser(parser)

        with patch("cli.load_hermes_config", return_value=cfg), patch(
            "config_sync.load_hermes_config", return_value=cfg
        ), patch("cli.save_hermes_config"), patch("config_sync.save_hermes_config"), patch(
            "heuristics.discover_existing_providers", return_value=["anthropic", "gemini"]
        ), patch(
            "config_sync.discover_existing_providers", return_value=["anthropic", "gemini"]
        ):
            # 1. resolve --json
            args_res = parser.parse_args(
                ["resolve", "--providers", "anthropic,gemini", "--sort-by", "greatest_context", "--json"]
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                cmd_fallback(args_res)
            payload = json.loads(buf.getvalue())
            self.assertGreater(payload["count"], 0)
            self.assertEqual(payload["candidates"][0]["provider"], "gemini")

            # 2. switch --json
            args_sw = parser.parse_args(
                ["switch", "--provider", "gemini", "--sort-by", "greatest_context", "--json"]
            )
            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                cmd_fallback(args_sw)
            sw_payload = json.loads(buf2.getvalue())
            self.assertTrue(sw_payload["ok"])
            self.assertEqual(sw_payload["provider"], "gemini")
            self.assertTrue(sw_payload["provider_switched"])

    def test_hooks_and_middleware_refresh_fallbacks_on_error(self):
        cfg = {
            "model": {"provider": "anthropic", "default": "claude-sonnet-4-5-20250929"},
            "fallback_heuristics": True,
        }
        with patch("handlers.load_hermes_config", return_value=cfg), patch(
            "config_sync.load_hermes_config", return_value=cfg
        ), patch("config_sync.save_hermes_config"), patch(
            "heuristics.discover_existing_providers", return_value=["anthropic", "gemini"]
        ), patch(
            "config_sync.discover_existing_providers", return_value=["anthropic", "gemini"]
        ), patch(
            "config_sync.is_provider_available", return_value=True
        ):
            on_session_start_hook()
            self.assertTrue(len(cfg.get("fallback_providers") or []) > 0)

            # Simulate auth/quota failure on anthropic via api_request_error_hook
            api_request_error_hook(
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
                reason="auth",
                status_code=401,
            )
            # Re-synced fallback chain must exclude anthropic and route to gemini!
            chain_after = cfg.get("fallback_providers") or []
            self.assertTrue(len(chain_after) > 0)
            self.assertTrue(all(e["provider"] != "anthropic" for e in chain_after))
            self.assertEqual(chain_after[0]["provider"], "gemini")

            # Test llm_request_middleware stripping reasoning when reasoning_effort="none"
            cfg["fallback_providers"][0]["reasoning_effort"] = "none"
            active_fb_model = cfg["fallback_providers"][0]["model"]
            req = {
                "model": active_fb_model,
                "reasoning_effort": "high",
                "extra_body": {"reasoning": {"effort": "high"}},
            }
            rewritten = llm_request_middleware(req, SimpleNamespace(model=active_fb_model))
            self.assertNotIn("reasoning_effort", rewritten)
            self.assertEqual(rewritten["extra_body"]["reasoning"], {"enabled": False})


class TestReviewFindingsF1ToF11AndN1ToN9(unittest.TestCase):
    """Preserve all F1–F11 and N1–N9 invariants from the Hermes Core PR."""

    def setUp(self):
        clear_candidate_cache()

    def tearDown(self):
        clear_candidate_cache()

    def test_f1_hard_filters_return_empty_on_no_match(self):
        paid_candidate = ModelCandidateMetadata(
            id="lab/paid-405b", provider="nous", is_free=False, supports_tools=True
        )
        with patch("heuristics.get_candidate_models", return_value=[paid_candidate]):
            self.assertEqual(
                resolve_fallback_entry({"provider": "nous", "criteria": {"free_only": True}}),
                [],
            )

        other_vendor = ModelCandidateMetadata(
            id="lab/model-a", provider="nous", vendor="lab", is_free=True, supports_tools=True
        )
        with patch("heuristics.get_candidate_models", return_value=[other_vendor]):
            self.assertEqual(
                resolve_fallback_entry(
                    {"provider": "nous", "criteria": {"vendor": "google", "free_only": True}}
                ),
                [],
            )

        no_tools = ModelCandidateMetadata(
            id="lab/no-tools", provider="nous", is_free=True, supports_tools=False
        )
        with patch("heuristics.get_candidate_models", return_value=[no_tools]):
            self.assertEqual(
                resolve_fallback_entry(
                    {"provider": "nous", "criteria": {"require_tools": True, "free_only": True}}
                ),
                [],
            )

    def test_f2_cmd_fallback_remove_preserves_stored_inactive_heuristics(self):
        initial_config = {
            "fallback_heuristics": False,
            "fallback_providers": [
                {"provider": "nous", "criteria": {"free_only": True}},
                {"provider": "openai", "model": "gpt-4o-mini"},
            ],
        }
        saved_configs = []
        with patch("cli.load_hermes_config", side_effect=lambda: dict(initial_config)), patch(
            "cli.save_hermes_config", side_effect=lambda c: saved_configs.append(c)
        ):
            cmd_fallback_remove(SimpleNamespace(index=2))
        self.assertEqual(len(saved_configs), 1)
        self.assertEqual(
            saved_configs[0]["fallback_providers"],
            [{"provider": "nous", "criteria": {"free_only": True}}],
        )

    def test_f3_normalization_idempotent_materialized_entries(self):
        candidates = [
            ModelCandidateMetadata(
                id=f"lab/model-{i}", provider="nous", is_free=True, supports_tools=True
            )
            for i in range(1, 4)
        ]
        with patch("heuristics.get_candidate_models", return_value=candidates):
            cfg = {
                "fallback_heuristics": True,
                "fallback_providers": [{"heuristic": "free", "max_candidates": 3}],
            }
            pass1 = get_fallback_chain(cfg)
            self.assertEqual(len(pass1), 3)
            pass2 = _iter_fallback_entries(pass1, config=cfg)
            self.assertEqual(len(pass2), 3)
            self.assertEqual([e["model"] for e in pass1], [e["model"] for e in pass2])

    def test_f4_f5_f7_f11_reasoning_effort_precedence(self):
        cfg = {
            "agent": {
                "reasoning_effort": "high",
                "reasoning_overrides": {
                    "lab/model-30b:free": "none",
                },
            }
        }
        # F4: absent effort on heuristic route respects per-model override
        entry = {"model": "lab/model-30b:free", "provider": "nous", "_is_heuristic": True}
        self.assertEqual(
            resolve_fallback_entry_reasoning_config(cfg, "lab/model-30b:free", entry),
            {"enabled": False},
        )

        # F5: YAML false preserved
        agent = SimpleNamespace(model="stepfun/step-3.7-flash:free", reasoning_config=None)
        reresolve_fallback_reasoning_config(agent, {"reasoning_effort": False})
        self.assertEqual(agent.reasoning_config, {"enabled": False})

        # F11: static fallback inherits global effort; heuristic route defaults to native
        static_entry = {"provider": "openrouter", "model": "google/gemini-2.5-flash"}
        self.assertEqual(
            resolve_fallback_entry_reasoning_config(cfg, "google/gemini-2.5-flash", static_entry),
            {"enabled": True, "effort": "high"},
        )
        heur_entry = {
            "provider": "nous",
            "model": "stepfun/step-3.7-flash:free",
            "_is_heuristic": True,
        }
        self.assertEqual(
            resolve_fallback_entry_reasoning_config(cfg, "stepfun/step-3.7-flash:free", heur_entry),
            {"native": True},
        )

    def test_n3_n4_n5_n6_n7_n8_n9(self):
        # N3: supports_tools defaults False, unknown context window is 0
        cand = ModelCandidateMetadata(id="unknown/unseen-model")
        self.assertFalse(cand.supports_tools)
        meta = enrich_model_metadata("completely-unknown-custom-slug-9999", provider="custom")
        self.assertEqual(meta.context_window, 0)
        self.assertGreater(
            _key_greatest_context(ModelCandidateMetadata(id="known", context_window=200000)),
            _key_greatest_context(ModelCandidateMetadata(id="unknown", context_window=0)),
        )

        # N4: stored rules & _write_chain preserve fallback_providers and fallback_model
        cfg = {
            "fallback_providers": [{"provider": "p1", "model": "m1"}],
            "fallback_model": [{"provider": "p2", "model": "m2"}],
        }
        rules = get_stored_fallback_rules(cfg)
        self.assertEqual(len(rules), 2)
        _write_chain(cfg, rules)
        self.assertEqual(cfg["fallback_providers"], [{"provider": "p1", "model": "m1"}])
        self.assertEqual(cfg["fallback_model"], [{"provider": "p2", "model": "m2"}])

        # N5: empty OpenRouter catalog returns empty dict
        with patch("heuristics._load_openrouter_live_catalog", return_value={}), patch(
            "heuristics._load_openrouter_static_catalog", return_value={}
        ), patch("heuristics._load_openrouter_disk_catalog", return_value={}):
            self.assertEqual(_gather_openrouter_candidates("openrouter"), {})

        # N6: disabled static entry pops criteria & checks provenance
        entry = {"provider": "nous", "model": "static-model", "criteria": {"free_only": True}}
        cfg_dis = {"fallback_heuristics": False, "agent": {"reasoning_effort": "high"}}
        entries = _iter_fallback_entries([entry], config=cfg_dis)
        self.assertEqual(len(entries), 1)
        self.assertNotIn("criteria", entries[0])

        # N7: scoped_fallback_chain inherits config heuristics flag
        declared = [{"provider": "nous", "criteria": {"free_only": True}}]
        cand_match = ModelCandidateMetadata(
            id="lab/free-match", provider="nous", is_free=True, supports_tools=True
        )
        with patch("heuristics.get_candidate_models", return_value=[cand_match]):
            chain = scoped_fallback_chain(
                None,
                declared,
                pinned=False,
                owner="subagent",
                config={"fallback_heuristics": True},
            )
            self.assertIsNotNone(chain)
            self.assertEqual(chain[0]["model"], "lab/free-match")

        # N8: negative caching
        clear_candidate_cache()
        with patch("heuristics._gather_generic_candidates", return_value={}):
            res = get_candidate_models(provider="empty-provider", force_refresh=False)
            self.assertEqual(res, [])
            self.assertTrue(any("empty-provider" in k for k in _CANDIDATE_CACHE))

        # N9: unknown size sorting & cross-vendor latest
        self.assertGreater(
            _key_largest_params(ModelCandidateMetadata(id="unknown-size", param_size_b=None)),
            _key_largest_params(ModelCandidateMetadata(id="nano-0.5b", param_size_b=0.5)),
        )
        self.assertEqual(
            _key_latest(ModelCandidateMetadata(id="vendor-a/model-v9", version_tuple=(9, 0)))[:3],
            (0.0, 0.0, (0,)),
        )


class TestHermesPluginRegistrationAndValidation(unittest.TestCase):
    def test_register_wires_all_declared_capabilities(self):
        tools = []
        hooks = []
        middlewares = []
        commands = []
        cli_commands = []
        prompt_sections = []

        ctx = SimpleNamespace(
            register_tool=lambda **kw: tools.append(kw["name"]),
            register_hook=lambda name, cb: hooks.append(name),
            register_middleware=lambda kind, cb: middlewares.append(kind),
            register_command=lambda **kw: commands.append(kw["name"]),
            register_cli_command=lambda **kw: cli_commands.append(kw["name"]),
            register_system_prompt_section=lambda **kw: prompt_sections.append(kw["id"]),
        )
        register(ctx)
        self.assertEqual(
            set(tools),
            {
                "fallback_heuristic_resolve",
                "fallback_heuristic_switch",
                "fallback_heuristic_configure",
                "fallback_heuristic_providers",
            },
        )
        self.assertEqual(set(hooks), {"on_session_start", "pre_llm_call", "api_request_error"})
        self.assertEqual(set(middlewares), {"llm_request", "llm_execution"})
        self.assertIn("fallback", commands)
        self.assertIn("fallback-heuristic", commands)
        self.assertIn("fallback-heuristic", cli_commands)
        self.assertEqual(prompt_sections, ["model_fallback_heuristic"])


class TestFirstLaunchOnboardingAndFreeProviderCheck(unittest.TestCase):
    def test_check_free_model_providers_lists_openrouter_and_others(self):
        from onboarding import (
            check_free_model_providers,
            format_first_launch_guide,
            is_first_launch,
            mark_first_launch_done,
        )

        cfg = {"model": {"provider": "anthropic", "default": "claude-sonnet-4-5-20250929"}}
        report = check_free_model_providers(config=cfg, allow_network=False)
        self.assertTrue(len(report) >= 10)
        provider_ids = [p["provider"] for p in report]
        # OpenRouter must be #1 recommended alongside Gemini, NVIDIA, OpenCode Zen, KiloCode, Nous, etc.
        self.assertEqual(provider_ids[0], "openrouter")
        for expected in (
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
            "ollama",
        ):
            self.assertIn(expected, provider_ids)

        guide = format_first_launch_guide(report, config=cfg)
        self.assertIn("First-Launch Free Provider Setup Guide", guide)
        self.assertIn("OpenRouter (`openrouter`)", guide)
        self.assertIn("https://openrouter.ai/keys", guide)
        self.assertIn("OPENROUTER_API_KEY", guide)

        self.assertTrue(is_first_launch(cfg))
        with patch("config_sync.save_hermes_config"):
            mark_first_launch_done(
                selected_providers=["openrouter", "gemini", "nvidia"], config=cfg, save=False
            )
        self.assertFalse(is_first_launch(cfg))
        settings = get_plugin_settings(cfg)
        self.assertEqual(settings["providers"], ["openrouter", "gemini", "nvidia"])

    def test_slash_and_cli_setup_and_free_providers(self):
        cfg = {"model": {"provider": "anthropic", "default": "claude-sonnet-4-5-20250929"}}
        with patch("handlers.load_hermes_config", return_value=cfg), patch(
            "config_sync.load_hermes_config", return_value=cfg
        ), patch("onboarding.load_hermes_config", return_value=cfg), patch(
            "config_sync.save_hermes_config"
        ):
            # 1. On first launch, /fallback status includes the first-launch guide
            status_out = handle_slash_command("status")
            self.assertIn("First-Launch Free Provider Setup Guide", status_out)
            self.assertIn("OpenRouter (`openrouter`)", status_out)

            # 2. /fallback setup openrouter,gemini,nvidia selects providers and marks first launch done
            setup_out = handle_slash_command("setup openrouter,gemini,nvidia")
            self.assertIn("Saved preferred fallback providers: openrouter, gemini, nvidia", setup_out)

            # 3. Subsequent /fallback status no longer shows the first-launch banner
            status_after = handle_slash_command("status")
            self.assertNotIn("First-Launch Free Provider Setup Guide", status_after)

            # 4. /fallback free-providers still displays the live check on demand
            free_out = handle_slash_command("free-providers")
            self.assertIn("OpenRouter (`openrouter`)", free_out)


if __name__ == "__main__":
    unittest.main()
