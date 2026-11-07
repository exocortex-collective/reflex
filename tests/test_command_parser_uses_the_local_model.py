"""The command parser must be built with the TIER 1 model, not tier 2.

Tier 2 is a cloud model. Building the parser with it sent every message that
merely LOOKED like a command off the estate, with no local attempt first --
and command text is capture text, whatever Devon typed. Ruled by Devon
2026-11-07. Tier 2 remains the escalation for CLASSIFICATION only.

Asserted on the wiring rather than on a comment, because a comment cannot
fail.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from reflex.services.bot import ReflexBot

ENV = {
    "DISCORD_REFLEX_CHANNEL_ID": "1",
    "REFLEX_WEBHOOK_TOKEN": "t",
    "POSTGRES_HOST": "h",
    "POSTGRES_DB": "d",
    "POSTGRES_USER": "u",
    "POSTGRES_PASSWORD": "p",
    "DUCKDB_API_URL": "http://duck.test",
    "LITELLM_BASE_URL": "http://proxy.test",
    "REFLEX_LLM_TIER1_MODEL": "local-tier-1",
    "REFLEX_LLM_TIER2_MODEL": "cloud/tier-2",
    "LITELLM_API_KEY": "sk-secret",
}


@pytest.fixture
def built():
    """Construct the bot with every outbound dependency doubled."""
    with mock.patch.dict(os.environ, ENV, clear=True):
        with (
            mock.patch("reflex.services.bot.psycopg2.connect"),
            mock.patch("reflex.services.bot.PostgresStorage"),
            mock.patch("reflex.services.bot.MarkdownExporter"),
            mock.patch("reflex.services.bot.ReflexClassifier") as classifier,
            mock.patch("reflex.services.bot.CommandParser") as parser,
            mock.patch("reflex.services.bot.AsyncIOScheduler"),
        ):
            ReflexBot()
            yield classifier, parser


def test_the_parser_is_built_with_the_tier_1_model(built):
    _, parser = built
    url, model, *rest = parser.call_args.args
    assert model == "local-tier-1"
    assert model != "cloud/tier-2"


def test_the_parser_gets_the_proxy_key(built):
    _, parser = built
    assert "sk-secret" in parser.call_args.args


def test_the_classifier_still_gets_both_tiers(built):
    """Tier 2 is not removed, only moved off the command path."""
    classifier, _ = built
    assert classifier.call_args.args[1] == "local-tier-1"
    assert classifier.call_args.args[2] == "cloud/tier-2"
