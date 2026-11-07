"""The command parser must be built with the tier-1 model, not tier 2.

Tier 2 is a cloud model, and the parser handles text the user typed. Devon's
decision, relayed via exocortex-pm 2026-11-07.

It does NOT make the command path local: validate_intent runs first and still
escalates to tier 2 on low confidence, which Devon allowed (reflex-3379).

Asserted on the wiring, because a comment cannot fail.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from reflex.services.bot import ReflexBot
from reflex.services.commands import VALID_ACTIONS

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
    """By KEYWORD, and the keyword is pinned. api_key is the 6th parameter of
    ReflexClassifier, so a positional call would bind to the wrong thing the
    next time either signature grows.
    """
    _, parser = built
    assert parser.call_args.kwargs["api_key"] == "sk-secret"


def test_the_classifier_gets_the_proxy_key(built):
    classifier, _ = built
    assert classifier.call_args.kwargs["api_key"] == "sk-secret"


def test_the_classifier_really_sends_the_header():
    """Through the REAL classifier and the real LLMClient, not the doubles
    above: the wiring tests prove the key is handed over, this proves the
    thing it is handed to does something with it.
    """
    from reflex.services.classifier import ReflexClassifier

    c = ReflexClassifier("http://proxy.test", "t1", "t2", api_key="sk-secret")
    assert c.llm.client.headers["Authorization"] == "Bearer sk-secret"


def test_no_key_configured_sends_no_header():
    """The direct-ollama configuration, which needs no credential."""
    from reflex.services.classifier import ReflexClassifier

    c = ReflexClassifier("http://proxy.test", "t1", "t2")
    assert "Authorization" not in c.llm.client.headers


def test_the_parser_really_sends_the_header():
    """CommandParser holds its OWN httpx client and does not use
    cortex_utils' LLMClient, so being handed the key proves nothing about
    whether it sends one. Without this, deleting the header from its client
    passed every other test here.
    """
    from reflex.services.commands import CommandParser

    p = CommandParser("http://proxy.test", "t1", api_key="sk-secret")
    try:
        assert p.http_client.headers["Authorization"] == "Bearer sk-secret"
    finally:
        p.close()


def test_the_parser_sends_no_header_without_a_key():
    from reflex.services.commands import CommandParser

    p = CommandParser("http://proxy.test", "t1")
    try:
        assert "Authorization" not in p.http_client.headers
    finally:
        p.close()


def test_the_classifier_still_gets_both_tiers(built):
    """Tier 2 is not removed, only moved off the command path."""
    classifier, _ = built
    assert classifier.call_args.args[1] == "local-tier-1"
    assert classifier.call_args.args[2] == "cloud/tier-2"


@pytest.mark.parametrize("key", [None, ""])
def test_the_bot_builds_without_a_key(key):
    """The direct-ollama configuration. LITELLM_API_KEY is optional, so
    construction must not require it -- `os.environ[...]` here would break
    every deployment that reaches a model directly.
    """
    env = dict(ENV)
    if key is None:
        del env["LITELLM_API_KEY"]
    else:
        env["LITELLM_API_KEY"] = key

    with mock.patch.dict(os.environ, env, clear=True):
        with (
            mock.patch("reflex.services.bot.psycopg2.connect"),
            mock.patch("reflex.services.bot.PostgresStorage"),
            mock.patch("reflex.services.bot.MarkdownExporter"),
            mock.patch("reflex.services.bot.ReflexClassifier"),
            mock.patch("reflex.services.bot.CommandParser") as parser,
            mock.patch("reflex.services.bot.AsyncIOScheduler"),
        ):
            ReflexBot()
    assert parser.call_args.kwargs["api_key"] == key


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_a_blank_key_sends_no_header(blank):
    """Documented as blank-is-absent, so it is asserted. A whitespace key
    would otherwise send a bare "Bearer " and earn a 401 that looks like a
    wrong credential rather than a missing one.
    """
    from reflex.services.classifier import ReflexClassifier
    from reflex.services.commands import CommandParser

    c = ReflexClassifier("http://proxy.test", "t1", "t2", api_key=blank)
    assert "Authorization" not in c.llm.client.headers

    p = CommandParser("http://proxy.test", "t1", api_key=blank)
    try:
        assert "Authorization" not in p.http_client.headers
    finally:
        p.close()


def _parser_returning(content: str):
    """A real CommandParser whose HTTP layer returns `content` as the model's
    reply, so parse() is driven end to end rather than stubbed.
    """
    import httpx

    from reflex.services.commands import CommandParser

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    p = CommandParser("http://proxy.test", "t1")
    p.http_client.close()
    p.http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return p


def test_a_bare_string_of_keywords_is_dropped_not_iterated():
    """resolve_target ITERATES target_keywords. A bare string iterates
    character by character, so "idea" becomes four one-letter keywords and
    "i" matches almost any title -- an archive or move landing on the wrong
    entry. Same shape as the set.update("urgent") bug clean_tags prevents.
    """
    p = _parser_returning(
        '{"action": "archive", "target": "that", "target_keywords": "idea"}'
    )
    try:
        parsed = p.parse("archive that idea")
    finally:
        p.http_client.close()
    assert parsed is not None
    assert parsed.target_keywords == []


@pytest.mark.parametrize(
    "body",
    [
        '{"action": "launch", "target": "that"}',
        '{"action": ["move"], "target": "that"}',
        '{"action": "move", "target": ["that"]}',
    ],
)
def test_an_action_outside_the_closed_set_or_a_bad_target_is_refused(body):
    p = _parser_returning(body)
    try:
        assert p.parse("do something") is None
    finally:
        p.http_client.close()


def test_non_string_keywords_are_filtered_out():
    """A list is not enough: resolve_target compares each keyword to a title,
    so a number or a blank among them is junk to drop, not a reason to drop
    the usable ones.
    """
    p = _parser_returning(
        '{"action": "archive", "target": "that", '
        '"target_keywords": ["idea", 5, "   ", "openforge"]}'
    )
    try:
        parsed = p.parse("archive that idea")
    finally:
        p.http_client.close()
    assert parsed is not None
    assert parsed.target_keywords == ["idea", "openforge"]


def test_the_parser_asks_for_json():
    """A local model is likelier than a cloud one to answer with prose or a
    fence. The fence-stripping in parse() stays either way; this is what
    keeps it from being the common path.
    """
    import httpx

    from reflex.services.commands import CommandParser

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen.update(_json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"action": "show", "target": "that"}'}}
                ]
            },
        )

    p = CommandParser("http://proxy.test", "t1")
    p.http_client.close()
    p.http_client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        p.parse("show that")
    finally:
        p.http_client.close()
    assert seen["response_format"] == {"type": "json_object"}


def test_non_object_parameters_are_dropped():
    p = _parser_returning(
        '{"action": "tag", "target": "that", "parameters": ["urgent"]}'
    )
    try:
        parsed = p.parse("tag that urgent")
    finally:
        p.http_client.close()
    assert parsed is not None
    assert parsed.parameters == {}


def test_an_http_fault_in_tier_1_escalates_to_tier_2():
    """An HTTP fault in tier 1 must escalate, not escape.

    The cascade depends on cortex_utils raising an LLMError for a non-2xx; a
    pin bump that reintroduced a raw httpx exception fails here.
    """
    import httpx

    from reflex.services.classifier import ReflexClassifier

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        if '"t1"' in body:
            seen.append("tier1")
            return httpx.Response(401, json={"detail": "no key"})
        seen.append("tier2")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"category": "idea", "confidence": 0.9,'
                            ' "reasoning": "r", "suggested_tags": [],'
                            ' "actionable": false}'
                        }
                    }
                ]
            },
        )

    c = ReflexClassifier("http://proxy.test", "t1", "t2", api_key="k")
    original = c.llm.client
    c.llm.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        result = c.classify("an idea about something")
    finally:
        c.llm.client.close()
        original.close()

    assert seen == ["tier1", "tier2"], "tier 1 must be tried, then escalated"
    assert result.category == "idea"
    assert result.model == "t2", "the tier-2 model answered"


def test_the_closed_set_holds_exactly_the_four_actions():
    """Written out, NOT derived. The acceptance test below parametrizes over
    VALID_ACTIONS, so dropping a member drops its own test case with it and
    the mutant survives -- a set under test cannot also be the test's source
    of truth. execute_command's if/elif chain is the other half of this
    contract, so a member added here needs a branch there.
    """
    assert VALID_ACTIONS == {"move", "tag", "archive", "show"}


@pytest.mark.parametrize("action", sorted(VALID_ACTIONS))
def test_every_action_in_the_closed_set_is_accepted(action):
    """Dropping a member from VALID_ACTIONS used to survive: only "archive"
    and "tag" were ever asserted as accepted, so "move" and "show" could have
    been removed unnoticed. Parametrized over the set itself, so a member
    added later is covered without touching this test.
    """
    p = _parser_returning('{"action": "%s", "target": "that"}' % action)
    try:
        parsed = p.parse("do the thing")
    finally:
        p.http_client.close()
    assert parsed is not None
    assert parsed.action == action


def test_a_non_string_action_is_refused_by_the_type_check():
    """`["move"] not in frozenset` raises TypeError (unhashable), which the
    broad except turns into None -- so the refusal happened for the wrong
    reason and the membership check never fired. An isinstance check now
    precedes it, and this asserts the message rather than just the None.
    """
    from reflex.services import commands

    p = _parser_returning('{"action": ["move"], "target": "that"}')
    with mock.patch.object(commands.logger, "error") as logged:
        try:
            assert p.parse("move that") is None
        finally:
            p.http_client.close()
    assert any("not a string" in str(c) for c in logged.call_args_list), (
        logged.call_args_list
    )


def test_the_bot_sends_no_mentions():
    """execute_move echoes a REJECTED new_category back to Discord verbatim,
    and the rejection path is exactly where an arbitrary model-supplied value
    arrives. Set on the client, so it covers every reply rather than the two
    that are known today.
    """
    with mock.patch.dict(os.environ, ENV, clear=True):
        with (
            mock.patch("reflex.services.bot.psycopg2.connect"),
            mock.patch("reflex.services.bot.PostgresStorage"),
            mock.patch("reflex.services.bot.MarkdownExporter"),
            mock.patch("reflex.services.bot.ReflexClassifier"),
            mock.patch("reflex.services.bot.CommandParser"),
            mock.patch("reflex.services.bot.AsyncIOScheduler"),
        ):
            bot = ReflexBot()
    # Field by field: AllowedMentions has no __eq__, so comparing instances
    # passes for ANY pair -- including a permissive one against none().
    m = bot.allowed_mentions
    assert (m.everyone, m.users, m.roles, m.replied_user) == (False, False, False, False)


def test_a_transport_failure_in_tier_1_also_escalates():
    """_post_completion lets httpx.RequestError through unwrapped -- the six
    public methods of LLMClient convert it, and reflex does not use them. So
    a proxy that is DOWN, as opposed to answering 401, would skip tier 2.
    """
    import httpx

    from reflex.services.classifier import ReflexClassifier

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if '"t1"' in request.content.decode():
            seen.append("tier1")
            raise httpx.ConnectError("proxy is down")
        seen.append("tier2")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"category": "idea", "confidence": 0.9,'
                            ' "reasoning": "r", "suggested_tags": [],'
                            ' "actionable": false}'
                        }
                    }
                ]
            },
        )

    c = ReflexClassifier("http://proxy.test", "t1", "t2", api_key="k")
    original = c.llm.client
    c.llm.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        result = c.classify("an idea")
    finally:
        c.llm.client.close()
        original.close()
    assert seen == ["tier1", "tier2"]
    assert result.model == "t2"


def test_a_transport_failure_in_intent_validation_also_escalates():
    """validate_intent's cascade is a separate handler from classify's, and
    only classify was pinned.
    """
    import httpx

    from reflex.services.classifier import ReflexClassifier

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if '"t1"' in request.content.decode():
            seen.append("tier1")
            raise httpx.ConnectError("proxy is down")
        seen.append("tier2")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"is_command": true, "confidence": 0.9}'}}
                ]
            },
        )

    c = ReflexClassifier("http://proxy.test", "t1", "t2", api_key="k")
    original = c.llm.client
    c.llm.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        is_command, confidence = c.validate_intent("move that idea")
    finally:
        c.llm.client.close()
        original.close()
    assert seen == ["tier1", "tier2"]
    assert is_command is True
