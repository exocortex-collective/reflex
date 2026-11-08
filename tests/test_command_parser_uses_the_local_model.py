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
    """Asserted written out, not derived from VALID_ACTIONS -- which this PR
    made derived, frozenset(get_args(CommandAction)).

    The acceptance test below parametrizes over
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


# ---------------------------------------------------------------------------
# Round 3. The tier-2 leg, the fields the fix batch guards, and the pings the
# client-level AllowedMentions.none() silently removed.
# ---------------------------------------------------------------------------

GOOD_CLASSIFICATION = (
    '{{"category": "idea", "confidence": {c}, "reasoning": "r",'
    ' "suggested_tags": [], "actionable": false}}'
)


def _classifier_on(handler, **kw):
    """A ReflexClassifier with `handler` as its transport and a real cascade.

    Only the socket is doubled, so `_post_completion` and both `except`
    clauses are the real ones -- the thing under test is never the stub.
    """
    import httpx

    from reflex.services.classifier import ReflexClassifier

    c = ReflexClassifier("http://proxy.test", "t1", "t2", api_key="k", **kw)
    c.llm.client.close()
    c.llm.client = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def _ok(content: str):
    import httpx

    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_a_tier_2_transport_failure_in_intent_validation_is_an_llm_error():
    """The tier-1 handler does not cover the tier-2 leg, and a proxy that is
    down kills both -- so `Raises: LLMError` was false for exactly the
    condition that handler exists to survive. bot.py catches Exception, so
    this is the contract being wrong rather than a crash.
    """
    import httpx
    from cortex_utils.llm import LLMError

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append("t1" if '"t1"' in request.content.decode() else "t2")
        raise httpx.ConnectError("proxy is down")

    c = _classifier_on(handler)
    try:
        with pytest.raises(LLMError):
            c.validate_intent("move that to project")
    finally:
        c.llm.client.close()
    assert seen == ["t1", "t2"], "both legs must be attempted"


def test_a_tier_2_failure_keeps_the_usable_tier_1_classification():
    """0.65 escalates (< tier1_threshold 0.7) but is storable (>= 0.6, the
    bar handle_capture actually gates on), so letting tier 2's failure
    propagate threw away a result that would have been filed normally -- the
    capture became "Sorry, something went wrong" with nothing retrying.
    """
    import httpx

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if '"t1"' in request.content.decode():
            seen.append("t1")
            return _ok(GOOD_CLASSIFICATION.format(c=0.65))
        seen.append("t2")
        raise httpx.ConnectError("proxy died between the two calls")

    c = _classifier_on(handler)
    try:
        result = c.classify("an idea about something")
    finally:
        c.llm.client.close()
    assert seen == ["t1", "t2"], "tier 2 must still be attempted"
    assert result.model == "t1", "the tier-1 answer must survive tier 2 failing"
    assert result.confidence == 0.65


def test_both_tiers_failing_classification_is_still_an_llm_error():
    """The fallback above must not swallow a total failure: with no tier-1
    result there is nothing to keep, and the caller must hear about it.
    """
    import httpx
    from cortex_utils.llm import LLMError

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("proxy is down")

    c = _classifier_on(handler)
    try:
        with pytest.raises(LLMError):
            c.classify("an idea")
    finally:
        c.llm.client.close()


def test_a_non_string_reasoning_is_dropped():
    """`reasoning` was the one model-supplied field with no type check, and it
    reaches a Postgres %s for a `text` column. psycopg2 ADAPTS a list to
    ARRAY[...] rather than refusing it, so a non-string got to the server.
    """
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok(
            '{"category": "idea", "confidence": 0.95,'
            ' "reasoning": ["a", "b"], "suggested_tags": [], "actionable": false}'
        )

    c = _classifier_on(handler)
    try:
        result = c.classify("an idea")
    finally:
        c.llm.client.close()
    assert result.category == "idea", "the rest of the answer is still used"
    assert result.reasoning == "", "a non-string reasoning must not reach storage"


def test_a_non_object_parse_body_is_refused_by_name():
    """`in` on a string is substring containment, so a bare JSON string passed
    the presence test and then raised TypeError on parsed["action"] -- into
    the blanket except, which refuses for an unnamed reason.
    """
    import httpx

    from reflex.services.commands import CommandParser

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok('"action and target"')

    p = CommandParser("http://proxy.test", "t1", api_key="k")
    p.http_client.close()
    p.http_client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with mock.patch("reflex.services.commands.logger") as log:
            assert p.parse("move that to project") is None
        said = " ".join(str(c) for c in log.error.call_args_list)
    finally:
        p.http_client.close()
    assert "not an object" in said, f"the refusal must name the shape: {said}"


@pytest.mark.asyncio
async def test_a_non_string_move_category_is_refused():
    """Forward guard. VALID_CATEGORIES is a list here, so an unhashable value
    is harmless today -- but reflex-791e is open to make it a frozenset, and
    `["project"] not in frozenset` raises TypeError into execute_command's
    broad except. The isinstance keeps that cleanup from reintroducing the
    bug the sibling fix just removed from commands.py.
    """
    from reflex.services.commands import ParsedCommand

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

    message = mock.MagicMock()
    message.reply = mock.AsyncMock()
    entry = mock.MagicMock()
    entry.category = "idea"
    parsed = ParsedCommand(
        action="move",
        target="that",
        target_keywords=[],
        parameters={"new_category": ["project"]},
    )

    # The set is a FROZENSET here, which is what reflex-791e will make it.
    # With a list, `["project"] not in [...]` is a harmless False and the
    # isinstance guard cannot be killed -- the test would pass whether or not
    # the guard exists, which is no test at all.
    with mock.patch.object(
        ReflexBot, "VALID_CATEGORIES", frozenset(ReflexBot.VALID_CATEGORIES)
    ):
        await bot.execute_move(message, entry, parsed)

    message.reply.assert_awaited_once()
    assert "Invalid category" in message.reply.await_args.args[0]
    assert entry.category == "idea", "the entry must not be moved"
    bot.storage.update_entry.assert_not_called()


def test_ping_only_allows_exactly_one_user():
    """An allow-list, not a switch. `users=True` would re-enable every mention
    in the content, and execute_move echoes a model-supplied value verbatim.
    """
    import discord

    from reflex.services.bot import _ping_only

    user = discord.Object(id=509879976879718411)
    m = _ping_only(user).to_dict()

    assert m.get("users") == [509879976879718411]
    assert "everyone" not in m.get("parse", []), "an echoed @everyone must not ping"
    assert "roles" not in m.get("parse", []), "an echoed role mention must not ping"
    assert "users" not in m.get("parse", []), "only the listed id, not all users"


def _bot_with_doubles():
    with mock.patch.dict(os.environ, ENV, clear=True):
        with (
            mock.patch("reflex.services.bot.psycopg2.connect"),
            mock.patch("reflex.services.bot.PostgresStorage"),
            mock.patch("reflex.services.bot.MarkdownExporter"),
            mock.patch("reflex.services.bot.ReflexClassifier"),
            mock.patch("reflex.services.bot.CommandParser"),
            mock.patch("reflex.services.bot.AsyncIOScheduler"),
        ):
            return ReflexBot()


@pytest.mark.asyncio
async def test_the_custom_snooze_prompt_still_pings_the_user():
    """The regression this catches: AllowedMentions.none() on the client also
    sets users=False and replied_user=False, and this prompt then WAITS in
    snooze_pending for a reply -- so it arrived with no notification at all
    and the entry stayed unsnoozed. No test touched this handler before.
    """
    import discord

    bot = _bot_with_doubles()

    user = mock.MagicMock(spec=discord.User)
    user.id = 509879976879718411
    user.mention = "<@509879976879718411>"
    reaction = mock.MagicMock()
    reaction.message.id = 77
    reaction.message.reply = mock.AsyncMock(return_value=mock.MagicMock(id=88))

    await bot.handle_snooze_entry_custom(reaction, user, entry_id=5)

    kwargs = reaction.message.reply.await_args.kwargs
    assert "allowed_mentions" in kwargs, "the prompt must be allowed to notify"
    assert kwargs["allowed_mentions"].to_dict().get("users") == [user.id]
    assert (88, user.id) in bot.snooze_pending, "the reply it waits for is keyed here"


def test_every_deliberate_mention_passes_an_allow_list():
    """A pattern guard, so a NEW reply that pings is not silently muted by the
    client default. Source-level on purpose: the defect was a send site that
    no test drove, so the thing to pin is that no such site can be added
    without an allowed_mentions of its own.

    Walked with ast rather than matched with a regex -- a regex over source
    missed the snooze confirmation, whose message contains a parenthesised
    strftime call, and a guard that silently finds fewer sites than exist is
    the same class of defect it is meant to catch.
    """
    import ast
    import inspect

    from reflex.services import bot as bot_module

    tree = ast.parse(inspect.getsource(bot_module))

    def mentions_user(call: ast.Call) -> bool:
        return any(
            isinstance(n, ast.Attribute)
            and n.attr == "mention"
            and isinstance(n.value, ast.Name)
            and n.value.id == "user"
            for arg in call.args
            for n in ast.walk(arg)
        )

    sends = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"reply", "send"}
    ]
    mentioning = [c for c in sends if mentions_user(c)]
    assert mentioning, "the walk must still find the mentioning sends"

    missing = [
        c.lineno
        for c in mentioning
        if not any(k.arg == "allowed_mentions" for k in c.keywords)
    ]
    assert not missing, (
        f"sends at lines {missing} interpolate user.mention with no "
        "allowed_mentions allow-list, so the client default mutes them"
    )
    assert len(mentioning) == 3, (
        f"expected the 3 known deliberate pings, found {len(mentioning)} at "
        f"lines {[c.lineno for c in mentioning]}; if a site was added or "
        "removed, update this count deliberately"
    )
