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


@pytest.mark.parametrize("conf", [0.6, 0.65], ids=["at-the-bar", "above-it"])
def test_a_tier_2_failure_keeps_the_usable_tier_1_classification(conf):
    """A confidence that escalates but is still storable must survive tier 2
    failing, or the capture is lost to "Sorry, something went wrong" with
    nothing retrying.

    0.6 is the BOUNDARY and is tested because the guard is `>=`: handle_capture
    gates on `< tier2_threshold`, so exactly 0.6 is stored. Without this case
    `>=` could be mutated to `>` and survive the whole suite -- the `at` half
    of "at or above" was the untested half.
    """
    import httpx

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if '"t1"' in request.content.decode():
            seen.append("t1")
            return _ok(GOOD_CLASSIFICATION.format(c=conf))
        seen.append("t2")
        raise httpx.ConnectError("proxy died between the two calls")

    c = _classifier_on(handler)
    try:
        result = c.classify("an idea about something")
    finally:
        c.llm.client.close()
    assert seen == ["t1", "t2"], "tier 2 must still be attempted"
    assert result.model == "t1", "the tier-1 answer must survive tier 2 failing"
    assert result.confidence == conf


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
@pytest.mark.parametrize("as_set", [list, frozenset], ids=["list", "frozenset"])
async def test_a_non_string_move_category_is_refused(as_set):
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

    # Parametrized over BOTH shapes. The frozenset case is the one with
    # teeth: with a list, `["project"] not in [...]` is a harmless False and
    # the isinstance mutant survives, so that case alone would be no test.
    # The list case is today's real configuration and must keep working.
    with mock.patch.object(
        ReflexBot, "VALID_CATEGORIES", as_set(ReflexBot.VALID_CATEGORIES)
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

    # The MERGED value is what ships: discord.py merges a per-send
    # AllowedMentions over the client default, and the whole fix rests on that
    # merge keeping `users`. Asserted here because every other test in this
    # file checks what was handed to a mock, so if the merge ever dropped
    # `users` all three pings would go silent with the suite still green.
    merged = discord.AllowedMentions.none().merge(_ping_only(user)).to_dict()
    assert merged.get("users") == [509879976879718411]
    assert merged.get("parse") == [], "nothing in the content may ping"
    # replied_user does NOT survive the merge -- the client default's False
    # wins over the unset sentinel -- so a reply that needs to notify must
    # interpolate the mention rather than rely on the reply ping.
    assert not merged.get("replied_user", False)


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
    """AllowedMentions.none() on the client also sets users=False, and this
    prompt then WAITS in snooze_pending for a reply -- so with the mention
    muted it arrives unnoticed and the entry stays unsnoozed.
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


def _mentioning_sends(tree: "ast.AST") -> list["ast.Call"]:
    """Every `.reply`/`.send` call in `tree` whose message mentions someone.

    Module-level so the synthetic tests below drive the SAME walk the guard
    applies to bot.py. Their expected values are literals, not computed by
    this function, so sharing it does not make it their oracle.
    """
    import ast

    def mentions_someone(call: ast.Call) -> bool:
        # Keywords too: `reply(content=f"{user.mention}")` slipped past a
        # version that walked `call.args` alone. `allowed_mentions` is
        # excluded, since _ping_only(user) is not a mention in the message.
        trees = list(call.args) + [
            k.value for k in call.keywords if k.arg != "allowed_mentions"
        ]
        nodes = [n for arg in trees for n in ast.walk(arg)]
        # Any `<anything>.mention`, not just the name `user`:
        # `message.author.mention` is the natural spelling in a capture
        # handler and `member.mention` in a guild one, and both were
        # invisible while this matched only one name.
        attribute = any(
            isinstance(n, ast.Attribute) and n.attr == "mention" for n in nodes
        )
        # A hand-spelled mention never touches `.mention`, so the attribute
        # walk cannot see it. Matched on the literal prefix instead.
        hand_spelled = any(
            isinstance(n, ast.Constant) and isinstance(n.value, str) and "<@" in n.value
            for n in nodes
        )
        return attribute or hand_spelled

    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"reply", "send"}
        and mentions_someone(n)
    ]


def _allow_list_kwarg(call: "ast.Call") -> "ast.expr | None":
    for k in call.keywords:
        if k.arg == "allowed_mentions":
            return k.value
    return None


def _is_ping_only_call(node: "ast.expr | None") -> bool:
    import ast

    return isinstance(node, ast.Call) and (
        (isinstance(node.func, ast.Name) and node.func.id == "_ping_only")
        or (isinstance(node.func, ast.Attribute) and node.func.attr == "_ping_only")
    )


def _allow_list_violations(tree: "ast.AST") -> tuple[list[int], list[int]]:
    """(lines with no allow-list, lines whose allow-list is not _ping_only).

    Shared by the real-source guard and the synthetic tests below. bot.py
    contains no violating send, so an assertion written only against bot.py
    cannot be killed -- putting the logic here lets a mutation to it die on
    source that does violate.
    """
    sends = _mentioning_sends(tree)
    missing = [c.lineno for c in sends if _allow_list_kwarg(c) is None]
    wrong = [
        c.lineno
        for c in sends
        if _allow_list_kwarg(c) is not None
        and not _is_ping_only_call(_allow_list_kwarg(c))
    ]
    return missing, wrong


SEND_SHAPES = [
    ('await m.reply(f"{user.mention} hi")', "positional f-string"),
    ('await m.reply(content=f"{user.mention} hi")', "keyword argument"),
    ('await m.reply(f"{m.author.mention} hi")', "mention via another object"),
    ('await m.reply(f"{member.mention} hi")', "mention via another name"),
    ('await m.reply(f"<@{user.id}> hi")', "hand-spelled mention"),
    ('await m.reply(f"ok, <@{user.id}>")', "hand-spelled, not at position 0"),
    ('await m.channel.send(f"{user.mention} hi")', "channel.send"),
]


@pytest.mark.parametrize("src,shape", SEND_SHAPES, ids=[s for _, s in SEND_SHAPES])
def test_the_mention_walk_sees_every_shape_it_claims_to(src, shape):
    """Drives the walk on synthetic source, because bot.py contains only one
    of these shapes today.

    Without this, removing the hand-spelled match from the walk survives the
    whole suite -- a forward guard with nothing to catch cannot be killed by
    the real file, which is the same trap as the category closed set.
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(f"async def f(m, user, member):\n    {src}\n"))
    found = _mentioning_sends(tree)
    assert len(found) == 1, f"the walk must see a {shape}: {ast.dump(tree)}"
    # Through the shared helper, so gutting its `missing` list dies here too.
    missing, wrong = _allow_list_violations(tree)
    assert missing == [found[0].lineno], (
        f"a {shape} with no allow-list must be reported missing "
        f"(missing={missing}, wrong={wrong})"
    )
    assert wrong == [], (
        "a send with no allow-list at all is MISSING, not wrong -- the two "
        f"lists must not both claim it (wrong={wrong})"
    )


WIDER_ALLOW_LISTS = [
    ("AM.all()", "an attribute call"),
    ("AllowedMentions(everyone=True)", "a bare name call"),
    ("None", "not a call at all"),
]


@pytest.mark.parametrize(
    "value,shape", WIDER_ALLOW_LISTS, ids=[s for _, s in WIDER_ALLOW_LISTS]
)
def test_any_allow_list_that_is_not_ping_only_is_rejected(value, shape):
    """The value check, on more than one shape.

    Pinned on three because the check is a disjunction over call forms: with
    only an attribute call in the table, dropping the Name arm or replacing
    the identity test with a tautology both survive, and the silent direction
    is a bare `AllowedMentions(everyone=True)` passing as if it were an
    allow-list of one.
    """
    import ast
    import textwrap

    tree = ast.parse(
        textwrap.dedent(
            f"""
            async def f(m, user):
                await m.reply(f"{{user.mention}} hi", allowed_mentions={value})
            """
        )
    )
    missing, wrong = _allow_list_violations(tree)
    assert missing == [], f"the keyword is present for {shape}"
    assert wrong == [3], (
        f"{shape} is not a _ping_only(...) call and must be reported wrong "
        f"(missing={missing}, wrong={wrong})"
    )


def test_a_wider_allow_list_than_ping_only_is_rejected():
    """The VALUE check. `allowed_mentions=AllowedMentions.all()` satisfies a
    presence check and fully overrides the client default -- measured as
    {'replied_user': True, 'parse': ['everyone','users','roles']} -- so an
    echoed @everyone would ping the server through a passing test.
    """
    import ast
    import textwrap

    tree = ast.parse(
        textwrap.dedent(
            """
            async def f(m, user):
                await m.reply(f"{user.mention} hi", allowed_mentions=AM.all())
            """
        )
    )
    missing, wrong = _allow_list_violations(tree)
    assert missing == [], "the keyword IS present, so this is not the missing case"
    assert wrong == [3], (
        "a non-_ping_only allow-list must be reported as wrong; presence is "
        f"not enough (missing={missing}, wrong={wrong})"
    )


def test_every_deliberate_mention_passes_an_allow_list():
    """A pattern guard, so a NEW reply that pings is not silently muted by the
    client default.

    Walked with ast rather than matched with a regex -- a regex over source
    missed the snooze confirmation, whose message contains a parenthesised
    strftime call.

    WHAT THIS CANNOT SEE, stated so the guard is not mistaken for a proof. A
    message built into a variable first (`msg = f"{user.mention}.."` then
    `reply(msg)`) is invisible to any source-level walk, and such a send is
    not even counted as mentioning -- so neither the presence check nor the
    value check runs on it, and the count assertion is still satisfied.
    Nothing here limits that case; the behavioural test per ping site is what
    covers the sites that exist.

    The shapes this DOES see are pinned on synthetic source by
    test_the_mention_walk_sees_every_shape_it_claims_to, because bot.py
    contains only one of them.
    """
    import inspect

    from reflex.services import bot as bot_module

    import ast

    tree = ast.parse(inspect.getsource(bot_module))
    mentioning = _mentioning_sends(tree)
    assert mentioning, "the walk must still find the mentioning sends"

    missing, wrong_value = _allow_list_violations(tree)
    assert missing == [], (
        f"sends at lines {missing} interpolate a mention with no "
        "allowed_mentions allow-list, so the client default mutes them"
    )
    assert wrong_value == [], (
        f"sends at lines {wrong_value} pass an allowed_mentions that is not a "
        "_ping_only(...) call; anything wider can be widened by model text"
    )

    assert len(mentioning) == 3, (
        f"expected the 3 known deliberate pings, found {len(mentioning)} at "
        f"lines {[c.lineno for c in mentioning]}; if a site was added or "
        "removed, update this count deliberately"
    )


@pytest.mark.asyncio
async def test_the_archive_confirmation_pings_the_reacting_user():
    """The AST guard cannot see WHOM a send allow-lists, only that it does.

    Every one of these replies is to the BOT's own message, so
    `reaction.message.author` IS the bot: allow-listing it notifies nobody
    while leaving the keyword in place, which the pattern guard accepts. Each
    ping site needs its own behavioural test for that reason.
    """
    import discord

    bot = _bot_with_doubles()

    user = mock.MagicMock(spec=discord.User)
    user.id = 509879976879718411
    user.mention = "<@509879976879718411>"
    reaction = mock.MagicMock()
    reaction.message.id = 77
    # The bot's own message, which is what makes the wrong-author swap silent.
    reaction.message.author = mock.MagicMock(spec=discord.User)
    reaction.message.author.id = 999
    reaction.message.reply = mock.AsyncMock()

    await bot.handle_archive_entry(reaction, user, entry_id=5, source="digest")

    kwargs = reaction.message.reply.await_args.kwargs
    assert "allowed_mentions" in kwargs, "the confirmation must be allowed to notify"
    assert kwargs["allowed_mentions"].to_dict().get("users") == [user.id], (
        "the allow-list must name the REACTING user, not the message author "
        "(which is the bot on every one of these replies)"
    )


@pytest.mark.asyncio
async def test_the_fixed_snooze_confirmation_pings_the_reacting_user():
    """Third of the three sites. Same wrong-author swap, same silence."""
    import discord

    bot = _bot_with_doubles()

    user = mock.MagicMock(spec=discord.User)
    user.id = 509879976879718411
    user.mention = "<@509879976879718411>"
    reaction = mock.MagicMock()
    reaction.message.id = 77
    reaction.message.author = mock.MagicMock(spec=discord.User)
    reaction.message.author.id = 999
    reaction.message.reply = mock.AsyncMock()

    await bot.handle_snooze_entry(reaction, user, entry_id=5, days=3, label="3 days")

    kwargs = reaction.message.reply.await_args.kwargs
    assert kwargs["allowed_mentions"].to_dict().get("users") == [user.id]


def test_a_tier_2_model_fault_keeps_the_models_own_diagnosis():
    """Without the bare re-raise this branch wraps the model's own diagnosis
    in a transport-failure message.

    Input that reaches it: tier 1 fails at the transport (so there is no
    tier-1 answer to keep) and tier 2 answers 200 with readable JSON that
    _classify_single then rejects. Without the bare re-raise, the model's own
    "category 'nope' is not one of ..." is replaced by a wrapper saying the
    transport failed -- which it did not. Same exception class either way, so
    bot.py behaves identically and only the operator is misled; that is the
    diagnosis-swapping defect this whole batch is about.
    """
    import httpx
    from cortex_utils.llm import LLMError

    legs: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if '"t1"' in request.content.decode():
            legs.append("t1")
            raise httpx.ConnectError("proxy is down")
        legs.append("t2")
        return _ok(
            '{"category": "nope", "confidence": 0.95, "reasoning": "r",'
            ' "suggested_tags": [], "actionable": false}'
        )

    c = _classifier_on(handler)
    try:
        with pytest.raises(LLMError) as caught:
            c.classify("an idea")
    finally:
        c.llm.client.close()

    assert legs == ["t1", "t2"]
    assert "nope" in str(caught.value), (
        "the model's own diagnosis must survive, not be replaced by a "
        f"transport-failure wrapper: {caught.value}"
    )


def test_a_tier_2_fault_is_counted_even_when_the_tier_1_answer_is_kept():
    """The keep path is the only outcome nothing else records.

    The capture SUCCEEDS on the tier-1 answer, so no exception reaches
    on_message and nothing else in the estate notes that tier 2 failed -- an
    expired tier-2 key would be indistinguishable from a tier 2 that merely
    agrees. The paths that raise are counted by on_message already, which is
    why the increment sits inside this branch rather than above it
    (test_a_failed_capture_is_not_counted_twice pins the other half).
    """
    import httpx
    from cortex_utils.metrics import ERRORS

    def handler(request: httpx.Request) -> httpx.Response:
        if '"t1"' in request.content.decode():
            return _ok(GOOD_CLASSIFICATION.format(c=0.65))
        raise httpx.ConnectError("tier 2 is down")

    counter = ERRORS.labels(service="reflex", error_type="classify_tier2")
    before = counter._value.get()

    c = _classifier_on(handler)
    try:
        result = c.classify("an idea")
    finally:
        c.llm.client.close()

    assert result.model == "t1", "the storable answer is still kept"
    assert counter._value.get() == before + 1, (
        "a tier-2 failure must stay visible in metrics even when the "
        "capture succeeds on the tier-1 answer"
    )


def test_an_unstorable_tier_1_answer_is_not_kept():
    """The guard is `>= tier2_threshold`, not `is not None`. `result` is set
    for the whole band below the escalation bar, and in the un-storable part
    of it handing the answer back converts a counted failure into a silent
    low-confidence reply -- no error reaction, no ERRORS increment.
    """
    import httpx
    from cortex_utils.llm import LLMError

    def handler(request: httpx.Request) -> httpx.Response:
        if '"t1"' in request.content.decode():
            return _ok(GOOD_CLASSIFICATION.format(c=0.30))
        raise httpx.ConnectError("tier 2 is down")

    c = _classifier_on(handler)
    try:
        with pytest.raises(LLMError):
            c.classify("an idea")
    finally:
        c.llm.client.close()


def test_a_tier_2_type_error_still_keeps_the_storable_tier_1_answer(caplog):
    """The reason the handlers catch `Exception` and not a tuple.

    `_post_completion` documents four raisable types and that list is
    INCOMPLETE: `_reject_if_truncated` runs inside its own `except LLMError`
    block and iterates `choices` with no type check, so a 200 carrying
    `{"choices": 5}` raises a bare `TypeError` over the top of the LLMError
    (utils-f643). A narrow tuple misses it and discards the storable answer.

    A ConnectError cannot pin this -- a narrow tuple catches that one -- which
    is why the assertion below is on the exception TYPE.
    """
    import httpx

    legs: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if '"t1"' in request.content.decode():
            legs.append("t1")
            return _ok(GOOD_CLASSIFICATION.format(c=0.65))
        legs.append("t2")
        return httpx.Response(200, json={"choices": 5})

    c = _classifier_on(handler)
    try:
        with caplog.at_level("WARNING"):
            result = c.classify("an idea about something")
    finally:
        c.llm.client.close()

    assert legs == ["t1", "t2"]
    assert result.model == "t1", (
        "a non-LLMError, non-RequestError fault in tier 2 must still leave "
        "the storable tier-1 answer standing"
    )
    assert result.confidence == 0.65
    # The EXCEPTION TYPE, not just the outcome. Without this the three
    # assertions above hold for any caught exception -- a plain LLMError body
    # passes them identically -- so when utils-f643 lands and
    # `_reject_if_truncated` stops leaking TypeError, this test would keep
    # passing while silently ceasing to guard the widened catch, and the
    # mutant that survived the first attempt would survive again.
    assert "'int' object is not iterable" in caplog.text, (
        "the caught fault must be the TypeError, not an LLMError that a "
        f"narrow tuple would also have caught: {caplog.text}"
    )


def test_a_tier_1_type_error_still_escalates_to_tier_2():
    """The leg where a narrow catch costs the WHOLE cascade, not a fallback.

    A `{"choices": 5}` envelope from the tier-1 model escaped a narrow tuple
    as a bare TypeError, so tier 2 was never asked and the capture was lost
    outright -- worse than the tier-2 case, where the cost is only a lost
    fallback. Broadening a tier-1 handler cannot invent or keep an answer;
    there is none yet, so it can only let tier 2 run.
    """
    import httpx

    legs: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if '"t1"' in request.content.decode():
            legs.append("t1")
            return httpx.Response(200, json={"choices": 5})
        legs.append("t2")
        return _ok(GOOD_CLASSIFICATION.format(c=0.9))

    c = _classifier_on(handler)
    try:
        result = c.classify("an idea")
    finally:
        c.llm.client.close()

    assert legs == ["t1", "t2"], "tier 2 must be asked after a tier-1 fault"
    assert result.model == "t2"


def test_a_tier_1_type_error_in_intent_validation_still_escalates():
    """Same gap, the other method. validate_intent's tier-1 handler was narrow
    too, so command-shaped text hit a raw TypeError instead of a second
    opinion.
    """
    import httpx

    legs: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if '"t1"' in request.content.decode():
            legs.append("t1")
            return httpx.Response(200, json={"choices": 5})
        legs.append("t2")
        return _ok('{"is_command": true, "confidence": 0.95}')

    c = _classifier_on(handler)
    try:
        is_command, confidence = c.validate_intent("move that to project")
    finally:
        c.llm.client.close()

    assert legs == ["t1", "t2"]
    assert is_command is True
    assert confidence == 0.95


def test_a_tier_2_type_error_in_intent_validation_is_an_llm_error():
    """validate_intent's tier-2 leg wrapped only RequestError, so the same
    envelope left it as a raw TypeError and `Raises: LLMError` was false for
    the exact input the fix was built around.

    No tier-1 fallback here by design -- this answer decides whether text is
    executed as a command -- so the only question is whether the caller gets
    the documented exception type.
    """
    import httpx
    from cortex_utils.llm import LLMError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": 5})

    c = _classifier_on(handler)
    try:
        with pytest.raises(LLMError):
            c.validate_intent("move that to project")
    finally:
        c.llm.client.close()


def test_a_failed_capture_is_not_counted_twice():
    """The increment lives inside the KEEP branch, not above it.

    On the paths that raise, the exception reaches `on_message`, which already
    increments ERRORS under error_type=message_handler. Counting here as well
    made one failed capture read as two in any sum over error_type. The keep
    path is the only outcome nothing else records, because the capture
    succeeds.
    """
    import httpx
    from cortex_utils.llm import LLMError
    from cortex_utils.metrics import ERRORS

    def handler(request: httpx.Request) -> httpx.Response:
        if '"t1"' in request.content.decode():
            # Below tier2_threshold, so there is nothing storable to keep.
            return _ok(GOOD_CLASSIFICATION.format(c=0.30))
        raise httpx.ConnectError("tier 2 is down")

    counter = ERRORS.labels(service="reflex", error_type="classify_tier2")
    before = counter._value.get()

    c = _classifier_on(handler)
    try:
        with pytest.raises(LLMError):
            c.classify("an idea")
    finally:
        c.llm.client.close()

    assert counter._value.get() == before, (
        "a failure that propagates is counted by on_message; counting it "
        "here too double-counts one event"
    )


# ---------------------------------------------------------------------------
# The handlers catch Exception. The tests above reach a non-LLMError only
# through one cortex-utils internal, and utils-f643 is open to fix exactly
# that -- at which point {"choices": 5} raises LLMError, a narrow tuple would
# catch it too, and those tests would keep passing while guarding nothing.
#
# These raise from reflex's OWN seam instead, so the coverage does not expire
# on a pin bump. The method under test is the HANDLER; _classify_single and
# _validate_intent_single are its collaborators, which is why patching them is
# not stubbing the thing under test.
# ---------------------------------------------------------------------------


def _classifier():
    from reflex.services.classifier import ReflexClassifier

    c = ReflexClassifier("http://proxy.test", "t1", "t2", api_key="k")
    c.llm.client.close()
    return c


def test_a_non_llm_error_from_tier_1_still_escalates():
    """A reflex-side fault in tier 1 must not skip tier 2."""
    from reflex.models.entry import ClassificationResult

    c = _classifier()
    good = ClassificationResult(
        category="idea", confidence=0.9, reasoning="r", suggested_tags=[], model="t2"
    )
    calls: list[str] = []

    def fake(message: str, model: str):
        calls.append(model)
        if model == "t1":
            raise ValueError("a reflex-side bug, not an LLMError")
        return good

    with mock.patch.object(c, "_classify_single", side_effect=fake):
        result = c.classify("an idea")

    assert calls == ["t1", "t2"], "tier 2 must be asked after a non-LLMError"
    assert result is good


def test_a_non_llm_error_from_tier_2_keeps_the_storable_tier_1_answer():
    """And must not discard a storable tier-1 answer."""
    from reflex.models.entry import ClassificationResult

    c = _classifier()
    tier1 = ClassificationResult(
        category="idea", confidence=0.65, reasoning="r", suggested_tags=[], model="t1"
    )

    def fake(message: str, model: str):
        if model == "t1":
            return tier1
        raise ValueError("a reflex-side bug, not an LLMError")

    with mock.patch.object(c, "_classify_single", side_effect=fake):
        result = c.classify("an idea")

    assert result is tier1


def test_a_non_llm_error_from_tier_1_intent_validation_still_escalates():
    """Same seam, the command path."""
    c = _classifier()
    calls: list[str] = []

    def fake(message: str, model: str):
        calls.append(model)
        if model == "t1":
            raise ValueError("a reflex-side bug, not an LLMError")
        return True, 0.95

    with mock.patch.object(c, "_validate_intent_single", side_effect=fake):
        is_command, confidence = c.validate_intent("move that to project")

    assert calls == ["t1", "t2"]
    assert is_command is True


def test_a_non_llm_error_from_tier_2_intent_validation_is_an_llm_error():
    """And the documented contract holds for a non-LLMError too."""
    from cortex_utils.llm import LLMError

    c = _classifier()

    with mock.patch.object(
        c, "_validate_intent_single", side_effect=ValueError("reflex-side bug")
    ):
        with pytest.raises(LLMError):
            c.validate_intent("move that to project")
