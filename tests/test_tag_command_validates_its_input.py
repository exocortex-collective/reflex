"""The tag command takes a model-parsed value and must not trust its type.

Two adjacent handlers in bot.py both act on a value straight out of the
model's parsed command, and they had diverged: `execute_move` checks
`new_category not in VALID_CATEGORIES` and refuses, while `execute_tag`
called `set.update(new_tags)` with no check at all.

`set.update` takes any iterable, so with a bare string the TYPE is the bug
rather than the content -- `set.update("urgent")` adds six single-character
tags. A model returning `"tags": "urgent"` where the prompt specifies a list
is an ordinary model error, not an attack.

Both handlers now share `clean_tags`, so they cannot diverge again.
"""

from __future__ import annotations

from unittest import mock

from reflex.models.entry import Entry
from reflex.services.bot import ReflexBot
from reflex.services.classifier import MAX_SUGGESTED_TAGS
from reflex.services.commands import ParsedCommand


def _bot():
    b = ReflexBot.__new__(ReflexBot)
    b.storage = mock.Mock()
    b.exporter = None  # export is not what this tests
    return b


def _entry(tags: list[str] | None = None) -> Entry:
    return Entry(
        id=1,
        discord_message_id="1",
        discord_channel_id="1",
        discord_user_id="1",
        category="idea",
        title="t",
        tags=list(tags or []),
        llm_confidence=0.9,
        llm_model="m",
        llm_reasoning="r",
        status="active",
    )


def _msg():
    m = mock.Mock()
    m.reply = mock.AsyncMock()
    return m


async def _tag(entry, value):
    """Drive the real handler, returning the reply text."""
    bot, msg = _bot(), _msg()
    parsed = ParsedCommand(action="tag", target="that", target_keywords=[], parameters={"tags": value})
    await bot.execute_tag(msg, entry, parsed)
    return msg.reply.await_args[0][0]


async def test_a_list_of_tags_is_added():
    """The control. Refusing everything would satisfy the tests below."""
    entry = _entry(["keep"])
    reply = await _tag(entry, ["urgent", "followup"])
    assert entry.tags == ["followup", "keep", "urgent"]
    assert "Added tags" in reply


async def test_a_bare_string_is_refused_not_split_into_characters():
    entry = _entry(["keep"])
    reply = await _tag(entry, "urgent")
    assert entry.tags == ["keep"], "a bare string was iterated into characters"
    assert "No usable tags" in reply


async def test_a_dict_is_refused_not_reduced_to_its_keys():
    entry = _entry(["keep"])
    await _tag(entry, {"urgent": 1})
    assert entry.tags == ["keep"]


async def test_non_string_elements_are_dropped():
    entry = _entry()
    await _tag(entry, [1, "ok", None, "  "])
    assert entry.tags == ["ok"]


async def test_the_order_is_deterministic():
    """A set's order varies with PYTHONHASHSEED, and these tags are written
    into a git-committed vault file -- an arbitrary order is a spurious diff
    on every re-export."""
    a, b = _entry(["zeta"]), _entry(["zeta"])
    await _tag(a, ["alpha", "mu"])
    await _tag(b, ["mu", "alpha"])
    assert a.tags == b.tags == ["alpha", "mu", "zeta"]


async def test_the_total_is_capped():
    entry = _entry([f"e{i}" for i in range(15)])
    await _tag(entry, [f"n{i}" for i in range(15)])
    assert len(entry.tags) == MAX_SUGGESTED_TAGS


async def test_nothing_is_written_when_the_value_is_unusable():
    """The row must not be touched, and the user must be told."""
    bot, msg = _bot(), _msg()
    parsed = ParsedCommand(action="tag", target="that", target_keywords=[], parameters={"tags": "urgent"})
    await bot.execute_tag(msg, _entry(["keep"]), parsed)
    bot.storage.update_entry.assert_not_called()
