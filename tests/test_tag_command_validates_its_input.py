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
    # A double, not None: with exporter=None the re-export branch was never
    # taken, so deleting the export call survived every test -- and the tags
    # reaching the vault is the whole reason their order matters.
    b.exporter = mock.Mock()
    # Snapshot the tags AS WRITTEN. `assert_called_once_with(entry)` compares
    # the same mutated object to itself, so it cannot see what the entry held
    # at call time -- moving the assignment to AFTER the write persists the
    # OLD tags and survived every test. The snapshot is what catches that.
    b.persisted = []
    b.storage.update_entry.side_effect = lambda e: b.persisted.append(list(e.tags))
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


async def _run(entry, value):
    """Drive the real handler. Returns (bot, reply text) so the writes can be
    asserted -- not just the reply."""
    bot, msg = _bot(), _msg()
    parsed = ParsedCommand(action="tag", target="that", target_keywords=[], parameters={"tags": value})
    await bot.execute_tag(msg, entry, parsed)
    return bot, msg.reply.await_args[0][0]


async def _tag(entry, value):
    _bot_unused, reply = await _run(entry, value)
    return reply


async def test_a_list_of_tags_is_added_and_persisted():
    """The control. Refusing everything would satisfy the tests below.

    Asserts the WRITES too: deleting `storage.update_entry` from the success
    path survived every test, so nothing checked the change was saved.
    """
    entry = _entry(["keep"])
    bot, reply = await _run(entry, ["urgent", "followup"])
    assert entry.tags == ["followup", "keep", "urgent"]
    assert "Added tags" in reply
    bot.storage.update_entry.assert_called_once_with(entry)
    bot.exporter.export_entry.assert_called_once_with(entry)
    # What reached the database, not what the object holds now.
    assert bot.persisted == [["followup", "keep", "urgent"]]


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


async def test_the_order_is_sorted_not_arbitrary():
    """A set's order varies with PYTHONHASHSEED, and these tags are written
    into a git-committed vault file -- an arbitrary order is a spurious diff
    on every re-export.

    Eight tags, inserted in reverse. Comparing two entries in one process
    proves nothing (they share a seed) and three elements come out sorted by
    chance one time in six -- measured, `sorted`->`list` survived 1 of 40
    seeds with three. With eight the chance is 1 in 40320.
    """
    want = [f"t{i}" for i in range(8)]
    entry = _entry([want[-1]])
    await _tag(entry, list(reversed(want[:-1])))
    assert entry.tags == want


async def test_existing_tags_are_never_dropped():
    """A cap on the TOTAL truncated alphabetically and so deleted the user's
    own tags: 15 existing z* plus 15 new a* lost ten of the existing ones,
    while the reply still claimed all 15 were added. Only the INCOMING batch
    is capped, by clean_tags."""
    existing = [f"z{i:02d}" for i in range(15)]
    entry = _entry(existing)
    await _tag(entry, [f"a{i:02d}" for i in range(15)])
    assert set(existing) <= set(entry.tags), "an existing tag was deleted"
    assert len(entry.tags) == 30


async def test_the_incoming_batch_is_capped():
    entry = _entry()
    await _tag(entry, [f"n{i:03d}" for i in range(500)])
    assert len(entry.tags) == MAX_SUGGESTED_TAGS


async def test_nothing_is_written_when_the_value_is_unusable():
    """The row must not be touched, and the user must be told."""
    bot, msg = _bot(), _msg()
    parsed = ParsedCommand(action="tag", target="that", target_keywords=[], parameters={"tags": "urgent"})
    await bot.execute_tag(msg, _entry(["keep"]), parsed)
    bot.storage.update_entry.assert_not_called()


async def test_the_cap_is_applied_after_the_filter_not_before():
    """Capping first would keep 20 junk entries and filter them to nothing.

    20 non-strings followed by 5 good tags: filter-then-cap gives the 5,
    cap-then-filter gives none.
    """
    entry = _entry()
    await _tag(entry, [None] * 20 + [f"g{i}" for i in range(5)])
    assert entry.tags == ["g0", "g1", "g2", "g3", "g4"]
