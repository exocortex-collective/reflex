"""Model-supplied values must not be able to introduce YAML structure.

The exported markdown is the one artifact a human and Obsidian actually read,
and it is git-committed and pushed to a remote. `docs/agent-isolation.md` item
4 is the rule: the deciding question is not whether the input is trusted --
almost none of it is -- but what the model's output CAUSES.

The frontmatter used to be hand-built with an f-string, rendering tags as
`tags: [{', '.join(entry.tags)}]`. Two failures came out of that one line:

  * a tag containing `]` and a newline closed the list and added its own keys,
    so the model could forge `category` and `status` past the closed set that
    classifier.py enforces;
  * a non-string tag made `', '.join` raise TypeError inside the daemon thread
    in `_export_and_commit_sync`, whose `except Exception` logs and swallows --
    so the database row existed, no markdown was ever written, and the user saw
    a tick. Silent loss from a one-line model response.

Serialising with `yaml.safe_dump` closes both for EVERY field at once, which
is why these tests assert on the parsed frontmatter rather than on the text:
the property is "no value can add a key", not "this string is absent".
"""

from __future__ import annotations

from datetime import datetime

import yaml

from reflex.models.entry import Entry
from reflex.services.classifier import MAX_SUGGESTED_TAGS
from reflex.storage.exporter import MarkdownExporter

from .test_suggested_tags_are_filtered import _tags as _classify_with_tags

# The exact string from cortex-57q3, which the original review demonstrated.
INJECTION = "x]\ncategory: admin\nstatus: archived\nz: ["

EXPECTED_KEYS = {
    "id",
    "discord_message_id",
    "category",
    "tags",
    "captured_at",
    "llm_model",
    "llm_confidence",
    "status",
}


def _entry(tags: list) -> Entry:
    return Entry(
        id=1,
        # An INT, so the str() coercion in the exporter is pinned: with a
        # str here the coercion is a no-op and its removal survived.
        discord_message_id=1460394971273363629,  # type: ignore[arg-type]
        discord_channel_id="1460394971273363629",
        discord_user_id="509879976879718411",
        category="idea",
        title="A normal title",
        tags=tags,
        llm_confidence=0.91,
        llm_model="qwen2.5:7b",
        llm_reasoning="because",
        status="active",
        captured_at=datetime(2026, 11, 7, 8, 0, 0),
        original_message="body",
    )


def _frontmatter(tags: list) -> dict:
    """Render and parse back. Unbound so no git repo is needed to construct one."""
    md = MarkdownExporter._generate_markdown(None, _entry(tags))  # type: ignore[arg-type]
    assert md.startswith("---\n")
    # split("---\n", 2): a tag containing `---` would otherwise
    # break the TEST rather than the code under test.
    return yaml.safe_load(md.split("---\n", 2)[1])


def test_a_crafted_tag_cannot_add_frontmatter_keys():
    fm = _frontmatter([INJECTION])
    assert set(fm) == EXPECTED_KEYS, f"the model introduced keys: {set(fm) - EXPECTED_KEYS}"
    # The closed-set values survive, and the attack is just a tag value.
    assert fm["category"] == "idea"
    assert fm["status"] == "active"
    assert fm["tags"] == [INJECTION]


def test_a_non_string_tag_does_not_lose_the_export():
    # Previously TypeError inside a swallowing daemon thread: no file, no error.
    fm = _frontmatter([1, {"a": 2}, "ok"])
    assert set(fm) == EXPECTED_KEYS
    assert fm["tags"] == [1, {"a": 2}, "ok"]


def test_the_ordinary_frontmatter_is_unchanged_in_shape():
    fm = _frontmatter(["reading", "ml"])
    assert set(fm) == EXPECTED_KEYS
    assert fm["tags"] == ["reading", "ml"]
    assert fm["category"] == "idea"
    assert fm["llm_confidence"] == 0.91
    # discord_message_id must stay a STRING; it is a 19-digit snowflake and
    # YAML would otherwise read it as an int and lose precision on the way back.
    assert fm["discord_message_id"] == "1460394971273363629"
    assert isinstance(fm["discord_message_id"], str)


def test_no_tags_still_renders():
    assert _frontmatter([])["tags"] == []


def test_the_rendered_text_keeps_the_previous_file_shape():
    """The serialiser OPTIONS, not the parsed result.

    `default_flow_style=None` and `sort_keys=False` are there to keep the file
    looking as it did -- an inline tag list and the original field order. Every
    other test here parses the YAML back, so all three options could be
    changed without failing anything. This one reads the text.
    """
    md = MarkdownExporter._generate_markdown(None, _entry(["reading", "ml"]))  # type: ignore[arg-type]
    body = md.split("---\n", 2)[1]
    lines = [line.split(":", 1)[0] for line in body.splitlines() if line and not line[0].isspace()]
    assert lines == [
        "id",
        "discord_message_id",
        "category",
        "tags",
        "captured_at",
        "llm_model",
        "llm_confidence",
        "status",
    ], "field order changed; sort_keys must stay False"
    assert "tags: [reading, ml]" in body, (
        "the tag list is no longer inline; default_flow_style must stay None"
    )


def test_a_dropped_tag_count_is_logged(monkeypatch):
    """A silent drop is what made the original bug invisible.

    The module's logger is replaced rather than captured. caplog does not see
    structlog output, and capsys does but only when no other test has
    reconfigured logging -- I tried both, and each passed in one arrangement
    and failed in the other. Who the code CALLS is the behaviour worth
    pinning, and it does not depend on global logging state.
    """
    from reflex.services import classifier as mod

    said: list[str] = []
    monkeypatch.setattr(mod.logger, "warning", lambda msg, *a, **k: said.append(str(msg)))
    _classify_with_tags([1, "ok", "  ", {"a": 2}])
    assert any("dropped 3 of 4 suggested_tags" in m for m in said), said


def test_a_non_list_tags_value_is_logged(monkeypatch):
    from reflex.services import classifier as mod

    said: list[str] = []
    monkeypatch.setattr(mod.logger, "warning", lambda msg, *a, **k: said.append(str(msg)))
    _classify_with_tags("urgent")
    assert any("not a list" in m and "str" in m for m in said), said


def test_a_non_ascii_tag_survives_unescaped():
    """Pins allow_unicode=True: without it safe_dump escapes non-ASCII to
    \\uXXXX, which is valid YAML but unreadable in the vault."""
    fm = _frontmatter(["café", "日本語"])
    assert fm["tags"] == ["café", "日本語"]
    md = MarkdownExporter._generate_markdown(None, _entry(["café"]))  # type: ignore[arg-type]
    assert "café" in md, "non-ASCII was escaped; allow_unicode must stay True"
