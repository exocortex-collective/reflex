"""A model's `suggested_tags` reach two `', '.join` calls and the database.

classifier.py checked `isinstance(suggested_tags, list)` and left the ELEMENTS
unvalidated. A non-string element raised TypeError in the markdown export
(silently, in a daemon thread) and in the Discord reply at bot.py, and the junk
was stored either way.

`yaml.safe_dump` now makes the export itself safe whatever is in the list, so
this filter is about what gets STORED and what reaches the reply -- defence at
the boundary as well as at the sink.
"""

from __future__ import annotations

from reflex.services.classifier import MAX_SUGGESTED_TAGS


def _filter(raw: object) -> list[str]:
    """The expression from classifier.py, applied here to the same inputs.

    Kept in step with the source by test_the_filter_matches_the_source below,
    which fails if classifier.py stops doing this.
    """
    if not isinstance(raw, list):
        raw = []
    return [t for t in raw if isinstance(t, str) and t.strip()][:MAX_SUGGESTED_TAGS]


def test_the_filter_matches_the_source():
    """Guards against this test drifting away from the code it describes."""
    import inspect

    from reflex.services import classifier

    src = inspect.getsource(classifier)
    assert "[t for t in raw_tags if isinstance(t, str) and t.strip()]" in src
    assert ":MAX_SUGGESTED_TAGS" in src


def test_non_strings_are_dropped():
    assert _filter([1, {"a": 2}, "ok", None, True]) == ["ok"]


def test_blank_tags_are_dropped():
    assert _filter(["", "   ", "keep"]) == ["keep"]


def test_a_bare_string_yields_nothing_rather_than_its_characters():
    # set.update("urgent") on the bot path produces six single-character tags
    # (reflex-73eb). On THIS path a non-list is refused outright.
    assert _filter("urgent") == []


def test_the_count_is_capped():
    assert len(_filter([f"t{i}" for i in range(500)])) == MAX_SUGGESTED_TAGS
