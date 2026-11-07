"""A model's `suggested_tags` reach several `', '.join` calls and the database.

`yaml.safe_dump` makes the markdown export safe whatever is in the list, so
this filter is about what gets STORED and what reaches the Discord reply --
defence at the boundary as well as at the sink.

These drive the REAL path: the `_classifier` helper mocks the HTTP response so
`_classify_single` parses a genuine model payload, rather than asserting
against a copy of the filter expression.
"""

from __future__ import annotations

import json
from unittest import mock

import httpx
from cortex_utils.llm import LLMClient

from reflex.services.classifier import MAX_SUGGESTED_TAGS, ReflexClassifier


def _classifier(payload):
    c = ReflexClassifier.__new__(ReflexClassifier)
    c.tier1_model, c.tier2_model = "m1", "m2"
    c.tier1_threshold, c.tier2_threshold = 0.7, 0.5
    c.llm = LLMClient("http://llm.test")
    c.llm.client = mock.Mock()
    c.llm.client.post.return_value = httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}]},
        request=httpx.Request("POST", "http://llm.test"),
    )
    return c


def _tags(suggested):
    """The tags a real classification run would end up with."""
    payload = {
        "category": "idea",
        "confidence": 0.9,
        "reasoning": "r",
        "suggested_tags": suggested,
    }
    return _classifier(payload)._classify_single("msg", "m1").suggested_tags


def test_good_tags_survive():
    """The control. Dropping everything would satisfy the tests below."""
    assert _tags(["reading", "ml"]) == ["reading", "ml"]


def test_non_strings_are_dropped():
    assert _tags([1, {"a": 2}, "ok", None, True]) == ["ok"]


def test_blank_tags_are_dropped():
    assert _tags(["", "   ", "keep"]) == ["keep"]


def test_a_bare_string_yields_nothing_rather_than_its_characters():
    """`set.update("urgent")` on the bot path produces six single-character
    tags (reflex-73eb). On THIS path a non-list is refused outright, and the
    isinstance check is what does it."""
    assert _tags("urgent") == []


def test_a_dict_yields_nothing_rather_than_its_keys():
    assert _tags({"urgent": 1}) == []


def test_the_count_is_capped_at_twenty():
    """Asserts the literal, not the constant: importing MAX_SUGGESTED_TAGS and
    comparing against it would follow the constant if someone widened it."""
    assert MAX_SUGGESTED_TAGS == 20
    assert len(_tags([f"t{i}" for i in range(500)])) == 20
