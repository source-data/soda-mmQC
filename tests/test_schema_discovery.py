"""Tests for schema leaf / object-list discovery."""

from __future__ import annotations

import json
from pathlib import Path

from soda_mmqc.core.schema_discovery import (
    LeafKind,
    discover_object_lists,
    discover_schema,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_toy_schema_leaves():
    schema = json.loads((FIXTURES / "toy_eval_schema.json").read_text())
    leaves = discover_schema(schema)
    patterns = [leaf.pattern for leaf in leaves]
    assert patterns == [
        "tags",
        "item.id",
        "item.label",
        "item.status",
        "item.meta.author",
        "item.meta.year",
        "panels[].id",
        "panels[].label",
        "panels[].status",
    ]
    assert leaves[0].kind is LeafKind.EXTENDED_PRIMITIVE
    assert leaves[1].kind is LeafKind.SCALAR
    assert leaves[6].kind is LeafKind.ROW
    assert leaves[6].object_list_name == "panels"


def test_toy_schema_object_lists():
    schema = json.loads((FIXTURES / "toy_eval_schema.json").read_text())
    lists = discover_object_lists(schema)
    assert len(lists) == 1
    assert lists[0].list_name == "panels"
    assert lists[0].by_list_key == "panels"
