"""Discover leaf properties and object lists from a JSON Schema."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional


class LeafKind(str, Enum):
    """Kind of leaf property discovered from schema."""

    SCALAR = "scalar"
    EXTENDED_PRIMITIVE = "extended_primitive"
    ROW = "row"


PRIMITIVE_TYPES = frozenset({"string", "number", "integer", "boolean"})


@dataclass(frozen=True)
class LeafPropertySpec:
    """One leaf property pattern from schema traversal."""

    pattern: str
    kind: LeafKind
    enum_values: Optional[tuple[str, ...]] = None
    object_list_name: Optional[str] = None


@dataclass(frozen=True)
class ObjectListSpec:
    """One list-of-objects property requiring row alignment."""

    list_name: str
    by_list_key: str
    parent: Optional[ObjectListSpec] = None
    parent_field: Optional[str] = None


def discover_schema(schema: Mapping[str, Any]) -> tuple[LeafPropertySpec, ...]:
    """Return all leaf property specs in stable order."""
    leaves: list[LeafPropertySpec] = []
    _walk_node(schema, prefix="", leaves=leaves)
    return tuple(leaves)


def discover_object_lists(schema: Mapping[str, Any]) -> tuple[ObjectListSpec, ...]:
    """Return object-list specs in root-to-leaf order."""
    lists: list[ObjectListSpec] = []
    _walk_object_lists(schema, prefix="", parent=None, parent_field=None, lists=lists)
    return tuple(lists)


def list_name_to_by_list_key(list_name: str) -> str:
    """Map alignment list name to ``by_list`` output key (no ``[]``)."""
    return list_name.replace("[].", ".")


def _walk_node(
    node: Mapping[str, Any],
    *,
    prefix: str,
    leaves: list[LeafPropertySpec],
    object_list_name: Optional[str] = None,
) -> None:
    node_type = _primary_type(node)
    if node_type == "object":
        for name, child in node.get("properties", {}).items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            _walk_node(
                child,
                prefix=child_prefix,
                leaves=leaves,
                object_list_name=object_list_name,
            )
        return

    if node_type == "array":
        items = node.get("items", {})
        items_type = _primary_type(items)
        if items_type in PRIMITIVE_TYPES:
            leaves.append(
                LeafPropertySpec(
                    pattern=prefix,
                    kind=(
                        LeafKind.ROW
                        if object_list_name
                        else LeafKind.EXTENDED_PRIMITIVE
                    ),
                    enum_values=_enum_values(items),
                    object_list_name=object_list_name,
                )
            )
            return

        if items_type == "object":
            list_name = prefix
            for name, child in items.get("properties", {}).items():
                row_pattern = f"{list_name}[].{name}"
                _walk_node(
                    child,
                    prefix=row_pattern,
                    leaves=leaves,
                    object_list_name=list_name,
                )
            return

    if node_type in PRIMITIVE_TYPES:
        leaves.append(
            LeafPropertySpec(
                pattern=prefix,
                kind=LeafKind.ROW if object_list_name else LeafKind.SCALAR,
                enum_values=_enum_values(node),
                object_list_name=object_list_name,
            )
        )


def _walk_object_lists(
    node: Mapping[str, Any],
    *,
    prefix: str,
    parent: Optional[ObjectListSpec],
    parent_field: Optional[str],
    lists: list[ObjectListSpec],
) -> None:
    node_type = _primary_type(node)
    if node_type == "object":
        for name, child in node.get("properties", {}).items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            _walk_object_lists(
                child,
                prefix=child_prefix,
                parent=parent,
                parent_field=parent_field,
                lists=lists,
            )
        return

    if node_type != "array":
        return

    items = node.get("items", {})
    if _primary_type(items) != "object":
        return

    list_name = prefix
    spec = ObjectListSpec(
        list_name=list_name,
        by_list_key=list_name_to_by_list_key(list_name),
        parent=parent,
        parent_field=parent_field,
    )
    lists.append(spec)

    for name, child in items.get("properties", {}).items():
        child_prefix = f"{list_name}[].{name}"
        _walk_object_lists(
            child,
            prefix=child_prefix,
            parent=spec,
            parent_field=name,
            lists=lists,
        )


def _primary_type(node: Mapping[str, Any]) -> Optional[str]:
    raw = node.get("type")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        for candidate in raw:
            if candidate != "null":
                return candidate
    return None


def _enum_values(node: Mapping[str, Any]) -> Optional[tuple[str, ...]]:
    raw = node.get("enum")
    if not isinstance(raw, list) or not raw:
        return None
    if not all(isinstance(value, str) for value in raw):
        return None
    return tuple(raw)
