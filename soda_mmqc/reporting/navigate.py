"""Navigate gold/pred JSON by explicit key and index steps."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence, TypeAlias

PathStep: TypeAlias = str | int

_SEGMENT_RE = re.compile(r"^([A-Za-z_][\w]*)\[(\d+)\]$")


class NavigationError(LookupError):
    """Failed to resolve ``steps`` in a document."""

    def __init__(
        self,
        message: str,
        *,
        side: str | None = None,
        doc_id: str | None = None,
        steps: Sequence[PathStep] | None = None,
        step_index: int | None = None,
        step: PathStep | None = None,
    ) -> None:
        parts = [message]
        if doc_id is not None:
            parts.append(f"doc_id={doc_id!r}")
        if side is not None:
            parts.append(f"side={side}")
        if steps is not None:
            parts.append(f"steps={list(steps)!r}")
        if step_index is not None:
            parts.append(f"at step {step_index}")
        if step is not None:
            parts.append(f"step={step!r}")
        super().__init__("; ".join(parts))
        self.side = side
        self.doc_id = doc_id
        self.steps = tuple(steps) if steps is not None else None
        self.step_index = step_index
        self.step = step


def path_string_to_steps(path: str) -> tuple[PathStep, ...]:
    """Convert a table ``path`` or ``context_path`` string to navigation steps.

    Example: ``outputs[7].scale_bar_on_image`` →
    ``("outputs", 7, "scale_bar_on_image")``.
    """
    if not path:
        raise ValueError("path must be non-empty")
    steps: list[PathStep] = []
    for segment in path.split("."):
        match = _SEGMENT_RE.match(segment)
        if match:
            steps.append(match.group(1))
            steps.append(int(match.group(2)))
        else:
            steps.append(segment)
    return tuple(steps)


def get_at_steps(
    doc: Mapping[str, Any],
    steps: Sequence[PathStep],
    *,
    side: str | None = None,
    doc_id: str | None = None,
) -> Any:
    """Return the value at ``steps`` in ``doc``.

    Use string keys for mappings and integer indices for sequences.
    """
    current: Any = doc
    for index, step in enumerate(steps):
        try:
            if isinstance(step, int):
                if not isinstance(current, Sequence) or isinstance(
                    current, str
                ):
                    raise NavigationError(
                        "expected sequence at step "
                        f"{index}, got {type(current).__name__}",
                        side=side,
                        doc_id=doc_id,
                        steps=steps,
                        step_index=index,
                        step=step,
                    )
                current = current[step]
            else:
                if not isinstance(current, Mapping):
                    raise NavigationError(
                        "expected mapping at step "
                        f"{index}, got {type(current).__name__}",
                        side=side,
                        doc_id=doc_id,
                        steps=steps,
                        step_index=index,
                        step=step,
                    )
                current = current[step]
        except (KeyError, IndexError, TypeError) as exc:
            raise NavigationError(
                f"navigation failed: {exc}",
                side=side,
                doc_id=doc_id,
                steps=steps,
                step_index=index,
                step=step,
            ) from exc
    return current


def get_at_steps_optional(
    doc: Mapping[str, Any] | None,
    steps: Sequence[PathStep],
    *,
    side: str | None = None,
    doc_id: str | None = None,
) -> Any | None:
    """Like :func:`get_at_steps` but returns ``None`` if ``doc`` is missing."""
    if doc is None:
        return None
    try:
        return get_at_steps(doc, steps, side=side, doc_id=doc_id)
    except NavigationError:
        return None


def parent_row_steps(steps: Sequence[PathStep]) -> tuple[PathStep, ...] | None:
    """Return parent row steps when ``steps`` ends with a leaf field name."""
    if len(steps) < 2:
        return None
    if not isinstance(steps[-1], str):
        return None
    if not isinstance(steps[-2], int):
        return None
    return tuple(steps[:-1])


def layer_s_row_steps(
    *,
    list_key: str,
    context_path: str | None,
    index: int | None,
) -> tuple[PathStep, ...] | None:
    """Build navigation steps for a Layer S row at ``index``."""
    if index is None:
        return None
    prefix = context_path or list_key
    if "[" in prefix or "." in prefix:
        base: tuple[PathStep, ...] = path_string_to_steps(prefix)
    else:
        base = (prefix,)
    return (*base, index)
