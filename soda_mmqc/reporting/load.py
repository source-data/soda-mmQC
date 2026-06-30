"""Load flat evaluation runs from analysis.json on disk."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from soda_mmqc import logger
from soda_mmqc.config import CHECKLIST_DIR, EVALUATION_DIR
from soda_mmqc.core.eval_manifest import EvalManifest, load_eval_manifest

_PROMPT_RE = re.compile(r"prompt\.(\d+)")


def normalize_prompt_name(prompt_name: str) -> str:
    """Normalize analysis prompt keys to ``prompt.N``."""
    if prompt_name.startswith("prompt."):
        return prompt_name

    match = _PROMPT_RE.search(prompt_name)
    if match:
        return f"prompt.{match.group(1)}"

    if "::version::" in prompt_name:
        try:
            version_num = int(prompt_name.split("::version::")[-1])
            return f"prompt.{version_num + 1}"
        except (ValueError, IndexError):
            pass

    trailing = re.search(r"(\d+)$", prompt_name)
    if trailing:
        return f"prompt.{int(trailing.group(1))}"

    logger.warning("Could not normalize prompt name: %s", prompt_name)
    return prompt_name


def _as_list(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _prompt_matches(raw_key: str, allowed: list[str] | None) -> bool:
    normalized = normalize_prompt_name(raw_key)
    return allowed is None or normalized in allowed


@dataclass(frozen=True)
class FlatRecord:
    """One per-doc flat analysis record."""

    doc_id: str | None
    metadata: dict[str, Any]
    analysis: dict[str, Any]
    expected_output: dict[str, Any] | None = None
    model_output: dict[str, Any] | None = None

    @property
    def has_payloads(self) -> bool:
        return (
            isinstance(self.expected_output, dict)
            and isinstance(self.model_output, dict)
        )


@dataclass(frozen=True)
class FlatRun:
    """One (checklist, check, model, prompt) evaluation slice."""

    checklist: str
    check: str
    model: str
    prompt: str
    records: tuple[FlatRecord, ...]
    manifest: EvalManifest


class FlatRuns(Sequence[FlatRun]):
    """Collection of loaded flat runs."""

    def __init__(self, runs: Sequence[FlatRun]) -> None:
        self._runs = tuple(runs)

    def __len__(self) -> int:
        return len(self._runs)

    def __iter__(self) -> Iterator[FlatRun]:
        return iter(self._runs)

    def __getitem__(self, index: int) -> FlatRun:
        return self._runs[index]


def _discover_models(eval_dir: Path) -> list[str]:
    models: list[str] = []
    if not eval_dir.is_dir():
        return models
    for child in sorted(eval_dir.iterdir()):
        if child.is_dir() and (child / "analysis.json").is_file():
            models.append(child.name)
    return models


def _load_analysis_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_payload(entry: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    payload = entry.get(key)
    if isinstance(payload, dict):
        return dict(payload)
    return None


def _parse_flat_records(
    entries: list[Mapping[str, Any]],
    *,
    include_payloads: bool = False,
) -> tuple[FlatRecord, ...]:
    records: list[FlatRecord] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        analysis = entry.get("analysis")
        if not isinstance(analysis, dict):
            logger.warning(
                "Skipping record without analysis dict: %s",
                entry.get("doc_id"),
            )
            continue
        expected_output = _parse_payload(entry, "expected_output") if include_payloads else None
        model_output = _parse_payload(entry, "model_output") if include_payloads else None
        records.append(
            FlatRecord(
                doc_id=entry.get("doc_id"),
                metadata=(
                    dict(entry["metadata"])
                    if isinstance(entry.get("metadata"), dict)
                    else {}
                ),
                analysis=analysis,
                expected_output=expected_output,
                model_output=model_output,
            )
        )
    return tuple(records)


def record_source(record: FlatRecord) -> str:
    """Benchmark example path for one flat record (``metadata.source``)."""
    source = record.metadata.get("source")
    if isinstance(source, str) and source:
        return source
    if record.doc_id:
        return record.doc_id
    raise ValueError("FlatRecord has no metadata.source or doc_id")


def _find_flat_entry(
    raw: Mapping[str, Any],
    *,
    prompt: str,
    source: str,
) -> dict[str, Any] | None:
    for prompt_key, prompt_payload in raw.items():
        if not _prompt_matches(prompt_key, [prompt]):
            continue
        if not isinstance(prompt_payload, dict):
            continue
        flat_entries = prompt_payload.get("flat")
        if not isinstance(flat_entries, list):
            continue
        for entry in flat_entries:
            if not isinstance(entry, dict):
                continue
            metadata = entry.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if metadata.get("source") == source:
                return entry
    return None


def load_record_payloads(
    checklist: str,
    check: str,
    model: str,
    prompt: str,
    source: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load ``expected_output`` and ``model_output`` for one flat record."""
    analysis_path = EVALUATION_DIR / checklist / check / model / "analysis.json"
    if not analysis_path.is_file():
        raise FileNotFoundError(f"Missing analysis file: {analysis_path}")

    raw = _load_analysis_file(analysis_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected object at root of {analysis_path}")

    entry = _find_flat_entry(raw, prompt=prompt, source=source)
    if entry is None:
        raise KeyError(
            f"No flat record for source={source!r} prompt={prompt!r} in {analysis_path}"
        )

    expected_output = _parse_payload(entry, "expected_output")
    model_output = _parse_payload(entry, "model_output")
    if expected_output is None or model_output is None:
        raise KeyError(
            f"Record {source!r} in {analysis_path} is missing expected_output or model_output"
        )
    return expected_output, model_output


def find_record(summary_records: Sequence[Any], *, source: str) -> FlatRecord:
    """Return the :class:`FlatRecord` matching ``metadata.source``."""
    for record in summary_records:
        if record_source(record) == source:
            return record
    raise KeyError(f"No record with source={source!r}")


def ensure_record_payloads(
    record: FlatRecord,
    *,
    checklist: str,
    check: str,
    model: str,
    prompt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return gold/pred payloads, loading from disk when not already on the record."""
    if record.has_payloads:
        assert record.expected_output is not None
        assert record.model_output is not None
        return record.expected_output, record.model_output
    return load_record_payloads(
        checklist,
        check,
        model,
        prompt,
        record_source(record),
    )


def load_eval_manifest_for_check(checklist: str, check: str) -> EvalManifest:
    """Load ``eval-manifest.json`` for a checklist check."""
    path = eval_manifest_path(checklist, check)
    if not path.is_file():
        raise FileNotFoundError(f"Missing eval manifest: {path}")
    return load_eval_manifest(path)


def load_prompt_text(checklist: str, check: str, prompt: str) -> str | None:
    """Load prompt text for a normalized prompt label (e.g. ``prompt.2``)."""
    check_dir = CHECKLIST_DIR / checklist / check
    prompt_path = check_dir / "prompts" / f"{prompt}.txt"
    if prompt_path.is_file():
        return prompt_path.read_text(encoding="utf-8")

    for candidate in (check_dir / "prompt.txt", check_dir / "prompts" / "prompt.txt"):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")

    logger.warning(
        "No prompt text found for %s / %s / %s under %s",
        checklist,
        check,
        prompt,
        check_dir,
    )
    return None


@dataclass(frozen=True)
class EvaluationCheckRef:
    """A checklist check that has at least one ``analysis.json`` on disk."""

    checklist: str
    check: str
    has_manifest: bool = True


def eval_manifest_path(checklist: str, check: str) -> Path:
    """Path to ``eval-manifest.json`` for a checklist check."""
    return CHECKLIST_DIR / checklist / check / "eval-manifest.json"


def check_has_eval_manifest(checklist: str, check: str) -> bool:
    """Return whether the check has an on-disk eval manifest."""
    return eval_manifest_path(checklist, check).is_file()


def try_load_run_summaries(
    checklist: str,
    check: str,
) -> tuple[RunSummaries | None, str | None]:
    """Load aggregated run summaries, or return ``(None, user_message)``."""
    from soda_mmqc.reporting.aggregate import RunSummaries, summarize_runs

    manifest_path = eval_manifest_path(checklist, check)
    if not manifest_path.is_file():
        return None, (
            "This check has evaluation output but no **eval-manifest.json**. "
            f"Add a manifest at `{manifest_path}`, then re-run `evaluate` if needed."
        )

    eval_dir = EVALUATION_DIR / checklist / check
    if not eval_dir.is_dir():
        return None, f"No evaluation directory at `{eval_dir}`. Run `evaluate` first."

    try:
        runs = load_flat_runs(checklist, check)
    except FileNotFoundError as exc:
        return None, str(exc)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"Could not load evaluation data: {exc}"

    if len(runs) == 0:
        return None, (
            f"No flat runs loaded under `{eval_dir}`. "
            "Check that `analysis.json` exists for at least one model and prompt."
        )

    return summarize_runs(runs), None


def discover_evaluation_checks() -> tuple[EvaluationCheckRef, ...]:
    """List checks under ``EVALUATION_DIR`` with evaluation results."""
    refs: list[EvaluationCheckRef] = []
    if not EVALUATION_DIR.is_dir():
        return ()
    for checklist_dir in sorted(EVALUATION_DIR.iterdir()):
        if not checklist_dir.is_dir():
            continue
        for check_dir in sorted(checklist_dir.iterdir()):
            if not check_dir.is_dir():
                continue
            models = _discover_models(check_dir)
            if models:
                refs.append(
                    EvaluationCheckRef(
                        checklist=checklist_dir.name,
                        check=check_dir.name,
                        has_manifest=check_has_eval_manifest(
                            checklist_dir.name,
                            check_dir.name,
                        ),
                    )
                )
    return tuple(refs)


def load_flat_runs(
    checklist: str,
    check: str,
    *,
    models: str | Sequence[str] | None = None,
    prompts: str | Sequence[str] | None = None,
    include_payloads: bool = False,
) -> FlatRuns:
    """Load flat analysis runs from ``EVALUATION_DIR``.

    Each ``(model, prompt)`` pair yields one :class:`FlatRun`. Missing files
    are skipped with a warning.

    When ``include_payloads`` is false (default), ``expected_output`` and
    ``model_output`` are omitted from :class:`FlatRecord` for lighter loads.
    """
    manifest = load_eval_manifest_for_check(checklist, check)
    eval_dir = EVALUATION_DIR / checklist / check
    model_list = _as_list(models) or _discover_models(eval_dir)
    prompt_filter = _as_list(prompts)

    if not model_list:
        logger.warning("No model directories found under %s", eval_dir)
        return FlatRuns(())

    runs: list[FlatRun] = []
    for model in model_list:
        analysis_path = eval_dir / model / "analysis.json"
        if not analysis_path.is_file():
            logger.warning("Missing analysis file: %s", analysis_path)
            continue

        try:
            raw = _load_analysis_file(analysis_path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load %s: %s", analysis_path, exc)
            continue

        if not isinstance(raw, dict):
            logger.warning("Expected object at root of %s", analysis_path)
            continue

        for prompt_key, prompt_payload in raw.items():
            if not _prompt_matches(prompt_key, prompt_filter):
                continue
            if not isinstance(prompt_payload, dict):
                logger.warning("Unexpected payload for prompt %s", prompt_key)
                continue
            flat_entries = prompt_payload.get("flat")
            if not isinstance(flat_entries, list):
                logger.warning(
                    "Prompt %s in %s has no 'flat' array; skipping",
                    prompt_key,
                    analysis_path,
                )
                continue

            runs.append(
                FlatRun(
                    checklist=checklist,
                    check=check,
                    model=model,
                    prompt=normalize_prompt_name(prompt_key),
                    records=_parse_flat_records(
                        flat_entries,
                        include_payloads=include_payloads,
                    ),
                    manifest=manifest,
                )
            )

    return FlatRuns(runs)
