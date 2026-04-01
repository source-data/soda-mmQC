#!/usr/bin/env python3
"""Sync changed checklist prompt files to Langfuse.

Usage:
    python scripts/sync_langfuse_checklist.py <comma-separated-files|ALL>

For each changed .txt or .json file under soda_mmqc/data/checklist/, this
script derives the Langfuse prompt key and creates a new prompt version.

Prompt key format:  checklists/{checklist_name}/{check_name}
  e.g.  checklists/fig-checklist/error-bars-defined
        checklists/doc-checklist/DAS-present-and-correct

Rules:
  - A changed prompt*.txt file → create a new Langfuse version from that file.
  - A changed schema.json or benchmark.json → re-sync ALL prompt*.txt files for
                                  that check so they pick up the updated config.
  - Other .json files are ignored for syncing.

Langfuse config structure per check:
  benchmark.json  → top-level config fields (name, description, example_class, examples, …)
  schema.json     → nested under the "output_schema" key
"""

import json
import os
import sys
from pathlib import Path

CHECKLIST_ROOT = Path("soda_mmqc/data/checklist")
WATCHED_CHECKLISTS = {"fig-checklist", "doc-checklist"}


# ---------------------------------------------------------------------------
# Langfuse client
# ---------------------------------------------------------------------------

def get_langfuse_client():
    from langfuse import Langfuse  # type: ignore

    return Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ["LANGFUSE_BASE_URL"],
    )


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def collect_all_files() -> list[str]:
    """Return every .txt, schema.json, and benchmark.json under the watched checklist dirs."""
    files: list[str] = []
    for checklist in WATCHED_CHECKLISTS:
        checklist_dir = CHECKLIST_ROOT / checklist
        if not checklist_dir.exists():
            continue
        for f in checklist_dir.rglob("*.txt"):
            files.append(str(f))
        for f in checklist_dir.rglob("schema.json"):
            files.append(str(f))
        for f in checklist_dir.rglob("benchmark.json"):
            files.append(str(f))
    return files


def extract_check_info(filepath: str) -> tuple[str, str] | None:
    """Return (checklist_name, check_name) from a filepath, or None."""
    p = Path(filepath)
    try:
        rel = p.relative_to(CHECKLIST_ROOT)
    except ValueError:
        return None
    parts = rel.parts  # ('fig-checklist', 'error-bars-defined', ...)
    if len(parts) < 2:
        return None
    checklist_name, check_name = parts[0], parts[1]
    if checklist_name not in WATCHED_CHECKLISTS:
        return None
    return checklist_name, check_name


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Sync logic
# ---------------------------------------------------------------------------

def sync_check(
    langfuse,
    checklist_name: str,
    check_name: str,
    only_txts: list[Path] | None = None,
) -> None:
    """Create new Langfuse prompt version(s) for a single check.

    Args:
        only_txts: If given, only upload those specific .txt files.
                   If None, upload ALL prompt*.txt files for the check.
    """
    check_dir = CHECKLIST_ROOT / checklist_name / check_name
    prompt_key = f"checklists/{checklist_name}/{check_name}"

    # Build config: benchmark.json fields at the top level, schema.json nested under "output_schema".
    benchmark_path = check_dir / "benchmark.json"
    schema_path = check_dir / "schema.json"
    config: dict = {}
    if benchmark_path.exists():
        try:
            config = json.loads(load_text(benchmark_path))
        except Exception as exc:
            print(f"  Warning: failed to parse benchmark.json for {check_name}: {exc}")
    if schema_path.exists():
        try:
            config["output_schema"] = json.loads(load_text(schema_path))
        except Exception as exc:
            print(f"  Warning: failed to parse schema.json for {check_name}: {exc}")

    # Determine which .txt files to upload
    if only_txts is not None:
        txt_files = sorted(only_txts)
    else:
        prompts_dir = check_dir / "prompts"
        if prompts_dir.exists():
            txt_files = sorted(prompts_dir.glob("prompt*.txt"))
        else:
            single = check_dir / "prompt.txt"
            txt_files = [single] if single.exists() else []

    if not txt_files:
        print(f"  Skipped {prompt_key}: no prompt .txt files found")
        return

    for txt_path in txt_files:
        prompt_text = load_text(txt_path)
        print(f"  Uploading: {prompt_key}  ({txt_path.name}) ...")
        langfuse.create_prompt(
            name=prompt_key,
            prompt=prompt_text,
            config=config,
            labels=["production"],
            type="text",
        )
        print(f"  Done:      {prompt_key}  ({txt_path.name})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: sync_langfuse_checklist.py <comma-separated-files|ALL>")
        sys.exit(1)

    arg = sys.argv[1].strip()
    if arg == "ALL":
        files = collect_all_files()
        print(f"Syncing ALL checklist files ({len(files)} file(s)) ...")
        # For a full resync, process every check completely.
        seen: set[tuple[str, str]] = set()
        for fp in files:
            info = extract_check_info(fp)
            if info:
                seen.add(info)
        checks_to_sync: dict[tuple[str, str], list[Path] | None] = {
            k: None for k in seen
        }
    else:
        changed_files = [f.strip() for f in arg.split(",") if f.strip()]
        print(f"Processing {len(changed_files)} changed file(s) ...")

        # Group by check; note whether schema.json changed for that check.
        schema_changed: set[tuple[str, str]] = set()
        txt_changed: dict[tuple[str, str], list[Path]] = {}

        for fp in changed_files:
            info = extract_check_info(fp)
            if info is None:
                print(f"  Skipping (not a watched checklist file): {fp}")
                continue
            key = info
            p = Path(fp)
            if p.name in ("schema.json", "benchmark.json"):
                # Config files changed → re-upload all prompts for this check.
                schema_changed.add(key)
            elif p.suffix == ".txt":
                txt_changed.setdefault(key, []).append(p)
            # Any other .json files are ignored.

        # Build the sync plan:
        #   schema changed → re-upload ALL prompts for that check
        #   only .txt changed → upload only those specific files
        all_keys = schema_changed | set(txt_changed.keys())
        checks_to_sync: dict[tuple[str, str], list[Path] | None] = {}
        for key in all_keys:
            if key in schema_changed:
                checks_to_sync[key] = None  # None = sync all
            else:
                checks_to_sync[key] = txt_changed[key]

    if not checks_to_sync:
        print("No relevant checklist files to sync.")
        return

    langfuse = get_langfuse_client()
    print(f"Connected to Langfuse at {os.environ.get('LANGFUSE_BASE_URL')}\n")

    errors: list[str] = []
    for (checklist_name, check_name), only_txts in sorted(checks_to_sync.items()):
        print(f"Syncing: {checklist_name}/{check_name}")
        try:
            sync_check(langfuse, checklist_name, check_name, only_txts)
        except Exception as exc:
            msg = f"ERROR syncing {checklist_name}/{check_name}: {exc}"
            print(f"  {msg}")
            errors.append(msg)

    langfuse.flush()

    if errors:
        print(f"\n{len(errors)} error(s) occurred:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\nAll done.")


if __name__ == "__main__":
    main()
