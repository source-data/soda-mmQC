#!/usr/bin/env python3
"""Sync prompt files to Langfuse for changed checks.

Usage:
    python .github/scripts/sync_langfuse_checklist.py <comma-separated-files|ALL>

How this script works:
1. Read changed files (or discover all files when input is "ALL").
2. Group files by check directory.
3. Build one Langfuse config dictionary per check by loading every JSON file
   in that check directory, keyed by JSON filename.
4. Upload prompt text files for each affected check.

Prompt key format:
    checklists/{checklist_name}/{check_name}
"""

import json
import os
import sys
from pathlib import Path

CHECKLIST_ROOT = Path(os.environ.get("CHECKLIST_ROOT", "soda_mmqc/data/checklist"))
WATCHED_CHECKLISTS = {
    x.strip()
    for x in os.environ.get("WATCHED_CHECKLISTS", "fig-checklist,doc-checklist").split(",")
    if x.strip()
}


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
    """Return every .txt and .json under watched checklist directories."""
    files: list[str] = []
    for checklist in WATCHED_CHECKLISTS:
        checklist_dir = CHECKLIST_ROOT / checklist
        if not checklist_dir.exists():
            continue
        for f in checklist_dir.rglob("*.txt"):
            files.append(str(f))
        for f in checklist_dir.rglob("*.json"):
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
    """Read a UTF-8 text file and return its contents."""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def load_check_json_config(check_dir: Path) -> dict:
    """Load all JSON files in a check directory into one config dictionary.

    Output format:
        {
            "schema": {...},
            "model_config": {...},
            ...
        }
    """
    config: dict = {}
    for json_path in sorted(check_dir.glob("*.json")):
        try:
            config[json_path.stem] = json.loads(load_text(json_path))
        except Exception as exc:
            print(f"  Warning: failed to parse {json_path.name} for {check_dir.name}: {exc}")
    return config


# ---------------------------------------------------------------------------
# Sync logic
# ---------------------------------------------------------------------------

def sync_check(
    langfuse,
    checklist_name: str,
    check_name: str,
    only_txts: list[Path] | None = None,
) -> None:
    """Create Langfuse prompt version(s) for one check.

    Args:
        only_txts: If given, only upload those specific .txt files.
                   If None, upload ALL prompt*.txt files for the check.
    """
    check_dir = CHECKLIST_ROOT / checklist_name / check_name
    prompt_key = f"checklists/{checklist_name}/{check_name}"

    # Bundle all JSON files for this check into the Langfuse config payload.
    config = load_check_json_config(check_dir)

    # Determine which prompt text files to upload.
    if only_txts is not None:
        txt_files = sorted(only_txts)
    else:
        # Support both source layout (prompts/prompt*.txt)
        # and snapshot layout (prompt*.txt in check root).
        txt_files = sorted((check_dir / "prompts").glob("prompt*.txt"))
        if not txt_files:
            txt_files = sorted(check_dir.glob("prompt*.txt"))
        if not txt_files:
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

        # Group by check; any JSON change triggers a full prompt re-upload for that check.
        json_changed: set[tuple[str, str]] = set()
        txt_changed: dict[tuple[str, str], list[Path]] = {}

        for fp in changed_files:
            info = extract_check_info(fp)
            if info is None:
                print(f"  Skipping (not a watched checklist file): {fp}")
                continue
            key = info
            p = Path(fp)
            if p.suffix == ".json":
                # Any JSON change in this check re-uploads all prompts.
                json_changed.add(key)
            elif p.suffix == ".txt":
                txt_changed.setdefault(key, []).append(p)

        # Build the sync plan:
        #   JSON changed  -> re-upload all prompts for that check
        #   only TXT changed -> upload only the changed prompt files
        all_keys = json_changed | set(txt_changed.keys())
        checks_to_sync: dict[tuple[str, str], list[Path] | None] = {}
        for key in all_keys:
            if key in json_changed:
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
