#!/usr/bin/env python3
"""Build Langfuse prompts using a production manifest file.

Usage:
    python .github/scripts/build_langfuse_checklist.py [manifest-path]

Default manifest path:
    soda_mmqc/data/production_prompts/production.json

Manifest expectations:
- path_templates: format-string templates that resolve repository file paths
- checks: per-check records containing values used to render templates

Per check, the script:
1. Renders the prompt path from path_templates.prompt.
2. Renders all JSON paths from templates ending in .json.
3. Loads JSON files into one config payload keyed by filename stem.
4. Uploads one Langfuse prompt version with the rendered prompt/config.
"""

import json
import os
import sys
from pathlib import Path
from string import Formatter
from typing import Any

DEFAULT_MANIFEST_PATH = Path("soda_mmqc/data/production_prompts/production.json")


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


def load_text(path: Path) -> str:
    """Read a UTF-8 text file and return its contents."""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and minimally validate the production manifest JSON."""
    data = json.loads(load_text(manifest_path))
    if not isinstance(data, dict):
        raise ValueError("Manifest root must be a JSON object")
    if "path_templates" not in data or "checks" not in data:
        raise ValueError("Manifest must contain 'path_templates' and 'checks'")
    if not isinstance(data["path_templates"], dict):
        raise ValueError("Manifest 'path_templates' must be an object")
    if not isinstance(data["checks"], list):
        raise ValueError("Manifest 'checks' must be a list")
    return data


def template_fields(template: str) -> set[str]:
    """Extract placeholder field names from a Python format template."""
    names: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name:
            names.add(field_name)
    return names


def render_path(template: str, context: dict[str, Any], repo_root: Path) -> Path:
    """Render a repository-relative template and return an absolute path.

    The rendered path must stay within repo_root to guard against accidental
    path traversal in manifest values.
    """
    rel_path = template.format(**context)
    path = (repo_root / rel_path).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Rendered path escapes repository root: {rel_path}") from exc
    return path


def build_config_from_manifest_paths(json_paths: list[Path]) -> dict[str, Any]:
    """Build Langfuse config from manifest-resolved JSON files.

        Keys use filename stems (without .json), for example:
            schema.json -> schema
            eval-manifest.json -> eval-manifest
    """
    config: dict[str, Any] = {}
    for json_path in json_paths:
        try:
            config[json_path.stem] = json.loads(load_text(json_path))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing JSON file: {json_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {json_path}: {exc}") from exc
    return config


def enrich_prompt_context(entry: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Return template context enriched from prompt_version.

    Adds these derived fields when prompt_version is present:
    - version: same numeric/string version value for templates using {version}
    - prompt_filename: prompt.<N>.txt
    - prompt_filename_nodot: prompt<N>.txt
    """
    context = dict(entry)
    prompt_version = context.get("prompt_version")
    if prompt_version is None:
        return context, "missing prompt_version"

    version_str = str(prompt_version).strip()
    if not version_str:
        return context, "empty prompt_version"

    context.setdefault("version", prompt_version)
    context.setdefault("prompt_filename", f"prompt.{version_str}.txt")
    context.setdefault("prompt_filename_nodot", f"prompt{version_str}.txt")
    return context, None


def build_check(langfuse, checklist_name: str, check_name: str, prompt_path: Path, config: dict[str, Any]) -> None:
    """Upload one prompt/config pair for a single check."""
    prompt_key = f"checklists/{checklist_name}/{check_name}"
    prompt_text = load_text(prompt_path)
    print(f"  Uploading: {prompt_key}  ({prompt_path.name}) ...")
    langfuse.create_prompt(
        name=prompt_key,
        prompt=prompt_text,
        config=config,
        labels=["production"],
        type="text",
    )
    print(f"  Done:      {prompt_key}  ({prompt_path.name})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    manifest_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_MANIFEST_PATH
    repo_root = Path.cwd().resolve()
    abs_manifest_path = (repo_root / manifest_path).resolve() if not manifest_path.is_absolute() else manifest_path.resolve()

    print(f"Loading manifest: {abs_manifest_path}")
    manifest = load_manifest(abs_manifest_path)
    path_templates = manifest["path_templates"]
    checks = manifest["checks"]

    # Required template for the prompt text payload.
    prompt_template = path_templates.get("prompt")
    if not isinstance(prompt_template, str):
        raise ValueError("Manifest path_templates.prompt must be a string")

    # Any template ending in .json is treated as part of Langfuse config.
    json_templates = {
        key: value
        for key, value in path_templates.items()
        if isinstance(value, str) and value.endswith(".json")
    }
    if not json_templates:
        raise ValueError("No JSON templates found in manifest path_templates")

    langfuse = get_langfuse_client()
    print(f"Connected to Langfuse at {os.environ.get('LANGFUSE_BASE_URL')}\n")

    errors: list[str] = []
    synced = 0
    for entry in checks:
        # Each entry should resolve to exactly one prompt upload.
        checklist_name = entry.get("checklist")
        check_name = entry.get("check")
        if not checklist_name or not check_name:
            errors.append(f"Invalid check entry missing checklist/check: {entry}")
            continue

        # Enrich template context with fields derived from prompt_version.
        context, prompt_context_error = enrich_prompt_context(dict(entry))
        if prompt_context_error:
            print(f"Skipping {checklist_name}/{check_name}: {prompt_context_error}")
            continue

        needed_fields = template_fields(prompt_template)
        missing_prompt_fields = [f for f in needed_fields if context.get(f) is None]
        if missing_prompt_fields:
            print(
                f"Skipping {checklist_name}/{check_name}: missing values for prompt template fields {missing_prompt_fields}"
            )
            continue

        try:
            prompt_path = render_path(prompt_template, context, repo_root)
            if not prompt_path.exists():
                raise FileNotFoundError(f"Missing prompt file: {prompt_path}")

            # Resolve and validate all JSON config paths declared in path_templates.
            json_paths: list[Path] = []
            for _, template in sorted(json_templates.items()):
                fields = template_fields(template)
                missing_json_fields = [f for f in fields if context.get(f) is None]
                if missing_json_fields:
                    raise ValueError(
                        f"Missing values for JSON template fields {missing_json_fields} in check {checklist_name}/{check_name}"
                    )
                json_paths.append(render_path(template, context, repo_root))

            config = build_config_from_manifest_paths(json_paths)
            print(f"Syncing: {checklist_name}/{check_name}")
            build_check(langfuse, checklist_name, check_name, prompt_path, config)
            synced += 1
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

    if synced == 0:
        print("\nNo checks were synced.")
    else:
        print(f"\nAll done. Synced {synced} check(s).")


if __name__ == "__main__":
    main()
