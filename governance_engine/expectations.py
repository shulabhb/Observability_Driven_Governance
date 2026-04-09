"""Load and merge expectations YAML (defaults + per-archetype)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPECTATIONS_PATH = WORKSPACE_ROOT / "config" / "expectations.yaml"


def load_expectations(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_EXPECTATIONS_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Expectations file not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Expectations root must be a mapping")
    return data


def merged_bands_for_archetype(config: dict[str, Any], archetype: str) -> dict[str, Any]:
    """Defaults overridden by per_archetype[archetype]."""
    defaults = deepcopy(config.get("defaults") or {})
    per = (config.get("per_archetype") or {}).get(archetype) or {}
    if not isinstance(per, dict):
        per = {}
    defaults.update(per)
    return defaults
