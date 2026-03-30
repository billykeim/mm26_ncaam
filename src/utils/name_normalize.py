"""Team name normalization using ``team_name_map.json``."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.utils.constants import TEAM_NAME_MAP_PATH


def load_team_name_map(path: Path | None = None) -> dict[str, Any]:
    """Load canonical team crosswalk JSON."""
    p = path or TEAM_NAME_MAP_PATH
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def torvik_to_canonical(team: str, mapping: dict[str, Any]) -> str:
    """Map a Torvik ``team`` string to canonical name; fallback to stripped input."""
    t = str(team).strip()
    for _canon, entry in mapping.items():
        if isinstance(entry, dict) and entry.get("torvik") == t:
            return str(entry.get("canonical", _canon))
    # direct canonical key match
    if t in mapping:
        return t
    return t


def school_to_canonical(school: str, mapping: dict[str, Any]) -> str:
    """Map Sports-Reference school string to canonical using ``sports_ref`` field."""
    s = str(school).strip()
    for _canon, entry in mapping.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("sports_ref") == s or entry.get("canonical") == s:
            return str(entry.get("canonical", _canon))
    return s


def slugify_cbbpy(name: str) -> str:
    """Lowercase hyphenated slug similar to ESPN team keys."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
    return s[:120] or "unknown"


def add_torvik_team(
    mapping: dict[str, Any],
    torvik_name: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Insert a new canonical entry for a Torvik name not yet covered."""
    t = str(torvik_name).strip()
    if not t:
        return mapping, None
    for _k, entry in mapping.items():
        if isinstance(entry, dict) and entry.get("torvik") == t:
            return mapping, None
    canon = t
    new_entry = {
        "canonical": canon,
        "torvik": t,
        "sports_ref": t,
        "cbbpy": slugify_cbbpy(t),
    }
    mapping[canon] = new_entry
    return mapping, new_entry
