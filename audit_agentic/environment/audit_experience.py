"""Load human-reviewer "audit experience" notes, keyed by source.

Used by the audit-experience prompt variants to inject reviewer-summarized
labeling heuristics for sources whose rule text / post boundary is subjective.
Data lives in
``environment/cache/experience/audit_experience.json`` and is edited by hand
by the audit/review team — this module only reads it.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

_DEFAULT_PATH = Path(__file__).resolve().parent / "cache" / "experience" / "audit_experience.json"

_lock = threading.Lock()
_cache: Optional[Dict[str, dict]] = None
_cache_path: Optional[Path] = None

_TRUE_VALUES = {"1", "true", "yes", "on", "all"}
_AUTO_VALUES = {"auto", "source", "per_source", "personalized"}


def _experience_path() -> Path:
    raw = os.environ.get("AUDIT_EXPERIENCE_PATH")
    return Path(raw) if raw else _DEFAULT_PATH


def _load(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, dict)}


def _get_cache() -> Dict[str, dict]:
    global _cache, _cache_path
    path = _experience_path()
    with _lock:
        if _cache is None or _cache_path != path:
            _cache = _load(path)
            _cache_path = path
        return _cache


def load_audit_experience_for_rule_previews(
    source: Optional[str],
    candidate_labels: list[str],
    rule_previews: list[dict],
) -> Tuple[str, list[dict]]:
    """Optionally place matched label notes beside previews, without duplication."""
    entry = _get_cache().get(source) if source else None
    if (not entry or not entry.get("enabled", True)
            or entry.get("label_experience_placement") != "rule_preview"):
        return load_audit_experience(source, labels=candidate_labels), rule_previews

    allowed = set(candidate_labels)
    label_notes = entry.get("labels") or {}
    previews = []
    for original in rule_previews:
        item = dict(original)
        label = item["label"]
        note = str(label_notes.get(label) or "").strip() if label in allowed else ""
        if note:
            # Append after preview truncation so the requested guidance survives.
            item["preview"] += "\n\n【该标签的审核经验（参考，不是规则原文）】\n" + note
            item["label_experience"] = note
        previews.append(item)
    return load_audit_experience(source), previews


def reload_audit_experience() -> None:
    """Force re-read from disk on next lookup (e.g. after editing the JSON)."""
    global _cache, _cache_path
    with _lock:
        _cache = None
        _cache_path = None


def planner_experience_enabled(
    source: Optional[str],
    mode: Optional[str] = None,
) -> bool:
    """Return whether Planner experience should be injected for ``source``.

    ``AUDIT_PLANNER_EXPERIENCE`` supports three modes:

    - false-like values: disable Planner experience for every source;
    - true-like values / ``all``: enable it for every enabled experience entry;
    - ``auto``: enable it only when that entry has ``planner_enabled: true``.

    Final experience remains controlled by the selected Final prompt template.
    """
    if not source:
        return False

    raw_mode = (
        mode if mode is not None else os.environ.get("AUDIT_PLANNER_EXPERIENCE", "")
    )
    normalized_mode = str(raw_mode).strip().lower()
    entry = _get_cache().get(source)
    if not entry or not entry.get("enabled", True):
        return False
    if normalized_mode in _TRUE_VALUES:
        return True
    if normalized_mode in _AUTO_VALUES:
        return bool(entry.get("planner_enabled", False))
    return False


def load_audit_experience(
    source: Optional[str],
    labels: Optional[list] = None,
    *,
    include_all_labels: bool = False,
) -> str:
    """Return the reviewer-experience text for ``source``, or "" if absent/disabled.

    If ``labels`` is given, label-specific notes (``entry["labels"][label]``)
    for any label present in ``labels`` are appended after the general text.
    If ``include_all_labels`` is true, every configured label note for the
    source is appended in JSON insertion order. This is intended for Planner
    experiments where experience should influence filtering across the whole
    candidate-label set. Final keeps passing its narrowed candidate labels.
    """
    if not source:
        return ""
    entry = _get_cache().get(source)
    if not entry or not entry.get("enabled", True):
        return ""

    parts = []
    # Planner sees every candidate label at once, so feeding it all detailed
    # Final notes can make the prompt longer than the rules themselves. Allow
    # entries to provide a compact Planner-only summary while preserving the
    # detailed label boundaries used by Final.
    text_key = "planner_text" if include_all_labels and entry.get("planner_text") else "text"
    text = str(entry.get(text_key) or "").strip()
    if text:
        parts.append(text)

    if include_all_labels and entry.get("planner_labels"):
        label_notes = entry.get("planner_labels") or {}
    else:
        label_notes = entry.get("labels") or {}
    selected_labels = list(label_notes) if include_all_labels else list(labels or [])
    if selected_labels and label_notes:
        seen = set()
        for label in selected_labels:
            if label in seen:
                continue
            seen.add(label)
            note = label_notes.get(label)
            if note and str(note).strip():
                parts.append(f"【{label}】{str(note).strip()}")

    return "\n\n".join(parts)
