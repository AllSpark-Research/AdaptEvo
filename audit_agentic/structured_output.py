"""Dynamic JSON Schema helpers for audit-agent structured outputs."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List


ROUTER_DUPLICATE_PENALTY_PER_ITEM = 0.01
ROUTER_DUPLICATE_PENALTY_MAX = 0.02
ROUTER_MISSING_PENALTY_PER_ITEM = 0.01
ROUTER_MISSING_PENALTY_MAX = 0.02
ROUTER_ABNORMAL_WHITESPACE_PENALTY = 0.01
ROUTER_QUALITY_PENALTY_MAX = 0.05
LATER_STAGE_QUALITY_PENALTY_MAX = 0.03


def structured_output_enabled(client: Any = None) -> bool:
    """Resolve structured-output enablement for sync and async clients."""
    raw = os.environ.get("AUDIT_STRUCTURED_OUTPUT")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    config = getattr(client, "config", None)
    return bool(getattr(config, "structured_output", True))


def router_bypass_max_labels() -> int:
    return max(0, int(os.environ.get("AUDIT_ROUTER_BYPASS_MAX_LABELS", "8")))


def router_min_labels() -> int:
    return max(1, int(os.environ.get("AUDIT_ROUTER_MIN_LABELS", "8")))


def router_max_labels() -> int:
    return max(router_min_labels(), int(os.environ.get("AUDIT_ROUTER_MAX_LABELS", "12")))


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def normalize_decision_basis(value: Any) -> List[str]:
    """Normalize a final rationale without iterating strings by character."""
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        return [
            text
            for item in value
            if (text := str(item).strip())
        ]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def json_schema_response_format(name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def router_response_format(candidate_labels: Iterable[str]) -> Dict[str, Any]:
    labels = _dedupe(candidate_labels)
    # Require a small useful shortlist while leaving room for the model to
    # finish naturally. The desired 8-12 unique labels are enforced softly by
    # the prompt/reward; a hard minItems=8 can force constrained decoding to
    # emit duplicate labels or long whitespace runs when it wants to close.
    minimum = min(4, len(labels))
    maximum = min(router_max_labels(), len(labels))
    return json_schema_response_format(
        "audit_router_output",
        {
            "type": "object",
            "properties": {
                "brief_analysis": {
                    "type": "string",
                    "description": "简要分析帖子内容与候选标签的相关性并给出需要保留的易混标签。",
                },
                "shortlist_labels": {
                    "type": "array",
                    "minItems": minimum,
                    "maxItems": maximum,
                    "items": {"type": "string", "enum": labels},
                },
            },
            "required": ["brief_analysis", "shortlist_labels"],
            "additionalProperties": False,
        },
    )


def router_output_quality(
    raw_response: str,
    parsed_output: Dict[str, Any],
    raw_labels: Iterable[Any],
    expected_min: int,
) -> Dict[str, float]:
    """Return a small, bounded quality penalty for valid Router JSON.

    Parse failures are handled by the existing hard format penalty. This helper
    only distinguishes clean structured output from parseable-but-degenerate
    output, so it deliberately caps the total penalty at 0.05.
    """
    if not parsed_output or parsed_output.get("_parse_fallback"):
        return _empty_output_quality()

    labels = [str(label).strip() for label in raw_labels if str(label).strip()]
    unique_labels = _dedupe(labels)
    duplicate_excess = max(0, len(labels) - len(unique_labels))
    missing_unique_count = max(0, int(expected_min) - len(unique_labels))

    duplicate_penalty = min(
        ROUTER_DUPLICATE_PENALTY_MAX,
        ROUTER_DUPLICATE_PENALTY_PER_ITEM * duplicate_excess,
    )
    missing_penalty = min(
        ROUTER_MISSING_PENALTY_MAX,
        ROUTER_MISSING_PENALTY_PER_ITEM * missing_unique_count,
    )

    layout = _output_layout_quality(raw_response, parsed_output)
    whitespace_penalty = layout["whitespace_penalty"]

    penalty = min(
        ROUTER_QUALITY_PENALTY_MAX,
        duplicate_penalty + missing_penalty + whitespace_penalty,
    )
    return {
        "penalty": float(penalty),
        "duplicate_penalty": float(duplicate_penalty),
        "missing_penalty": float(missing_penalty),
        "whitespace_penalty": float(whitespace_penalty),
        "duplicate_excess": float(duplicate_excess),
        "missing_unique_count": float(missing_unique_count),
        "raw_item_count": float(len(labels)),
        "unique_item_count": float(len(unique_labels)),
        **layout,
    }


def planner_output_quality(
    raw_response: str,
    parsed_output: Dict[str, Any],
) -> Dict[str, float]:
    """Penalize parseable Planner JSON with repeated labels or bad layout."""
    if not parsed_output or parsed_output.get("_parse_fallback"):
        return _empty_output_quality()

    filtered = parsed_output.get("filtered_labels") or {}
    tool_required = filtered.get("tool_required_labels") or []
    possible = filtered.get("possible_labels") or []

    def labels_of(entries: Iterable[Any]) -> List[str]:
        return [
            str(entry.get("label", "")).strip()
            for entry in entries
            if isinstance(entry, dict) and str(entry.get("label", "")).strip()
        ]

    tool_labels = labels_of(tool_required)
    possible_labels = labels_of(possible)
    labels = tool_labels + possible_labels
    duplicate_excess = max(0, len(labels) - len(_dedupe(labels)))
    cross_bucket_duplicate_count = len(set(tool_labels) & set(possible_labels))
    duplicate_penalty = min(
        ROUTER_DUPLICATE_PENALTY_MAX,
        ROUTER_DUPLICATE_PENALTY_PER_ITEM * duplicate_excess,
    )
    layout = _output_layout_quality(raw_response, parsed_output)
    penalty = min(
        LATER_STAGE_QUALITY_PENALTY_MAX,
        duplicate_penalty + layout["whitespace_penalty"],
    )
    return {
        "penalty": float(penalty),
        "duplicate_penalty": float(duplicate_penalty),
        "missing_penalty": 0.0,
        "duplicate_excess": float(duplicate_excess),
        "cross_bucket_duplicate_count": float(cross_bucket_duplicate_count),
        "raw_item_count": float(len(labels)),
        "unique_item_count": float(len(_dedupe(labels))),
        "missing_unique_count": 0.0,
        **layout,
    }


def final_output_quality(
    raw_response: str,
    parsed_output: Dict[str, Any],
) -> Dict[str, float]:
    """Penalize parseable Final JSON with repeated labels or bad layout."""
    if not parsed_output or parsed_output.get("_parse_fallback"):
        return _empty_output_quality()

    raw_labels = parsed_output.get("predict_label") or []
    labels = [str(label).strip() for label in raw_labels if str(label).strip()]
    duplicate_excess = max(0, len(labels) - len(_dedupe(labels)))
    duplicate_penalty = min(
        ROUTER_DUPLICATE_PENALTY_MAX,
        ROUTER_DUPLICATE_PENALTY_PER_ITEM * duplicate_excess,
    )
    layout = _output_layout_quality(raw_response, parsed_output)
    penalty = min(
        LATER_STAGE_QUALITY_PENALTY_MAX,
        duplicate_penalty + layout["whitespace_penalty"],
    )
    return {
        "penalty": float(penalty),
        "duplicate_penalty": float(duplicate_penalty),
        "missing_penalty": 0.0,
        "duplicate_excess": float(duplicate_excess),
        "cross_bucket_duplicate_count": 0.0,
        "raw_item_count": float(len(labels)),
        "unique_item_count": float(len(_dedupe(labels))),
        "missing_unique_count": 0.0,
        **layout,
    }


def _output_layout_quality(
    raw_response: str,
    parsed_output: Dict[str, Any],
) -> Dict[str, float]:
    raw = str(raw_response or "")
    max_newline_run = max((len(run) for run in re.findall(r"\n+", raw)), default=0)
    compact = json.dumps(parsed_output, ensure_ascii=False, separators=(",", ":"))
    compact_length = len(compact)
    inflated_whitespace = len(raw) > compact_length * 1.5 + 64
    abnormal_whitespace = bool(max_newline_run >= 3 or inflated_whitespace)
    whitespace_penalty = (
        ROUTER_ABNORMAL_WHITESPACE_PENALTY if abnormal_whitespace else 0.0
    )
    return {
        "whitespace_penalty": float(whitespace_penalty),
        "abnormal_whitespace": float(abnormal_whitespace),
        "max_newline_run": float(max_newline_run),
        "raw_length": float(len(raw)),
        "compact_length": float(compact_length),
    }


def _empty_output_quality() -> Dict[str, float]:
    return {
        "penalty": 0.0,
        "duplicate_penalty": 0.0,
        "missing_penalty": 0.0,
        "whitespace_penalty": 0.0,
        "duplicate_excess": 0.0,
        "cross_bucket_duplicate_count": 0.0,
        "missing_unique_count": 0.0,
        "raw_item_count": 0.0,
        "unique_item_count": 0.0,
        "abnormal_whitespace": 0.0,
        "max_newline_run": 0.0,
        "raw_length": 0.0,
        "compact_length": 0.0,
    }


def planner_response_format(
    shortlist_labels: Iterable[str],
    tool_names: Iterable[str],
) -> Dict[str, Any]:
    labels = _dedupe(shortlist_labels)
    tools = _dedupe(tool_names)
    tool_item: Dict[str, Any] = {"type": "string"}
    if tools:
        tool_item["enum"] = tools

    label_item = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": labels},
            "reason": {"type": "string"},
        },
        "required": ["label", "reason"],
        "additionalProperties": False,
    }
    tool_required_item = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": labels},
            "reason": {"type": "string"},
            "required_tools": {
                "type": "array",
                "minItems": 1,
                "items": tool_item,
            },
        },
        "required": ["label", "reason", "required_tools"],
        "additionalProperties": False,
    }
    return json_schema_response_format(
        "audit_planner_output",
        {
            "type": "object",
            "properties": {
                "brief_analysis": {
                    "type": "string",
                    "description": "概括仍需关注的风险方向和工具需求。",
                },
                "filtered_labels": {
                    "type": "object",
                    "properties": {
                        "tool_required_labels": {
                            "type": "array",
                            "items": tool_required_item,
                        },
                        "possible_labels": {"type": "array", "items": label_item},
                    },
                    "required": ["tool_required_labels", "possible_labels"],
                    "additionalProperties": False,
                },
            },
            "required": ["brief_analysis", "filtered_labels"],
            "additionalProperties": False,
        },
    )


def final_response_format(candidate_labels: Iterable[str]) -> Dict[str, Any]:
    labels = _dedupe(candidate_labels)
    return json_schema_response_format(
        "audit_final_output",
        {
            "type": "object",
            "properties": {
                "predict_label": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {"type": "string", "enum": labels},
                },
                "audit_trace": {
                    "type": "string",
                    "description": "说明命中的规则、帖子证据、工具事实或豁免原因。",
                },
            },
            "required": ["predict_label", "audit_trace"],
            "additionalProperties": False,
        },
    )


def build_final_candidate_labels(
    original_candidates: Iterable[str],
    shortlist: Iterable[str],
    possible_labels: Iterable[Any],
    tool_required_labels: Iterable[Any],
) -> List[str]:
    original = _dedupe(original_candidates)
    original_set = set(original)

    def label_of(entry: Any) -> str:
        if isinstance(entry, dict):
            return str(entry.get("label", "")).strip()
        return str(getattr(entry, "label", "")).strip()

    kept = _dedupe(
        [label_of(entry) for entry in tool_required_labels]
        + [label_of(entry) for entry in possible_labels]
    )
    kept = [label for label in kept if label in original_set]
    if not kept:
        kept = [label for label in _dedupe(shortlist) if label in original_set]
    if "通过" in original_set and "通过" not in kept:
        kept.append("通过")
    return kept or original
