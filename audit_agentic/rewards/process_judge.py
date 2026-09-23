"""Validation and deterministic arithmetic for Process Reward Judge scores."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any


DIMENSION_WEIGHTS = {
    "factual_grounding": 0.25,
    "rule_fidelity": 0.25,
    "evidence_coverage": 0.20,
    "tool_use": 0.15,
    "decision_consistency": 0.15,
}

PROCESS_JUDGE_DIMENSIONS = tuple(DIMENSION_WEIGHTS)
PROCESS_JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "process_reward_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                name: {"type": "number", "minimum": 0, "maximum": 1}
                for name in PROCESS_JUDGE_DIMENSIONS
            },
            "required": list(PROCESS_JUDGE_DIMENSIONS),
            "additionalProperties": False,
        },
    },
}


class ProcessJudgeOutputError(ValueError):
    """Raised when an LLM Process Judge response violates the contract."""


def _parse_payload(raw: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        raise ProcessJudgeOutputError("Process Judge output must be JSON or a mapping")
    try:
        payload = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise ProcessJudgeOutputError("Process Judge output is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ProcessJudgeOutputError("Process Judge output must be a JSON object")
    return payload


def _score(payload: Mapping[str, Any], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool):
        raise ProcessJudgeOutputError(f"{name} must be a number in [0, 1]")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ProcessJudgeOutputError(f"{name} must be a number in [0, 1]") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ProcessJudgeOutputError(f"{name} must be a finite number in [0, 1]")
    return score


def normalize_process_judge_output(
    raw: str | Mapping[str, Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate the five Judge scores and compute Process Reward locally."""
    payload = _parse_payload(raw)
    if strict:
        expected = set(PROCESS_JUDGE_DIMENSIONS)
        actual = set(payload)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ProcessJudgeOutputError(
                "Process Judge output keys do not match the strict schema: "
                f"missing={missing}, extra={extra}",
            )
    normalized: dict[str, Any] = {
        name: _score(payload, name)
        for name in DIMENSION_WEIGHTS
    }
    process_reward = sum(
        normalized[name] * weight
        for name, weight in DIMENSION_WEIGHTS.items()
    )
    normalized["process_reward"] = float(process_reward)
    return normalized
