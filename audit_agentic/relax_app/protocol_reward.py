"""Reward helpers for AgentScope model-protocol errors in Relax."""

from __future__ import annotations

import os
from typing import Any

from audit_agentic.rewards import reward_final


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def score_agentscope_protocol(
    *,
    prediction: list[str],
    gt_labels: list[str],
    invalid_final_output: bool,
    unknown_tool_call_count: int = 0,
    tool_name_format_error_count: int = 0,
    invalid_tool_arguments_count: int = 0,
    final_protocol_error_count: int = 0,
) -> dict[str, float]:
    """Score classification quality and model-originated protocol mistakes."""
    if invalid_final_output:
        classification: dict[str, Any] = {
            "score": 0.0,
            "nOA": 0.0,
            "wOA": 0.0,
            "overlap": 0.0,
            "label_f1": 0.0,
        }
    else:
        classification = reward_final(
            predict_label=list(prediction or []),
            gt_labels=list(gt_labels or []),
            mode="hybrid",
        )

    tool_protocol_error_count = max(0, int(unknown_tool_call_count)) + max(
        0,
        int(tool_name_format_error_count),
    ) + max(0, int(invalid_tool_arguments_count))
    per_error_penalty = max(
        0.0,
        _env_float("AUDIT_AGENTSCOPE_TOOL_PROTOCOL_PENALTY", 0.1),
    )
    penalty_cap = max(
        0.0,
        _env_float("AUDIT_AGENTSCOPE_TOOL_PROTOCOL_PENALTY_CAP", 0.3),
    )
    tool_protocol_penalty = min(
        tool_protocol_error_count * per_error_penalty,
        penalty_cap,
    )
    reward_floor = _env_float("AUDIT_AGENTSCOPE_REWARD_FLOOR", -1.0)

    if invalid_final_output:
        score = max(
            reward_floor,
            _env_float("AUDIT_AGENTSCOPE_INVALID_FINAL_REWARD", 0.0),
        )
    else:
        score = max(
            reward_floor,
            float(classification["score"]) - tool_protocol_penalty,
        )
    protocol_penalty = float(classification["score"]) - score

    return {
        "score": float(score),
        "classification_score": float(classification["score"]),
        "protocol_penalty": float(protocol_penalty),
        "tool_protocol_penalty": float(tool_protocol_penalty),
        "tool_protocol_error_count": float(tool_protocol_error_count),
        "protocol_error_count": float(
            tool_protocol_error_count + max(0, int(final_protocol_error_count)),
        ),
        "nOA": float(classification["nOA"]),
        "wOA": float(classification["wOA"]),
        "overlap": float(classification["overlap"]),
        "label_f1": float(classification["label_f1"]),
    }
