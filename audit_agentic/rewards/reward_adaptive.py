"""GT-confidence-aware reward for content-audit Agentic RL.

The process judge is intentionally left as an external hook. Until a process
score is supplied, its contribution is zero and all intermediate weights are
returned for offline analysis and future advantage weighting.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


PASS_LABEL = "通过"
RULE_SUPPORT_GATES = {
    "supported": 1.0,
    "ambiguous": 0.6,
    "unsupported": 0.2,
}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def canonical_label(label: Any) -> str:
    """Normalize plain labels and legacy ``id|label`` values."""
    text = str(label or "").strip()
    if "|" in text:
        prefix, name = text.split("|", 1)
        if prefix.strip().isdigit():
            text = name.strip()
    return text


def _label_set(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        if "labels" in value:
            return _label_set(value.get("labels"))
        if "label" in value:
            return _label_set(value.get("label"))
        return ()
    if isinstance(value, str):
        normalized = canonical_label(value)
        return (normalized,) if normalized else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        labels = {
            canonical_label(item)
            for item in value
            if canonical_label(item)
        }
        return tuple(sorted(labels))
    normalized = canonical_label(value)
    return (normalized,) if normalized else ()


def extract_four_round_labels(metadata: Mapping[str, Any]) -> list[Any]:
    """Extract the four human votes from cleaned dataset metadata."""
    block = metadata.get("four_round_labels") or {}
    rounds = block.get("rounds") if isinstance(block, Mapping) else None
    if not isinstance(rounds, Sequence):
        return []
    labels: list[Any] = []
    for round_info in rounds:
        if isinstance(round_info, Mapping):
            labels.append(round_info.get("labels") or round_info.get("raw"))
        else:
            labels.append(round_info)
    return labels


def human_agreement_confidence(
    labels: Sequence[Any],
    *,
    expected_rounds: int = 4,
) -> float:
    """Compute entropy-based confidence for the four human-review rounds.

    Missing rounds are treated as distinct unknown votes. This conservatively
    lowers confidence instead of interpreting incomplete annotation as full
    agreement.
    """
    if expected_rounds <= 1:
        raise ValueError("expected_rounds must be greater than 1")

    votes: list[tuple[str, ...]] = []
    for index, raw_vote in enumerate(list(labels or [])[:expected_rounds]):
        vote = _label_set(raw_vote)
        votes.append(vote or (f"__missing_round_{index}",))
    for index in range(len(votes), expected_rounds):
        votes.append((f"__missing_round_{index}",))

    counts = Counter(votes)
    entropy = -sum(
        (count / expected_rounds) * math.log(count / expected_rounds)
        for count in counts.values()
        if count > 0
    )
    confidence = 1.0 - entropy / math.log(expected_rounds)
    return _clip01(confidence)


def rule_support_gate(
    verdict: str | Mapping[str, Any] | None,
    *,
    missing_gate: float | None = None,
) -> float:
    """Map Rule Judge verdict to its multiplicative confidence gate."""
    if isinstance(verdict, Mapping):
        verdict = verdict.get("verdict")
    normalized = str(verdict or "").strip().lower()
    if normalized in RULE_SUPPORT_GATES:
        return RULE_SUPPORT_GATES[normalized]
    if missing_gate is None:
        missing_gate = _env_float("AUDIT_GT_CONFIDENCE_MISSING_RULE_GATE", 0.6)
    return _clip01(missing_gate)


def outcome_reward(gt: Any, prediction: Any) -> float:
    """Return 1.0 exact, 0.6 wrong violation subtype, or 0 binary error."""
    gt_labels = set(_label_set(gt))
    pred_labels = set(_label_set(prediction))
    if not gt_labels or not pred_labels:
        return 0.0

    gt_has_pass = PASS_LABEL in gt_labels
    pred_has_pass = PASS_LABEL in pred_labels
    gt_violations = gt_labels - {PASS_LABEL}
    pred_violations = pred_labels - {PASS_LABEL}

    # A prediction containing both pass and violation labels is contradictory.
    if pred_has_pass and pred_violations:
        return 0.0
    if gt_labels == pred_labels:
        return 1.0
    if gt_violations and pred_violations:
        return 0.6
    if gt_has_pass and pred_has_pass and not gt_violations and not pred_violations:
        return 1.0
    return 0.0


def process_reward(judge_output: Any) -> float | None:
    """Validate Process Judge output and return its local weighted score."""
    if judge_output is None:
        return None
    from audit_agentic.rewards.process_judge import normalize_process_judge_output

    return float(
        normalize_process_judge_output(
            judge_output,
            strict=True,
        )["process_reward"]
    )


def compute_final_reward(
    *,
    human_labels: Sequence[Any],
    rule_verdict: str | Mapping[str, Any] | None,
    gt: Any,
    prediction: Any,
    process_reward_value: float | None = None,
    rho: float | None = None,
    gamma: float | None = None,
    gt_weight_min: float | None = None,
    gt_weight_max: float | None = None,
    invalid_final_output: bool = False,
    unknown_tool_call_count: int = 0,
    tool_name_format_error_count: int = 0,
    invalid_tool_arguments_count: int = 0,
    additional_format_error_count: int = 0,
    format_penalty_per_error: float | None = None,
    format_penalty_cap: float | None = None,
    invalid_final_reward: float | None = None,
    reward_floor: float | None = None,
) -> dict[str, Any]:
    """Mix confidence-weighted outcome/process reward and format penalties.

    ``process_reward_value=None`` means the process judge is unavailable. The
    process component is then left at zero rather than fabricated. The result
    exposes ``advantage_weight`` because pure per-prompt z-score normalization
    would otherwise cancel a constant GT-confidence scale.
    """
    rho = _env_float("AUDIT_GT_CONFIDENCE_RHO", 0.1) if rho is None else float(rho)
    gamma = _env_float("AUDIT_GT_CONFIDENCE_GAMMA", 1.0) if gamma is None else float(gamma)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if gamma <= 0.0:
        raise ValueError("gamma must be greater than 0")

    human_confidence = human_agreement_confidence(human_labels)
    gate = rule_support_gate(rule_verdict)
    gt_confidence = _clip01(human_confidence * gate)
    out_reward = outcome_reward(gt, prediction)

    configured_weight_min = os.environ.get("AUDIT_GT_WEIGHT_MIN")
    configured_weight_max = os.environ.get("AUDIT_GT_WEIGHT_MAX")
    bounded_weight_enabled = any(
        value is not None
        for value in (
            gt_weight_min,
            gt_weight_max,
            configured_weight_min,
            configured_weight_max,
        )
    )
    if bounded_weight_enabled:
        if gt_weight_min is None:
            gt_weight_min = _env_float("AUDIT_GT_WEIGHT_MIN", 0.0)
        if gt_weight_max is None:
            gt_weight_max = _env_float("AUDIT_GT_WEIGHT_MAX", 1.0 - rho)
        gt_weight_min = float(gt_weight_min)
        gt_weight_max = float(gt_weight_max)
        if not 0.0 <= gt_weight_min <= gt_weight_max <= 1.0:
            raise ValueError(
                "GT weight bounds must satisfy "
                "0 <= gt_weight_min <= gt_weight_max <= 1"
            )
        outcome_weight = gt_weight_min + (
            (gt_weight_max - gt_weight_min) * (gt_confidence ** gamma)
        )
    else:
        gt_weight_min = 0.0
        gt_weight_max = 1.0 - rho
        outcome_weight = gt_weight_max * (gt_confidence ** gamma)
    process_weight = 1.0 - outcome_weight
    process_available = process_reward_value is not None
    process_value = _clip01(process_reward_value) if process_available else 0.0
    outcome_component = outcome_weight * out_reward
    process_component = process_weight * process_value if process_available else 0.0
    mixed_reward = outcome_component + process_component

    tool_format_error_count = sum(
        max(0, int(value))
        for value in (
            unknown_tool_call_count,
            tool_name_format_error_count,
            invalid_tool_arguments_count,
            additional_format_error_count,
        )
    )
    if format_penalty_per_error is None:
        format_penalty_per_error = _env_float(
            "AUDIT_AGENTSCOPE_TOOL_PROTOCOL_PENALTY",
            0.1,
        )
    if format_penalty_cap is None:
        format_penalty_cap = _env_float(
            "AUDIT_AGENTSCOPE_TOOL_PROTOCOL_PENALTY_CAP",
            0.3,
        )
    format_penalty = min(
        max(0.0, float(format_penalty_cap)),
        tool_format_error_count * max(0.0, float(format_penalty_per_error)),
    )

    if invalid_final_reward is None:
        invalid_final_reward = _env_float("AUDIT_AGENTSCOPE_INVALID_FINAL_REWARD", 0.0)
    if reward_floor is None:
        reward_floor = _env_float("AUDIT_AGENTSCOPE_REWARD_FLOOR", -1.0)

    if invalid_final_output:
        final_reward = max(float(reward_floor), float(invalid_final_reward))
    else:
        final_reward = max(float(reward_floor), mixed_reward - format_penalty)

    verdict_value = (
        rule_verdict.get("verdict")
        if isinstance(rule_verdict, Mapping)
        else rule_verdict
    )
    return {
        "human_confidence": float(human_confidence),
        "rule_verdict": str(verdict_value or "missing").strip().lower(),
        "rule_gate": float(gate),
        "gt_confidence": float(gt_confidence),
        "outcome_reward": float(out_reward),
        "process_reward": float(process_value),
        "process_reward_available": float(process_available),
        "outcome_weight": float(outcome_weight),
        "process_weight": float(process_weight),
        "bounded_weight_enabled": float(bounded_weight_enabled),
        "gt_weight_min": float(gt_weight_min),
        "gt_weight_max": float(gt_weight_max),
        "effective_process_weight": float(process_weight if process_available else 0.0),
        "outcome_component": float(outcome_component),
        "process_component": float(process_component),
        "mixed_reward_before_format": float(mixed_reward),
        "format_error_count": float(tool_format_error_count),
        "format_penalty": float(format_penalty),
        "invalid_final_output": float(invalid_final_output),
        "advantage_weight": float(1.0 if process_available else outcome_weight),
        "rho": float(rho),
        "gamma": float(gamma),
        "final_reward": float(final_reward),
    }
