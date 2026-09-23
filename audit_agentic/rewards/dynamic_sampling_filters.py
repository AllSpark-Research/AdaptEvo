"""Audit-specific dynamic sampling filters for Relax rollouts."""

from __future__ import annotations

from relax.engine.filters.base_types import DynamicFilterOutput
from relax.engine.filters.dynamic_sampling_filters import check_reward_nonzero_std
from relax.utils.types import Sample


__all__ = ["check_process_judge_available", "check_reward_nonzero_std_and_process_judge"]


def _metadata_number(sample: Sample, key: str) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    try:
        return float(metadata.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def check_process_judge_available(
    args,
    samples: list[Sample],
    **kwargs,
) -> DynamicFilterOutput:
    """Drop unavailable required Judge scores without filtering reward spread.

    Invalid Agent final outputs intentionally skip the Process Judge and remain
    trainable so their protocol/format penalty is preserved. For otherwise
    valid outputs, a missing Judge score means the external Judge failed even
    after its configured retry budget, so the whole GRPO group is discarded.
    """
    missing_required_judge = any(
        _metadata_number(sample, "process_judge/enabled") > 0.5
        and _metadata_number(sample, "process_judge/success") < 0.5
        and _metadata_number(sample, "agentscope/invalid_final_output") < 0.5
        for sample in samples
    )
    if missing_required_judge:
        return DynamicFilterOutput(keep=False, reason="process_judge_failed")

    return DynamicFilterOutput(keep=True)


def check_reward_nonzero_std_and_process_judge(
    args,
    samples: list[Sample],
    **kwargs,
) -> DynamicFilterOutput:
    """Preserve the existing combined availability and reward-spread filter."""
    available = check_process_judge_available(args, samples, **kwargs)
    if not available.keep:
        return available
    return check_reward_nonzero_std(args, samples, **kwargs)
