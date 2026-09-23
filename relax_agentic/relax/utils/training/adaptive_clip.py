"""GT-confidence-conditioned policy clipping helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def extract_gt_confidence(metadata: Any, fallback: float) -> float:
    """Read a static GT confidence from rollout metadata and clamp it to [0, 1]."""
    value: Any = None
    if isinstance(metadata, Mapping):
        value = metadata.get("reward/gt_confidence")
        if value is None:
            nested = metadata.get("gt_confidence")
            if isinstance(nested, Mapping):
                value = nested.get("value")
            elif nested is not None:
                value = nested

    try:
        confidence = float(value if value is not None else fallback)
    except (TypeError, ValueError):
        confidence = float(fallback)
    if not math.isfinite(confidence):
        confidence = float(fallback)
    return max(0.0, min(1.0, confidence))


def compute_gt_confidence_clip_margins(
    confidence: torch.Tensor,
    *,
    eps_clip: float,
    eps_clip_high: float,
    scale_min: float,
    scale_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-token lower/upper clipping margins and their shared scale."""
    confidence = confidence.clamp(0.0, 1.0)
    scale = scale_min + (scale_max - scale_min) * confidence
    return eps_clip * scale, eps_clip_high * scale, scale


def expand_sample_scalars_to_token_parts(
    sample_values: torch.Tensor | Sequence[float],
    token_parts: torch.Tensor | Sequence[torch.Tensor],
    *,
    fallback_token_counts: Sequence[int] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Expand one scalar per sample to match packed per-sample token tensors."""
    values = torch.as_tensor(sample_values, device=device, dtype=dtype).reshape(-1)
    sample_count = int(values.numel())

    parts: list[torch.Tensor]
    if isinstance(token_parts, (list, tuple)):
        parts = list(token_parts)
    elif isinstance(token_parts, torch.Tensor) and token_parts.is_nested:
        parts = [token_parts[index] for index in range(token_parts.size(0))]
    elif isinstance(token_parts, torch.Tensor) and token_parts.ndim >= 2 and token_parts.size(0) == sample_count:
        parts = [token_parts[index] for index in range(sample_count)]
    elif isinstance(token_parts, torch.Tensor) and sample_count == 1:
        parts = [token_parts]
    elif (
        isinstance(token_parts, torch.Tensor)
        and fallback_token_counts is not None
        and len(fallback_token_counts) == sample_count
        and sum(int(count) for count in fallback_token_counts) == token_parts.numel()
    ):
        parts = list(token_parts.reshape(-1).split([int(count) for count in fallback_token_counts]))
    else:
        raise ValueError(
            "Cannot align GT confidence with token parts: "
            f"sample_count={sample_count}, token_parts_type={type(token_parts).__name__}"
        )

    if len(parts) != sample_count:
        raise ValueError(
            "GT confidence/sample count mismatch: "
            f"confidences={sample_count}, token_parts={len(parts)}"
        )

    expanded = [
        values[index].expand(int(part.numel()))
        for index, part in enumerate(parts)
    ]
    if not expanded:
        return torch.empty(0, device=device, dtype=dtype)
    return torch.cat(expanded, dim=0)
