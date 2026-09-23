# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch

from relax.engine.filters.base_types import DynamicFilterOutput
from relax.utils.types import Sample


__all__ = ["check_reward_nonzero_std"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    reward_tensor = torch.tensor(rewards, dtype=torch.float64)
    reward_range = reward_tensor.max() - reward_tensor.min()
    keep = bool(reward_range > 1e-8)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
