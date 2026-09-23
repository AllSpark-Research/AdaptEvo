"""Reward dispatcher for router_mix RL training.

Routes by metadata["reward_type"]:
  - "vl_qa_llmjudge"        → LLM judge for VL QA (multi-choice)
  - "router_label_select"   → Recall reward using audit_agentic.eval.parsing
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from relax.engine.rewards.router_label_select_reward import router_label_select_reward
from relax.engine.rewards.vl_qa_llmjudge import vl_qa_llmjudge_reward


RewardFunc = Callable[..., Awaitable[Any]]

REWARD_FUNCS: dict[str, RewardFunc] = {
    "vl_qa_llmjudge": vl_qa_llmjudge_reward,
    "router_label_select": router_label_select_reward,
}

ALIASES: dict[str, str] = {
    "vl_qa_llmjudge": "vl_qa_llmjudge",
    "vl_qa": "vl_qa_llmjudge",
    "vl": "vl_qa_llmjudge",
    "router_label_select": "router_label_select",
    "router": "router_label_select",
}


def _metadata(sample) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _infer_reward_type(sample) -> str:
    metadata = _metadata(sample)
    value = metadata.get("reward_type", "")
    if isinstance(value, str) and value.strip():
        key = value.strip().lower()
        if key in ALIASES:
            return ALIASES[key]
    raise ValueError(
        f"Cannot infer reward_type. Set metadata.reward_type to one of {sorted(REWARD_FUNCS)}. "
        f"Got: {value!r}"
    )


async def router_mixrl_reward(args, samples, **kwargs):
    if not isinstance(samples, list):
        samples = [samples]

    groups: dict[str, list] = {}
    indices: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        rt = _infer_reward_type(s)
        groups.setdefault(rt, []).append(s)
        indices.setdefault(rt, []).append(i)

    results = [None] * len(samples)
    for rt, group in groups.items():
        func = REWARD_FUNCS[rt]
        group_results = await func(args, group, **kwargs)
        if not isinstance(group_results, list):
            group_results = [group_results]
        for idx, res in zip(indices[rt], group_results):
            results[idx] = res

    return results
