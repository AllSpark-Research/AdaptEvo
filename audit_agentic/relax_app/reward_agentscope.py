"""Custom group advantage for the single-agent AgentScope Relax rollout."""

from __future__ import annotations

from statistics import mean, stdev
from typing import Any, Dict, List


def _safe_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    if len(values) == 1:
        return [0.0]
    avg = mean(values)
    std = stdev(values)
    if std == 0:
        return [0.0 for _ in values]
    return [(value - avg) / std for value in values]


def advantage_func(
    groups: List[Dict[Any, Dict[str, Any]]],
) -> List[Dict[Any, float]]:
    """Normalize ``score/overall`` per exported unit across rollout slots.

    Implicit AgentScope exports use ``None`` as the unit name. The generic
    implementation also supports named explicit units for future extensions.
    """
    if not groups:
        return []

    output: List[Dict[Any, float]] = [{} for _ in groups]
    unit_names = {
        unit_name
        for group in groups
        for unit_name in group
    }
    for unit_name in unit_names:
        entries: List[tuple[int, float]] = []
        for slot_idx, group in enumerate(groups):
            metadata = group.get(unit_name) or {}
            score = metadata.get("score/overall")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                entries.append((slot_idx, float(score)))
        normalized = _safe_norm([score for _, score in entries])
        for (slot_idx, _), advantage in zip(entries, normalized, strict=True):
            output[slot_idx][unit_name] = advantage
    return output
