"""Custom advantage hook for audit_agentic.

Two agents (main + planner). Imported by Relax via:

    --agentic-custom-advantage-path audit_agentic.relax_app.reward_audit.advantage_func

Reads ``metadata["score/overall"]`` per slot when present, falling back to
``metadata["score/<role>"]`` for compatibility. The upstream raw reward scores
are kept in [0, 1], but this hook normalises them within each role by sample
standard deviation, then scales by ROLE_COEFFICIENTS. Therefore the returned
advantages are centred z-scores, not [0, 1] rewards.

V5: all slots participate in normalisation. Pass-case samples are not masked
out of the planner group — V5's ``reward_planner_v5`` gives them a meaningful
format/support score, so they carry training signal. ``is_pass_case/planner``
is no longer emitted; the diagnostic field is ``planner/is_gt_pass`` and it
does not affect masking.
"""

from __future__ import annotations

from statistics import mean, stdev
from typing import Any, Dict, List


# main agent does both router and final turns in one history. The agent app can
# emit one shared score/overall for all roles so dynamic filtering sees the
# same case-level reward for main and planner; role coefficients still weight
# the training signal toward the final main decision.
ROLE_COEFFICIENTS: Dict[str, float] = {
    "main": 0.7,
    "planner": 0.3,
}


def _safe_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    if len(values) == 1:
        return [0.0]
    m = mean(values)
    s = stdev(values)
    if s == 0:
        return [0.0 for _ in values]
    return [(v - m) / s for v in values]


def advantage_func(groups: List[Dict[str, Dict[str, Any]]]) -> List[Dict[str, float]]:
    """groups: list of {role_name: metadata_dict} per rollout slot.

    Returns same-length list of {role_name: scalar advantage}. Roles not in a
    slot get omitted (callers should treat missing as 0 / no-op). Case-level
    zero-std filtering is handled centrally by Relax's dynamic sampling filter.
    """
    if not groups:
        return []

    out: List[Dict[str, float]] = [{} for _ in groups]
    roles: set = set()
    for g in groups:
        roles.update(g.keys())

    for role in roles:
        coef = ROLE_COEFFICIENTS.get(role, 1.0)

        # Collect (slot_idx, score) for this role.
        entries: List[tuple] = []
        for i, g in enumerate(groups):
            md = g.get(role) or {}
            score = md.get("score/overall", md.get(f"score/{role}"))
            if not isinstance(score, (int, float)):
                continue
            entries.append((i, float(score)))

        if not entries:
            continue

        scores = [s for (_i, s) in entries]
        normed = _safe_norm(scores)
        for (slot_idx, _score), adv in zip(entries, normed):
            out[slot_idx][role] = coef * adv

    return out
