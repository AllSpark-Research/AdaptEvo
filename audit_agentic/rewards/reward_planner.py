"""Planner reward.

Two versions live side-by-side:

  * ``reward_planner`` (V2): pass cases short-circuit to 1.0; ``is_pass_case``
    is exposed so ``advantage_func`` can mask those slots out of the group
    normalisation. Kept for backwards-compat with old training scripts.

  * ``reward_planner_v3``: removes the pass-case shortcut. Pass cases also
    participate in training — the planner is graded on whether it ruled out
    the violation candidates that don't apply. Violation cases keep
    ``gt_survival`` as the dominant signal but add explicit "通过" handling.

V3 derives ``rule_out`` implicitly:

    rule_out = shortlist - (possible_labels ∪ tool_required_labels)

The current planner JSON schema only emits ``possible_labels`` and
``tool_required_labels`` (anything else from the shortlist is implicitly
ruled out), so we don't need an extra bucket on the agent side.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


PASS_LABEL = "通过"


def _normalize_label(label: Any) -> str:
    if not isinstance(label, str):
        return ""
    label = label.strip()
    if "|" in label:
        return label.split("|", 1)[1].strip()
    return label


def _to_set(xs: Optional[List[str]]) -> Set[str]:
    return {x for x in (_normalize_label(v) for v in (xs or [])) if x}


def reward_planner(
    possible_labels: List[str],
    tool_required_labels: List[str],
    shortlist: List[str],
    gt_labels: Optional[List[str]],
    tools_called: int,
    tools_total_available: int,
    target_kept: int = 4,
) -> Dict[str, float]:
    gt = _to_set(gt_labels)

    # Pass case (or empty GT): planner not responsible — full credit.
    if not gt or gt == {PASS_LABEL}:
        return {
            "score": 1.0,
            "gt_survival": 1.0,
            "rule_out_accuracy": 1.0,
            "kept_size": float(len(_to_set(possible_labels) | _to_set(tool_required_labels))),
            "is_pass_case": 1.0,
        }

    sl = _to_set(shortlist)
    kept = _to_set(possible_labels) | _to_set(tool_required_labels)
    ruled_out = sl - kept

    # Severe penalty: planner kept nothing
    if len(kept) == 0:
        return {
            "score": -1.0,
            "gt_survival": 0.0,
            "rule_out_accuracy": 0.0,
            "kept_size": 0.0,
            "is_pass_case": 0.0,
        }

    # Core: GT survival
    gt_survival = len(gt & kept) / len(gt)

    # Rule-out accuracy: of wrong labels in shortlist, how many were ruled out
    wrong_labels = sl - gt - {PASS_LABEL}
    if wrong_labels:
        rule_out_accuracy = len(ruled_out & wrong_labels) / len(wrong_labels)
    else:
        rule_out_accuracy = 1.0

    excess = max(0, len(kept) - target_kept)
    excess_penalty = 0.05 * excess

    tool_signal = 0.0
    if tools_total_available > 0:
        tool_signal = 0.1 if tools_called > 0 else -0.05

    score = 0.7 * gt_survival + 0.2 * rule_out_accuracy - excess_penalty + tool_signal
    score = max(-0.5, min(1.0, score))

    return {
        "score": float(score),
        "gt_survival": float(gt_survival),
        "rule_out_accuracy": float(rule_out_accuracy),
        "kept_size": float(len(kept)),
        "tools_called": float(tools_called),
        "tool_signal": float(tool_signal),
        "excess_penalty": float(excess_penalty),
        "is_pass_case": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# V3
# ─────────────────────────────────────────────────────────────────────────────

# Tunable constants. Kept here so future calibration is one place.
V3_PASS_W_PASS_SURVIVAL = 0.30
V3_PASS_W_RULE_OUT_ACC = 0.45
V3_PASS_W_POSSIBLE_CLEAN = 0.25
V3_PASS_POSSIBLE_VIOL_PENALTY = 0.08    # per possible-bucket violation beyond the first
V3_PASS_TR_VIOL_SOFT_PENALTY = 0.03     # per tool_required-bucket violation beyond 3
V3_PASS_TOOL_COST = 0.03                # per tool call beyond 2

V3_VIOL_W_GT_SURVIVAL = 0.60
V3_VIOL_W_RULE_OUT_WRONG = 0.20
V3_VIOL_W_PASS_RULE_OUT = 0.10
V3_VIOL_W_COMPACT_KEPT = 0.10
V3_VIOL_GT_RULE_OUT_PENALTY = 0.5       # hard penalty if GT was ruled out
V3_VIOL_PASS_KEPT_PENALTY = 0.2         # penalty if "通过" was kept in a violation case
V3_VIOL_EXCESS_KEPT_WEIGHT = 0.05       # per kept label beyond target_kept
V3_VIOL_TOOL_COST = 0.03                # per tool call beyond 2
V3_TARGET_KEPT = 4                      # ideal upper bound for |kept|
V3_KEPT_SOFT_BUFFER = 6                 # compact_kept = 1 - clamp((kept - gt) / buffer)


def reward_planner_v3(
    possible_labels: List[str],
    tool_required_labels: List[str],
    shortlist: List[str],
    gt_labels: Optional[List[str]],
    tools_called: int,
    tools_total_available: int,
) -> Dict[str, float]:
    """Planner reward, V3. See module docstring for design notes."""
    sl = _to_set(shortlist)
    # Defensive: planner_agent.run_plan() already drops out-of-shortlist labels
    # and disallows overlap, but the relax_app path uses a lighter parse so we
    # re-clip here.
    possible = _to_set(possible_labels) & sl
    tool_required = _to_set(tool_required_labels) & sl
    kept = possible | tool_required
    rule_out = sl - kept

    gt = _to_set(gt_labels)
    is_gt_pass = bool(gt) and gt == {PASS_LABEL}
    violation_shortlist = sl - {PASS_LABEL}

    tool_cost_penalty = V3_PASS_TOOL_COST * max(0, tools_called - 2)

    base_metrics = {
        "is_gt_pass": float(is_gt_pass),
        "possible_cnt": float(len(possible)),
        "tool_required_cnt": float(len(tool_required)),
        "rule_out_cnt": float(len(rule_out)),
        "kept_cnt": float(len(kept)),
        "shortlist_cnt": float(len(sl)),
        "tools_called": float(tools_called),
        "tool_cost_penalty": float(tool_cost_penalty),
    }

    if is_gt_pass:
        # GT=通过: planner should rule out the violation candidates.
        if PASS_LABEL in sl:
            pass_survival = 1.0 if PASS_LABEL in kept else 0.0
        else:
            pass_survival = 1.0  # router never gave us 通过, planner is off the hook

        if violation_shortlist:
            rule_out_accuracy = len(rule_out & violation_shortlist) / len(violation_shortlist)
        else:
            rule_out_accuracy = 1.0

        possible_violation_cnt = len(possible & violation_shortlist)
        # possible_clean: fraction of violation candidates that did NOT slip into possible
        possible_clean = 1.0 - min(
            1.0, possible_violation_cnt / max(1, len(violation_shortlist))
        )
        possible_violation_penalty = V3_PASS_POSSIBLE_VIOL_PENALTY * max(
            0, possible_violation_cnt - 1
        )
        tool_required_violation_cnt = len(tool_required & violation_shortlist)
        tool_required_soft_penalty = V3_PASS_TR_VIOL_SOFT_PENALTY * max(
            0, tool_required_violation_cnt - 3
        )

        score = (
            V3_PASS_W_PASS_SURVIVAL * pass_survival
            + V3_PASS_W_RULE_OUT_ACC * rule_out_accuracy
            + V3_PASS_W_POSSIBLE_CLEAN * possible_clean
            - possible_violation_penalty
            - tool_required_soft_penalty
            - tool_cost_penalty
        )
        score = max(0.0, min(1.0, score))

        return {
            **base_metrics,
            "score": float(score),
            "pass_survival": float(pass_survival),
            "rule_out_accuracy": float(rule_out_accuracy),
            "possible_clean": float(possible_clean),
            "possible_violation_cnt": float(possible_violation_cnt),
            "tool_required_violation_cnt": float(tool_required_violation_cnt),
            "possible_violation_penalty": float(possible_violation_penalty),
            "tool_required_soft_penalty": float(tool_required_soft_penalty),
            # violation-only fields zeroed for log uniformity
            "gt_survival": 0.0,
            "gt_rule_out_penalty": 0.0,
            "rule_out_wrong_acc": 0.0,
            "pass_rule_out_bonus": 0.0,
            "pass_kept_penalty": 0.0,
            "excess_kept_penalty": 0.0,
            "compact_kept": 0.0,
        }

    # Violation case (or empty GT): keep GT, rule out wrong, rule out 通过.
    if gt:
        gt_survival = len(gt & kept) / len(gt)
    else:
        gt_survival = 0.0  # no GT to credit
    gt_rule_out_penalty = V3_VIOL_GT_RULE_OUT_PENALTY if (gt & rule_out) else 0.0

    wrong_violation = violation_shortlist - gt
    if wrong_violation:
        rule_out_wrong_acc = len(rule_out & wrong_violation) / len(wrong_violation)
    else:
        rule_out_wrong_acc = 1.0

    if PASS_LABEL in sl:
        pass_rule_out_bonus = 1.0 if PASS_LABEL in rule_out else 0.0
        pass_kept_penalty = V3_VIOL_PASS_KEPT_PENALTY if PASS_LABEL in kept else 0.0
    else:
        # 通过 not in shortlist → planner had nothing to do here; give the
        # bonus by default so we don't penalise unrelated factors.
        pass_rule_out_bonus = 1.0
        pass_kept_penalty = 0.0

    excess_kept_penalty = V3_VIOL_EXCESS_KEPT_WEIGHT * max(0, len(kept) - V3_TARGET_KEPT)
    tool_cost_penalty_v = V3_VIOL_TOOL_COST * max(0, tools_called - 2)

    compact_kept = 1.0 - min(
        1.0,
        max(0, len(kept) - max(1, len(gt))) / V3_KEPT_SOFT_BUFFER,
    )

    score = (
        V3_VIOL_W_GT_SURVIVAL * gt_survival
        + V3_VIOL_W_RULE_OUT_WRONG * rule_out_wrong_acc
        + V3_VIOL_W_PASS_RULE_OUT * pass_rule_out_bonus
        + V3_VIOL_W_COMPACT_KEPT * compact_kept
        - gt_rule_out_penalty
        - pass_kept_penalty
        - excess_kept_penalty
        - tool_cost_penalty_v
    )
    score = max(-0.5, min(1.0, score))

    return {
        **base_metrics,
        "tool_cost_penalty": float(tool_cost_penalty_v),  # override for violation path
        "score": float(score),
        "gt_survival": float(gt_survival),
        "gt_rule_out_penalty": float(gt_rule_out_penalty),
        "rule_out_wrong_acc": float(rule_out_wrong_acc),
        "pass_rule_out_bonus": float(pass_rule_out_bonus),
        "pass_kept_penalty": float(pass_kept_penalty),
        "excess_kept_penalty": float(excess_kept_penalty),
        "compact_kept": float(compact_kept),
        # pass-only fields zeroed for log uniformity
        "pass_survival": 0.0,
        "rule_out_accuracy": 0.0,
        "possible_clean": 0.0,
        "possible_violation_cnt": 0.0,
        "tool_required_violation_cnt": 0.0,
        "possible_violation_penalty": 0.0,
        "tool_required_soft_penalty": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# V4
# ─────────────────────────────────────────────────────────────────────────────
# Key changes from V3:
#
# Pass case (GT=通过):
#   * No pass_survival — the "通过" judgment is final's responsibility.
#   * No rule_out_accuracy / possible_clean — these signals conflict with
#     violation-case gt_survival and cause the model to learn "don't flag
#     violations" when pass cases dominate training data.
#   * Score ≈ format_score only, with a mild penalty when the planner
#     outputs a violation label in possible_labels with preliminary_opinion
#     = "support" (actively misleading main agent).
#   * Effect: in GRPO, pass-case rollouts all score similarly → advantage
#     ≈ 0 → planner barely trains on pass data → no conflicting signal.
#
# Violation case (GT≠通过):
#   * gt_rule_out_penalty is proportional (0.3 * ratio), not a cliff (0.5).
#   * format_score added as a sub-signal.
#   * Fewer sub-terms for cleaner gradient signal.
#
# Both:
#   * format_score: JSON-valid + correct schema + labels in shortlist.
# ─────────────────────────────────────────────────────────────────────────────

# ── V4 format score weights ─────────────────────────────────────────────────
V4_FMT_W_JSON = 0.40           # parseable JSON
V4_FMT_W_SCHEMA = 0.30         # has filtered_labels + tool_calls structure
V4_FMT_W_LABELS = 0.30         # all kept labels come from shortlist

# ── V4 pass case ────────────────────────────────────────────────────────────
V4_PASS_W_FORMAT = 1.0                     # pass case is entirely about format
V4_PASS_SUPPORT_PENALTY = 0.10             # per violation label with "support" opinion

# ── V4 violation case ──────────────────────────────────────────────────────
V4_VIOL_W_GT_SURVIVAL = 0.50
V4_VIOL_W_FORMAT = 0.20
V4_VIOL_W_RULE_OUT_WRONG = 0.15
V4_VIOL_W_COMPACT = 0.10
V4_VIOL_W_PASS_RULE_OUT = 0.05
V4_VIOL_GT_PENALTY_WEIGHT = 0.30           # proportional: 0.3 * (gt_ruled_out / gt_total)
V4_VIOL_PASS_KEPT_PENALTY = 0.10
V4_VIOL_EXCESS_KEPT_WEIGHT = 0.03
V4_VIOL_TARGET_KEPT = 4
V4_VIOL_COMPACT_BUFFER = 6


def _v4_format_score(
    possible_labels_raw: List[Any],
    tool_required_labels_raw: List[Any],
    shortlist_set: Set[str],
    json_valid: bool,
    has_filtered: bool,
    has_tool_calls: bool,
) -> float:
    """Graduated format score in [0, 1]."""
    if not json_valid:
        return 0.0
    score = V4_FMT_W_JSON
    if has_filtered and has_tool_calls:
        score += V4_FMT_W_SCHEMA
    # labels validity: all kept labels must come from shortlist
    all_labels: Set[str] = set()
    for x in possible_labels_raw:
        if isinstance(x, dict) and x.get("label"):
            all_labels.add(_normalize_label(x["label"]))
    for x in tool_required_labels_raw:
        if isinstance(x, dict) and x.get("label"):
            all_labels.add(_normalize_label(x["label"]))
    if not all_labels or all_labels <= shortlist_set:
        score += V4_FMT_W_LABELS
    return score


def reward_planner_v4(
    possible_labels_raw: List[Any],
    tool_required_labels_raw: List[Any],
    shortlist: List[str],
    gt_labels: Optional[List[str]],
    tools_called: int,
    tools_total_available: int,
    *,
    json_valid: bool = True,
    has_filtered: bool = True,
    has_tool_calls: bool = True,
) -> Dict[str, float]:
    """Planner reward, V4. See module docstring + V4 comment block for design.

    Parameters
    ----------
    possible_labels_raw : list of dict
        Raw dicts from planner JSON, each with at least ``label`` and
        optionally ``preliminary_opinion``.
    tool_required_labels_raw : list of dict
        Raw dicts from planner JSON, each with at least ``label``.
    json_valid, has_filtered, has_tool_calls : bool
        Format-quality flags computed by the caller (agent.py) from
        the parsed planner output.
    """
    sl = _to_set(shortlist)

    # Extract label names (clip to shortlist like V3)
    possible: Set[str] = set()
    for x in possible_labels_raw:
        label = _normalize_label(x.get("label")) if isinstance(x, dict) else ""
        if label in sl:
            possible.add(label)
    tool_required: Set[str] = set()
    for x in tool_required_labels_raw:
        label = _normalize_label(x.get("label")) if isinstance(x, dict) else ""
        if label in sl:
            tool_required.add(label)

    kept = possible | tool_required
    rule_out = sl - kept

    gt = _to_set(gt_labels)
    is_gt_pass = bool(gt) and gt == {PASS_LABEL}
    violation_shortlist = sl - {PASS_LABEL}

    format_score = _v4_format_score(
        possible_labels_raw, tool_required_labels_raw, sl,
        json_valid, has_filtered, has_tool_calls,
    )

    base_metrics: Dict[str, float] = {
        "is_gt_pass": float(is_gt_pass),
        "format_score": float(format_score),
        "json_valid": float(json_valid),
        "possible_cnt": float(len(possible)),
        "tool_required_cnt": float(len(tool_required)),
        "rule_out_cnt": float(len(rule_out)),
        "kept_cnt": float(len(kept)),
        "shortlist_cnt": float(len(sl)),
        "tools_called": float(tools_called),
    }

    # ── Pass case ──
    if is_gt_pass:
        # Count violation labels in possible_labels that have "support" opinion
        # → this actively misleads main agent on a clean note.
        support_viol_cnt = sum(
            1 for x in possible_labels_raw
            if isinstance(x, dict)
            and _normalize_label(x.get("label")) in violation_shortlist
            and str(x.get("preliminary_opinion", "")).strip().lower() == "support"
        )
        support_penalty = V4_PASS_SUPPORT_PENALTY * support_viol_cnt

        score = V4_PASS_W_FORMAT * format_score - support_penalty
        score = max(0.0, min(1.0, score))

        return {
            **base_metrics,
            "score": float(score),
            "support_violation_cnt": float(support_viol_cnt),
            "support_penalty": float(support_penalty),
            # violation-only fields zeroed for log uniformity
            "gt_survival": 0.0,
            "gt_ruled_out_ratio": 0.0,
            "gt_rule_out_penalty": 0.0,
            "rule_out_wrong_acc": 0.0,
            "pass_rule_out_bonus": 0.0,
            "pass_kept_penalty": 0.0,
            "excess_kept_penalty": 0.0,
            "compact_kept": 0.0,
        }

    # ── Violation case (or empty GT) ──
    if gt:
        gt_survival = len(gt & kept) / len(gt)
        gt_ruled_out_ratio = len(gt & rule_out) / len(gt)
    else:
        gt_survival = 0.0
        gt_ruled_out_ratio = 0.0
    gt_rule_out_penalty = V4_VIOL_GT_PENALTY_WEIGHT * gt_ruled_out_ratio

    wrong_violation = violation_shortlist - gt
    rule_out_wrong_acc = (
        len(rule_out & wrong_violation) / len(wrong_violation)
        if wrong_violation else 1.0
    )

    if PASS_LABEL in sl:
        pass_rule_out_bonus = 1.0 if PASS_LABEL in rule_out else 0.0
        pass_kept_penalty = V4_VIOL_PASS_KEPT_PENALTY if PASS_LABEL in kept else 0.0
    else:
        pass_rule_out_bonus = 1.0
        pass_kept_penalty = 0.0

    excess_kept_penalty = V4_VIOL_EXCESS_KEPT_WEIGHT * max(0, len(kept) - V4_VIOL_TARGET_KEPT)

    compact_kept = 1.0 - min(
        1.0,
        max(0, len(kept) - max(1, len(gt))) / V4_VIOL_COMPACT_BUFFER,
    )

    score = (
        V4_VIOL_W_GT_SURVIVAL * gt_survival
        + V4_VIOL_W_FORMAT * format_score
        + V4_VIOL_W_RULE_OUT_WRONG * rule_out_wrong_acc
        + V4_VIOL_W_COMPACT * compact_kept
        + V4_VIOL_W_PASS_RULE_OUT * pass_rule_out_bonus
        - gt_rule_out_penalty
        - pass_kept_penalty
        - excess_kept_penalty
    )
    score = max(-0.5, min(1.0, score))

    return {
        **base_metrics,
        "score": float(score),
        "gt_survival": float(gt_survival),
        "gt_ruled_out_ratio": float(gt_ruled_out_ratio),
        "gt_rule_out_penalty": float(gt_rule_out_penalty),
        "rule_out_wrong_acc": float(rule_out_wrong_acc),
        "pass_rule_out_bonus": float(pass_rule_out_bonus),
        "pass_kept_penalty": float(pass_kept_penalty),
        "excess_kept_penalty": float(excess_kept_penalty),
        "compact_kept": float(compact_kept),
        # pass-only fields zeroed for log uniformity
        "support_violation_cnt": 0.0,
        "support_penalty": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# V5
# ─────────────────────────────────────────────────────────────────────────────
# Changes from V4:
#   * Removed rule_out_wrong_acc (was 0.15 weight).
#   * pass_rule_out_bonus weight: 0.05 → 0.2.
#   * Removed gt_rule_out_penalty (and gt_ruled_out_ratio).
#   * Removed excess_kept_penalty.
#   * All scores normalized to [0, 1] via (raw - raw_min) / (raw_max - raw_min).
#
# Violation case positive terms (raw_max = 1.0):
#   gt_survival(0.70) + format_score(0.10) + pass_rule_out_bonus(0.20)
#
# Violation case penalties:
#   pass_kept_penalty(0.30) if "通过" kept in violation case.
#
# Pass case unchanged from V4 (format_score only + support_penalty), but
# now normalized the same way.
# ─────────────────────────────────────────────────────────────────────────────

# ── V5 format score weights (same as V4) ─────────────────────────────────────
V5_FMT_W_JSON = V4_FMT_W_JSON
V5_FMT_W_SCHEMA = V4_FMT_W_SCHEMA
V5_FMT_W_LABELS = V4_FMT_W_LABELS

# ── V5 pass case ─────────────────────────────────────────────────────────────
V5_PASS_W_FORMAT = V4_PASS_W_FORMAT                    # 1.0
V5_PASS_POSSIBLE_VIOLATION_PENALTY = 0.20
V5_PASS_SUPPORT_VIOLATION_PENALTY = 0.40

# ── V5 violation case ────────────────────────────────────────────────────────
V5_VIOL_W_GT_SURVIVAL = 0.70
V5_VIOL_W_FORMAT = 0.10
V5_VIOL_W_COMPACT = 0.00
V5_VIOL_W_PASS_RULE_OUT = 0.20          # was 0.05 in V4
V5_VIOL_PASS_KEPT_PENALTY = 0.30
V5_VIOL_COMPACT_BUFFER = V4_VIOL_COMPACT_BUFFER  # 6


def _v5_format_score(
    possible_labels_raw: List[Any],
    tool_required_labels_raw: List[Any],
    shortlist_set: Set[str],
    json_valid: bool,
    has_filtered: bool,
    has_tool_calls: bool,
) -> float:
    """Same graduated format score as V4."""
    if not json_valid:
        return 0.0
    score = V5_FMT_W_JSON
    if has_filtered and has_tool_calls:
        score += V5_FMT_W_SCHEMA
    all_labels: Set[str] = set()
    for x in possible_labels_raw:
        if isinstance(x, dict) and x.get("label"):
            all_labels.add(_normalize_label(x["label"]))
    for x in tool_required_labels_raw:
        if isinstance(x, dict) and x.get("label"):
            all_labels.add(_normalize_label(x["label"]))
    if not all_labels or all_labels <= shortlist_set:
        score += V5_FMT_W_LABELS
    return score


def reward_planner_v5(
    possible_labels_raw: List[Any],
    tool_required_labels_raw: List[Any],
    shortlist: List[str],
    gt_labels: Optional[List[str]],
    tools_called: int,
    tools_total_available: int,
    *,
    json_valid: bool = True,
    has_filtered: bool = True,
    has_tool_calls: bool = True,
) -> Dict[str, float]:
    """Planner reward, V5. See V5 comment block for design.

    Score is normalized to [0, 1] via (raw - raw_min) / (raw_max - raw_min)
    where:
      raw_max = sum of all positive-term maxima
      raw_min = -(sum of actual penalties)
    """
    sl = _to_set(shortlist)

    possible: Set[str] = set()
    for x in possible_labels_raw:
        label = _normalize_label(x.get("label")) if isinstance(x, dict) else ""
        if label in sl:
            possible.add(label)
    tool_required: Set[str] = set()
    for x in tool_required_labels_raw:
        label = _normalize_label(x.get("label")) if isinstance(x, dict) else ""
        if label in sl:
            tool_required.add(label)

    kept = possible | tool_required
    rule_out = sl - kept

    gt = _to_set(gt_labels)
    is_gt_pass = bool(gt) and gt == {PASS_LABEL}
    violation_shortlist = sl - {PASS_LABEL}

    format_score = _v5_format_score(
        possible_labels_raw, tool_required_labels_raw, sl,
        json_valid, has_filtered, has_tool_calls,
    )

    base_metrics: Dict[str, float] = {
        "is_gt_pass": float(is_gt_pass),
        "format_score": float(format_score),
        "json_valid": float(json_valid),
        "possible_cnt": float(len(possible)),
        "tool_required_cnt": float(len(tool_required)),
        "rule_out_cnt": float(len(rule_out)),
        "kept_cnt": float(len(kept)),
        "shortlist_cnt": float(len(sl)),
        "tools_called": float(tools_called),
    }

    # ── Pass case ──
    if is_gt_pass:
        possible_viol_cnt = 0
        support_viol_cnt = 0
        for x in possible_labels_raw:
            label = _normalize_label(x.get("label")) if isinstance(x, dict) else ""
            if label not in violation_shortlist:
                continue
            possible_viol_cnt += 1
            if str(x.get("preliminary_opinion", "")).strip().lower() == "support":
                support_viol_cnt += 1
        non_support_viol_cnt = max(0, possible_viol_cnt - support_viol_cnt)
        possible_violation_penalty = (
            V5_PASS_POSSIBLE_VIOLATION_PENALTY * non_support_viol_cnt
            + V5_PASS_SUPPORT_VIOLATION_PENALTY * support_viol_cnt
        )

        raw = V5_PASS_W_FORMAT * format_score - possible_violation_penalty

        raw_max = V5_PASS_W_FORMAT                       # 1.0
        raw_min = -possible_violation_penalty
        denom = raw_max - raw_min
        score = (raw - raw_min) / denom if denom > 0 else 0.0
        score = max(0.0, min(1.0, score))

        return {
            **base_metrics,
            "score": float(score),
            "possible_violation_cnt": float(possible_viol_cnt),
            "non_support_violation_cnt": float(non_support_viol_cnt),
            "support_violation_cnt": float(support_viol_cnt),
            # Keep the historical metric name populated with the total
            # pass-case penalty so existing dashboards continue to work.
            "support_penalty": float(possible_violation_penalty),
            "possible_violation_penalty": float(possible_violation_penalty),
            "raw_score": float(raw),
            "raw_max": float(raw_max),
            "raw_min": float(raw_min),
            # violation-only fields zeroed for log uniformity
            "gt_survival": 0.0,
            "pass_rule_out_bonus": 0.0,
            "pass_kept_penalty": 0.0,
            "compact_kept": 0.0,
        }

    # ── Violation case (or empty GT) ──
    if gt:
        gt_survival = len(gt & kept) / len(gt)
    else:
        gt_survival = 0.0

    if PASS_LABEL in sl:
        pass_rule_out_bonus = 1.0 if PASS_LABEL in rule_out else 0.0
        pass_kept_penalty = V5_VIOL_PASS_KEPT_PENALTY if PASS_LABEL in kept else 0.0
    else:
        pass_rule_out_bonus = 1.0
        pass_kept_penalty = 0.0

    compact_kept = 1.0 - min(
        1.0,
        max(0, len(kept) - max(1, len(gt))) / V5_VIOL_COMPACT_BUFFER,
    )

    raw = (
        V5_VIOL_W_GT_SURVIVAL * gt_survival
        + V5_VIOL_W_FORMAT * format_score
        + V5_VIOL_W_COMPACT * compact_kept
        + V5_VIOL_W_PASS_RULE_OUT * pass_rule_out_bonus
        - pass_kept_penalty
    )

    raw_max = (
        V5_VIOL_W_GT_SURVIVAL
        + V5_VIOL_W_FORMAT
        + V5_VIOL_W_COMPACT
        + V5_VIOL_W_PASS_RULE_OUT
    )                                                    # 1.0
    raw_min = -pass_kept_penalty
    denom = raw_max - raw_min
    score = (raw - raw_min) / denom if denom > 0 else 0.0
    score = max(0.0, min(1.0, score))

    return {
        **base_metrics,
        "score": float(score),
        "gt_survival": float(gt_survival),
        "pass_rule_out_bonus": float(pass_rule_out_bonus),
        "pass_kept_penalty": float(pass_kept_penalty),
        "compact_kept": float(compact_kept),
        "raw_score": float(raw),
        "raw_max": float(raw_max),
        "raw_min": float(raw_min),
        # pass-only fields zeroed for log uniformity
        "support_violation_cnt": 0.0,
        "support_penalty": 0.0,
    }
