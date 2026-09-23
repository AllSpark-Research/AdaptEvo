"""Main-agent reward functions.

The main agent does TWO turns inside one continuation conversation:
  * Router turn (turn 1): outputs <answer>shortlist labels</answer>
  * Final turn (turn 2): outputs predict_label JSON

Reward versions:
  * V2:
      - ``reward_recall``: router shortlist vs GT (recall + size/invalid penalties)
      - ``reward_final``: final predict_label vs GT (hybrid: exact > overlap > binary)
  * V5 (preferred by relax_app):
      - ``reward_recall_v5``: pass / violation case branches; explicitly penalises
        ``"通过"`` mixed into violation shortlists and oversized pass shortlists.
      - ``reward_final``: latest final-only label score: overlap with GT is
        full credit; wrong violation label with correct violation direction is
        partial credit; pass/violation direction mistakes are zero.

Active raw reward scores are kept in [0, 1]. The custom Relax advantage hook
may z-normalise these scores later, so advantages are not constrained to [0, 1].

The current Relax agent app uses final-only ``score/overall`` for both main and
planner samples, then subtracts parse-format penalties outside this module.
``combine_main_score`` is kept for diagnostics/backwards compatibility.
Legacy ``reward_main``/``reward_main_final``/``reward_router`` kept for
backwards-compat with V1 callers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set


PASS_LABEL = "通过"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _normalize_label(label: str) -> str:
    """Normalize optional numeric-id prefixes before reward matching."""
    if isinstance(label, str) and "|" in label:
        return label.split("|", 1)[1]
    return label


def _to_set(xs: Optional[List[str]]) -> Set[str]:
    return {_normalize_label(x) for x in (xs or [])}


def label_score(predicted: List[str], gt: List[str]) -> float:
    """Simple F1-style label match (kept for legacy callers)."""
    p = _to_set(predicted)
    g = _to_set(gt)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    precision = tp / len(p)
    recall = tp / len(g)
    return 2 * precision * recall / (precision + recall)


# ─────────────────────────────────────────────────────────────────────────────
# V2 rewards (preferred)
# ─────────────────────────────────────────────────────────────────────────────


def reward_recall(
    shortlist: List[str],
    gt_labels: Optional[List[str]],
    candidate_labels: List[str],
    target_size: int = 10,
) -> Dict[str, float]:
    """Router stage reward. Returns dict with score + sub-metrics."""
    cand_set = _to_set(candidate_labels)
    sl = _to_set(shortlist)
    gt = _to_set(gt_labels)

    if len(sl) == 0:
        return {"score": -0.5, "gt_recall": 0.0, "size_penalty": 0.0, "invalid_count": 0.0}

    if gt:
        recall = len(gt & sl) / len(gt)
    else:
        recall = 1.0 if not sl else 0.0

    over_size = max(0, len(sl) - target_size)
    size_penalty = 0.03 * over_size

    invalid_count = len(sl - cand_set)
    invalid_penalty = 0.3 * invalid_count

    score = recall - size_penalty - invalid_penalty
    score = max(0.0, min(1.0, score))

    return {
        "score": float(score),
        "gt_recall": float(recall),
        "size_penalty": float(size_penalty),
        "invalid_count": float(invalid_count),
        "shortlist_size": float(len(sl)),
    }


def reward_final(
    predict_label: List[str],
    gt_labels: Optional[List[str]],
    mode: str = "hybrid",
) -> Dict[str, float]:
    """Final stage reward. Returns dict with score + nOA / wOA / overlap metrics."""
    pred = _to_set(predict_label)
    gt = _to_set(gt_labels)

    pred_is_pass = pred == {PASS_LABEL}
    gt_is_pass = gt == {PASS_LABEL}

    n_oa = 1.0 if pred == gt else 0.0
    w_oa = 1.0 if pred_is_pass == gt_is_pass else 0.0
    overlap = 1.0 if (pred & gt) else 0.0
    label_f1 = label_score(predict_label, gt_labels or [])

    if mode == "exact":
        score = n_oa
    elif mode == "binary":
        score = w_oa
    else:  # hybrid / latest final-only semantics
        if gt_is_pass:
            # GT=通过: only exact pass is rewarded.
            score = 1.0 if pred_is_pass else 0.0
        elif pred_is_pass:
            # GT=违规 but model predicts 通过: hard wrong.
            score = 0.0
        elif pred & gt:
            # Any overlapping violation label is treated as correct enough.
            score = 1.0
        else:
            # Correct binary direction (违规) but wrong violation label.
            score = 0.6

    return {
        "score": _clamp01(score),
        "nOA": float(n_oa),
        "wOA": float(w_oa),
        "overlap": float(overlap),
        "label_f1": float(label_f1),
    }


def combine_main_score(
    r_recall: Dict[str, float],
    r_final: Dict[str, float],
    alpha: float = 0.3,
    beta: float = 0.7,
) -> Dict[str, float]:
    """Combine R_recall + R_final into one main-agent score (used by advantage)."""
    recall_score = _clamp01(r_recall["score"])
    final_score = _clamp01(r_final["score"])
    weight_sum = max(0.0, float(alpha) + float(beta))
    if weight_sum <= 0:
        score = 0.0
    else:
        score = (float(alpha) * recall_score + float(beta) * final_score) / weight_sum
    return {
        "score": _clamp01(score),
        "recall_score": float(recall_score),
        "final_score": float(final_score),
        "alpha": float(alpha),
        "beta": float(beta),
        "weight_sum": float(weight_sum),
    }


# ─────────────────────────────────────────────────────────────────────────────
# V3 rewards
# ─────────────────────────────────────────────────────────────────────────────
# Router reward V3: split by GT-is-pass vs GT-is-violation.
#
# Motivation (from eval analysis on the V2-trained checkpoint):
#   * V2's pass-case reward was "GT={通过} → recall=1.0 as long as 通过 is in
#     the shortlist", with no penalty for piling on extra violation labels.
#     The trained model exploited this by always returning very large
#     shortlists (avg 8.6, mostly capped at 10) regardless of whether the
#     note actually looked violating.
#   * V2's violation-case reward never penalised mixing "通过" into the
#     shortlist for a violating note, so the same fix-everything strategy
#     was costless from the recall side.
#
# V3 keeps the same shape (single score in [0, 1]) but:
#   * On pass cases, encourages a small shortlist anchored on "通过";
#   * On violation cases, penalises "通过" being mixed in.
# ─────────────────────────────────────────────────────────────────────────────


# ── tunable constants (kept here so future calibration is one place) ─────────
V3_INVALID_LABEL_WEIGHT = 0.3       # per off-candidate label
V3_PASS_EXTRA_SIZE_WEIGHT = 0.05    # per shortlist item over out_min in pass cases
V3_PASS_RANK_BONUS = 0.05           # bonus when "通过" is in the top 3 of a pass shortlist
V3_VIOL_PASS_PENALTY = 0.2          # penalty when "通过" appears in a violation shortlist
V3_VIOL_OVERSIZE_WEIGHT = 0.03      # per shortlist item over out_max in violation cases
V3_VIOL_UNDERSIZE_WEIGHT = 0.03     # per missing item below out_min in violation cases


def _router_target_size(n_cand: int) -> tuple[int, int]:
    """Mirror the size policy baked into prompt_template/main_router.jinja2."""
    if n_cand <= 6:
        return min(n_cand, 4) - 1, n_cand
    if n_cand <= 12:
        return (n_cand + 1) // 2, (2 * n_cand + 2) // 3
    return 8, 10


def reward_recall_v3(
    shortlist: List[str],
    gt_labels: Optional[List[str]],
    candidate_labels: List[str],
) -> Dict[str, float]:
    """Router-stage reward, V3.

    ``shortlist`` is expected to preserve order and be deduplicated by the
    caller (relax_app.agent already calls ``parse_answer_labels`` and dedupes
    via ``dict.fromkeys``). Sub-metrics in the returned dict are all flat
    floats so they can be passed straight to ``RoleAgent.record(**metadata)``.
    """
    candidate_list = list(candidate_labels or [])
    n_cand = len(candidate_list)
    out_min, out_max = _router_target_size(n_cand)

    cand_set = _to_set(candidate_list)
    pred_set = _to_set(shortlist)
    gt = _to_set(gt_labels)

    has_pass = PASS_LABEL in pred_set
    invalid_cnt = len(pred_set - cand_set)
    invalid_penalty = V3_INVALID_LABEL_WEIGHT * invalid_cnt
    is_gt_pass = bool(gt) and gt == {PASS_LABEL}

    # Empty shortlist: hard floor, but still surface enough sub-metrics so
    # logging doesn't have None gaps.
    if len(shortlist) == 0:
        return {
            "score": -0.5,
            "is_gt_pass": float(is_gt_pass),
            "has_pass": 0.0,
            "gt_recall": 0.0,
            "gt_any_hit": 0.0,
            "pass_hit": 0.0,
            "pass_rank_bonus": 0.0,
            "pass_penalty": 0.0,
            "size_penalty": 0.0,
            "extra_size_penalty": 0.0,
            "too_few_penalty": 0.0,
            "invalid_cnt": float(invalid_cnt),
            "invalid_penalty": float(invalid_penalty),
            "shortlist_size": 0.0,
            "out_min": float(out_min),
            "out_max": float(out_max),
        }

    if is_gt_pass:
        pass_hit = 1.0 if has_pass else 0.0
        extra_size_penalty = V3_PASS_EXTRA_SIZE_WEIGHT * max(0, len(shortlist) - out_min)
        if has_pass and shortlist.index(PASS_LABEL) <= 2:
            pass_rank_bonus = V3_PASS_RANK_BONUS
        else:
            pass_rank_bonus = 0.0
        score = pass_hit + pass_rank_bonus - extra_size_penalty - invalid_penalty
        score = max(0.0, min(1.0, score))
        return {
            "score": float(score),
            "is_gt_pass": 1.0,
            "has_pass": float(has_pass),
            "gt_recall": 1.0 if has_pass else 0.0,
            "gt_any_hit": float(has_pass),
            "pass_hit": float(pass_hit),
            "pass_rank_bonus": float(pass_rank_bonus),
            "pass_penalty": 0.0,
            "size_penalty": 0.0,
            "extra_size_penalty": float(extra_size_penalty),
            "too_few_penalty": 0.0,
            "invalid_cnt": float(invalid_cnt),
            "invalid_penalty": float(invalid_penalty),
            "shortlist_size": float(len(shortlist)),
            "out_min": float(out_min),
            "out_max": float(out_max),
        }

    # Violation case (GT is non-pass; may be empty, treat as "no GT to hit").
    if gt:
        gt_recall = len(pred_set & gt) / len(gt)
    else:
        gt_recall = 0.0
    gt_any_hit = 1.0 if (pred_set & gt) else 0.0
    pass_penalty = V3_VIOL_PASS_PENALTY if has_pass else 0.0
    size_penalty = V3_VIOL_OVERSIZE_WEIGHT * max(0, len(shortlist) - out_max)
    too_few_penalty = V3_VIOL_UNDERSIZE_WEIGHT * max(0, out_min - len(shortlist))

    score = (
        0.8 * gt_recall
        + 0.2 * gt_any_hit
        - pass_penalty
        - size_penalty
        - too_few_penalty
        - invalid_penalty
    )
    score = max(0.0, min(1.0, score))

    return {
        "score": float(score),
        "is_gt_pass": 0.0,
        "has_pass": float(has_pass),
        "gt_recall": float(gt_recall),
        "gt_any_hit": float(gt_any_hit),
        "pass_hit": 0.0,
        "pass_rank_bonus": 0.0,
        "pass_penalty": float(pass_penalty),
        "size_penalty": float(size_penalty),
        "extra_size_penalty": 0.0,
        "too_few_penalty": float(too_few_penalty),
        "invalid_cnt": float(invalid_cnt),
        "invalid_penalty": float(invalid_penalty),
        "shortlist_size": float(len(shortlist)),
        "out_min": float(out_min),
        "out_max": float(out_max),
    }


# ─────────────────────────────────────────────────────────────────────────────
# V5 rewards
# ─────────────────────────────────────────────────────────────────────────────
# Changes from V3:
#   * pass_rank_bonus:           0.05 → 0.2
#   * extra_size_penalty weight: 0.05 → 0.1
#   * invalid_penalty weight:    0.3  → 0.1  (both pass & violation cases)
#   * Violation case: removed size_penalty and too_few_penalty
#   * Violation case: added gt_rank_bonus (+0.2 if any GT label in top-3)
#   * Violation case: pass_penalty: 0.2 → 0.1
#   * All scores normalized to [0, 1] via (raw - raw_min) / (raw_max - raw_min)
#     where raw_max = sum of positive-term maxima, raw_min = -(actual penalties)
# ─────────────────────────────────────────────────────────────────────────────

V5_INVALID_LABEL_WEIGHT = 0.1       # per off-candidate label (was 0.3 in V3)
V5_PASS_EXTRA_SIZE_WEIGHT = 0.1     # per shortlist item over out_min in pass cases (was 0.05)
V5_PASS_RANK_BONUS = 0.2            # bonus when "通过" is in top 3 (was 0.05)
V5_VIOL_PASS_PENALTY = 0.1          # penalty when "通过" in violation shortlist (was 0.2)
V5_VIOL_GT_RANK_BONUS = 0.2         # bonus when any GT label is in top 3


def reward_recall_v5(
    shortlist: List[str],
    gt_labels: Optional[List[str]],
    candidate_labels: List[str],
) -> Dict[str, float]:
    """Router-stage reward, V5. See V5 comment block for design notes.

    Score is normalized to [0, 1] via (raw - raw_min) / (raw_max - raw_min)
    where:
      raw_max = sum of all positive-term maxima  (1.2 for both cases)
      raw_min = -(sum of actual penalties)        (data-dependent)
    """
    candidate_list = list(candidate_labels or [])
    n_cand = len(candidate_list)
    out_min, out_max = _router_target_size(n_cand)

    cand_set = _to_set(candidate_list)
    pred_set = _to_set(shortlist)
    gt = _to_set(gt_labels)

    has_pass = PASS_LABEL in pred_set
    invalid_cnt = len(pred_set - cand_set)
    invalid_penalty = V5_INVALID_LABEL_WEIGHT * invalid_cnt
    is_gt_pass = bool(gt) and gt == {PASS_LABEL}

    if len(shortlist) == 0:
        return {
            "score": 0.0,
            "is_gt_pass": float(is_gt_pass),
            "has_pass": 0.0,
            "gt_recall": 0.0,
            "gt_any_hit": 0.0,
            "pass_hit": 0.0,
            "pass_rank_bonus": 0.0,
            "pass_penalty": 0.0,
            "size_penalty": 0.0,
            "extra_size_penalty": 0.0,
            "too_few_penalty": 0.0,
            "invalid_cnt": float(invalid_cnt),
            "invalid_penalty": float(invalid_penalty),
            "shortlist_size": 0.0,
            "out_min": float(out_min),
            "out_max": float(out_max),
            "gt_rank_bonus": 0.0,
            "raw_score": 0.0,
            "raw_max": 1.2,
            "raw_min": 0.0,
        }

    if is_gt_pass:
        # ── Case A: GT = "通过" ──
        pass_hit = 1.0 if has_pass else 0.0
        extra_size_penalty = V5_PASS_EXTRA_SIZE_WEIGHT * max(0, len(shortlist) - out_min)
        if has_pass and shortlist.index(PASS_LABEL) <= 2:
            pass_rank_bonus = V5_PASS_RANK_BONUS
        else:
            pass_rank_bonus = 0.0

        raw = pass_hit + pass_rank_bonus - extra_size_penalty - invalid_penalty

        positive_max = 1.0 + V5_PASS_RANK_BONUS          # 1.2
        neg_sum = extra_size_penalty + invalid_penalty
        raw_max = positive_max
        raw_min = -neg_sum
        denom = raw_max - raw_min
        score = (raw - raw_min) / denom if denom > 0 else 0.0
        score = max(0.0, min(1.0, score))

        return {
            "score": float(score),
            "is_gt_pass": 1.0,
            "has_pass": float(has_pass),
            "gt_recall": 1.0 if has_pass else 0.0,
            "gt_any_hit": float(has_pass),
            "pass_hit": float(pass_hit),
            "pass_rank_bonus": float(pass_rank_bonus),
            "pass_penalty": 0.0,
            "size_penalty": 0.0,
            "extra_size_penalty": float(extra_size_penalty),
            "too_few_penalty": 0.0,
            "invalid_cnt": float(invalid_cnt),
            "invalid_penalty": float(invalid_penalty),
            "shortlist_size": float(len(shortlist)),
            "out_min": float(out_min),
            "out_max": float(out_max),
            "gt_rank_bonus": 0.0,
            "raw_score": float(raw),
            "raw_max": float(raw_max),
            "raw_min": float(raw_min),
        }

    # ── Case B: GT = 违规 ──
    if gt:
        gt_recall = len(pred_set & gt) / len(gt)
    else:
        gt_recall = 0.0
    gt_any_hit = 1.0 if (pred_set & gt) else 0.0

    top3 = set(shortlist[:3])
    gt_rank_bonus = V5_VIOL_GT_RANK_BONUS if (gt & top3) else 0.0

    pass_penalty = V5_VIOL_PASS_PENALTY if has_pass else 0.0

    raw = (
        0.8 * gt_recall
        + 0.2 * gt_any_hit
        + gt_rank_bonus
        - pass_penalty
        - invalid_penalty
    )

    positive_max = 0.8 + 0.2 + V5_VIOL_GT_RANK_BONUS      # 1.2
    neg_sum = pass_penalty + invalid_penalty
    raw_max = positive_max
    raw_min = -neg_sum
    denom = raw_max - raw_min
    score = (raw - raw_min) / denom if denom > 0 else 0.0
    score = max(0.0, min(1.0, score))

    return {
        "score": float(score),
        "is_gt_pass": 0.0,
        "has_pass": float(has_pass),
        "gt_recall": float(gt_recall),
        "gt_any_hit": float(gt_any_hit),
        "pass_hit": 0.0,
        "pass_rank_bonus": 0.0,
        "pass_penalty": float(pass_penalty),
        "size_penalty": 0.0,
        "extra_size_penalty": 0.0,
        "too_few_penalty": 0.0,
        "invalid_cnt": float(invalid_cnt),
        "invalid_penalty": float(invalid_penalty),
        "shortlist_size": float(len(shortlist)),
        "out_min": float(out_min),
        "out_max": float(out_max),
        "gt_rank_bonus": float(gt_rank_bonus),
        "raw_score": float(raw),
        "raw_max": float(raw_max),
        "raw_min": float(raw_min),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Legacy V1 wrappers (kept for old callers)
# ─────────────────────────────────────────────────────────────────────────────


def reward_router(
    shortlist_labels: List[str],
    gt_labels: Optional[List[str]],
    invalid_labels: int = 0,
    weight: float = 1.0,
    size_penalty_weight: float = 0.05,
    invalid_label_weight: float = 0.2,
) -> Dict[str, float]:
    """V1 router reward — kept for backwards compat. Prefer ``reward_recall``."""
    if not gt_labels:
        gt_recall = 0.0
    else:
        gt_recall = len(_to_set(shortlist_labels) & _to_set(gt_labels)) / max(1, len(gt_labels))
    size_pen = max(0, len(shortlist_labels) - 3) * size_penalty_weight
    score = weight * gt_recall - size_pen - invalid_label_weight * invalid_labels
    return {
        "score": float(score),
        "gt_recall": float(gt_recall),
        "shortlist_size": float(len(shortlist_labels)),
        "invalid_labels": float(invalid_labels),
    }


def reward_main_final(
    predict_label: List[str],
    gt_labels: Optional[List[str]],
    judge_pass: bool = True,
    revision_gain: float = 0.0,
    hallucination: float = 0.0,
    over_select: float = 0.0,
) -> Dict[str, float]:
    """V1 final reward — kept for backwards compat. Prefer ``reward_final``."""
    base = label_score(predict_label, gt_labels or [])
    score = (
        1.0 * base
        + 0.1 * (1.0 if judge_pass else 0.0)
        + 0.1 * revision_gain
        - 0.2 * hallucination
        - 0.1 * over_select
    )
    return {
        "score": float(score),
        "label_f1": float(base),
        "judge_pass": float(judge_pass),
        "revision_gain": float(revision_gain),
    }


def reward_main(
    router_metrics: Dict[str, float],
    initial_metrics: Dict[str, float],
    final_metrics: Dict[str, float],
) -> Dict[str, float]:
    """V1 combined main reward — kept for backwards compat. Prefer ``combine_main_score``."""
    score = (
        0.3 * router_metrics.get("score", 0.0)
        + 0.2 * initial_metrics.get("score", 0.0)
        + 0.5 * final_metrics.get("score", 0.0)
    )
    return {
        "score": float(score),
        "router_score": float(router_metrics.get("score", 0.0)),
        "initial_score": float(initial_metrics.get("score", 0.0)),
        "final_score": float(final_metrics.get("score", 0.0)),
    }
