"""Router label-select reward for audit_agentic RL.

Uses audit_agentic.eval.parsing.parse_answer_labels (same logic as eval
scripts) so results are directly comparable to evaluation metrics.

score = 1.0 if any GT label is in predicted shortlist
        - 0.1 per non-candidate label in prediction
        - 0.2 if <answer>...</answer> format is missing
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

# audit_agentic must be on PYTHONPATH (set by run_agent_app.sh / base script)
from audit_agentic.eval.parsing import parse_answer_labels

PASS_LABEL = "通过"

_ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|<\\answer>)", re.DOTALL | re.IGNORECASE)
_NON_CANDIDATE_PENALTY = float(os.environ.get("ROUTER_NON_CANDIDATE_PENALTY", "0.1"))
_FORMAT_PENALTY        = float(os.environ.get("ROUTER_FORMAT_PENALTY", "0.2"))


def _read_gt_labels(sample) -> list[str]:
    """Read GT from metadata.audit_input.gt_labels (list) or sample.label."""
    md = sample.metadata if isinstance(sample.metadata, dict) else {}

    # Prefer the structured list in audit_input
    ai = md.get("audit_input") or {}
    gt = ai.get("gt_labels") or ai.get("labels")
    if isinstance(gt, list) and gt:
        return [str(x).strip() for x in gt if str(x).strip()]

    # Fallback: metadata.labels
    labels_field = md.get("labels") or md.get("active_labels")
    if isinstance(labels_field, list):
        return [str(x).strip() for x in labels_field if str(x).strip()]

    # Last resort: sample.label (may be comma-joined string)
    label = getattr(sample, "label", None) or ""
    if not isinstance(label, str):
        label = str(label)
    label = label.strip()
    if not label:
        return []
    # Split on comma in case it's comma-joined
    parts = [p.strip() for p in label.split(",") if p.strip()]
    return parts if len(parts) > 1 else [label]


def _read_candidate_labels(sample) -> list[str]:
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    cands = md.get("candidate_labels")
    if isinstance(cands, list) and cands:
        return [str(x).strip() for x in cands if str(x).strip()]
    ai = md.get("audit_input") or {}
    cands = ai.get("candidate_labels")
    if isinstance(cands, list) and cands:
        return [str(x).strip() for x in cands if str(x).strip()]
    return []


def _score_one(sample) -> dict[str, Any]:
    response = getattr(sample, "response", "") or ""
    gt = _read_gt_labels(sample)
    candidates = _read_candidate_labels(sample)

    # Check format
    has_answer_tag = bool(_ANSWER_RE.search(response))

    # Use audit_agentic's parse_answer_labels (handles slash-labels, fuzzy match, etc.)
    predicted, answer_text, parse_source, non_candidate_labels, fuzzy_mappings = parse_answer_labels(
        response, candidates
    )

    gt_set = set(gt)
    pred_set = set(predicted)
    overlap_correct = bool(gt_set and pred_set and (gt_set & pred_set))
    base_reward = 1.0 if overlap_correct else 0.0

    non_candidate_count = len(non_candidate_labels)
    format_error = 0 if has_answer_tag else 1
    non_candidate_penalty = _NON_CANDIDATE_PENALTY * non_candidate_count
    format_penalty = _FORMAT_PENALTY * format_error
    score = base_reward - non_candidate_penalty - format_penalty

    gt_is_pass = gt_set == {PASS_LABEL}
    pred_has_pass = PASS_LABEL in pred_set

    return {
        "score": score,
        "base_reward": base_reward,
        "overlap_correct": int(overlap_correct),
        "format_ok": int(format_error == 0),
        "format_error": format_error,
        "format_penalty": -format_penalty,
        "non_candidate_count": non_candidate_count,
        "non_candidate_penalty": -non_candidate_penalty,
        "raw_predicted_labels": predicted,
        "ground_truth_labels": gt,
        "candidate_label_count": len(candidates),
        "parse_source": parse_source,
        "non_candidate_labels": non_candidate_labels,
        "fuzzy_label_mappings": fuzzy_mappings,
        "pred_label_count": len(predicted),
        "gt_is_pass": int(gt_is_pass),
        "gt_is_violation": int(bool(gt_set) and not gt_is_pass),
        "pred_has_pass": int(pred_has_pass),
    }


async def router_label_select_reward(args, samples, **kwargs):
    if isinstance(samples, list):
        return [_score_one(s) for s in samples]
    return _score_one(samples)
