"""Router-stage metrics, mirroring recompute_top5_label_select_metrics.py.

Adapted for the audit_agentic schema (gt comes from metadata.labels). Reports:

  * success_rate          - fraction of rows where the model returned text
  * parse_rate            - fraction where >=1 label parsed successfully
  * overlap_accuracy      - fraction where predicted ∩ gt is non-empty
  * gt_pass_*             - rows whose gt = ["通过"]: did we keep "通过"?
  * gt_violation_*        - rows whose gt is a violation: did we recall it?
  * pred_label_count.*    - distribution of how many labels the model emitted

Plus router-specific:
  * recall_gt_in_shortlist  - same as overlap_accuracy but framed for router
  * shortlist_size_*        - distribution of |shortlist|
  * over_select_rate        - frac of rows where |shortlist| > target_max
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, List

from .parsing import PASS_LABEL, parse_label_list


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def _target_range(n_candidates: int) -> tuple[int, int]:
    """Conditional routing target: <=8 bypasses; larger queues select 8-12."""
    if n_candidates <= 8:
        return (n_candidates, n_candidates)
    return (8, min(12, n_candidates))


def compute_router_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    success_rows = [r for r in rows if r.get("success")]
    parsed_rows = [r for r in success_rows if r.get("predicted_labels")]
    overlap_correct = sum(1 for r in rows if r.get("overlap_correct"))

    pred_lens = [len(r.get("predicted_labels") or []) for r in success_rows]
    pred_count_dist: Counter = Counter(str(c) for c in pred_lens)

    parse_source_counts: Counter = Counter()
    for r in rows:
        parse_source_counts[str(r.get("parse_source") or "unknown")] += 1

    # GT-pass vs GT-violation breakdown
    gt_pass = [r for r in rows if set(parse_label_list(r.get("gt_labels"))) == {PASS_LABEL}]
    gt_viol = [r for r in rows if set(parse_label_list(r.get("gt_labels"))) != {PASS_LABEL}]
    gt_pass_contains_pass = sum(
        1 for r in gt_pass if PASS_LABEL in set(parse_label_list(r.get("predicted_labels")))
    )
    gt_viol_overlap = sum(1 for r in gt_viol if r.get("overlap_correct"))
    gt_viol_pred_pass_only = sum(
        1 for r in gt_viol if set(parse_label_list(r.get("predicted_labels"))) == {PASS_LABEL}
    )

    # Router-specific: shortlist size vs target range
    in_range = 0
    under_min = 0
    over_max = 0
    for r in success_rows:
        n_cand = len(r.get("candidate_labels") or [])
        lo, hi = _target_range(n_cand)
        sz = len(r.get("predicted_labels") or [])
        if sz < lo:
            under_min += 1
        elif sz > hi:
            over_max += 1
        else:
            in_range += 1

    # Per-source breakdown
    by_source: Dict[str, Dict[str, int]] = {}
    for r in rows:
        s = r.get("source") or "<unknown>"
        bs = by_source.setdefault(s, {"total": 0, "overlap": 0, "gt_pass": 0, "gt_pass_hit": 0, "gt_viol": 0, "gt_viol_hit": 0})
        bs["total"] += 1
        gt_set = set(parse_label_list(r.get("gt_labels")))
        if r.get("overlap_correct"):
            bs["overlap"] += 1
        if gt_set == {PASS_LABEL}:
            bs["gt_pass"] += 1
            if PASS_LABEL in set(parse_label_list(r.get("predicted_labels"))):
                bs["gt_pass_hit"] += 1
        else:
            bs["gt_viol"] += 1
            if r.get("overlap_correct"):
                bs["gt_viol_hit"] += 1
    per_source = {
        s: {
            "total": v["total"],
            "overlap_accuracy": _ratio(v["overlap"], v["total"]),
            "gt_pass_total": v["gt_pass"],
            "gt_pass_hit_rate": _ratio(v["gt_pass_hit"], v["gt_pass"]),
            "gt_violation_total": v["gt_viol"],
            "gt_violation_recall": _ratio(v["gt_viol_hit"], v["gt_viol"]),
        }
        for s, v in sorted(by_source.items(), key=lambda kv: -kv[1]["total"])
    }

    return {
        "total": total,
        "success": len(success_rows),
        "success_rate": _ratio(len(success_rows), total),
        "parsed_rows": len(parsed_rows),
        "parse_rate": _ratio(len(parsed_rows), total),
        "overlap_correct": overlap_correct,
        "overlap_accuracy": _ratio(overlap_correct, total),
        "gt_pass_total": len(gt_pass),
        "gt_pass_contains_pass": gt_pass_contains_pass,
        "gt_pass_contains_pass_rate": _ratio(gt_pass_contains_pass, len(gt_pass)),
        "gt_violation_total": len(gt_viol),
        "gt_violation_contains_gt": gt_viol_overlap,
        "gt_violation_contains_gt_rate": _ratio(gt_viol_overlap, len(gt_viol)),
        "gt_violation_pred_pass_only": gt_viol_pred_pass_only,
        "gt_violation_pred_pass_only_rate": _ratio(gt_viol_pred_pass_only, len(gt_viol)),
        "pred_label_count": {
            "avg": statistics.mean(pred_lens) if pred_lens else 0.0,
            "median": statistics.median(pred_lens) if pred_lens else 0.0,
            "max": max(pred_lens) if pred_lens else 0,
        },
        "pred_label_count_distribution": dict(
            sorted(pred_count_dist.items(), key=lambda kv: int(kv[0]))
        ),
        "shortlist_size_vs_target": {
            "in_range": in_range,
            "under_min": under_min,
            "over_max": over_max,
            "in_range_rate": _ratio(in_range, len(success_rows)),
        },
        "parse_source_counts": dict(parse_source_counts),
        "per_source": per_source,
    }
