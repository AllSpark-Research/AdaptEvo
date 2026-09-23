"""Judge reward stub (optional, used only if Judge participates in training)."""

from __future__ import annotations

from typing import Dict, List, Optional


def reward_judge(
    verdict: str,
    main_was_correct: bool,
    detected_issues: List[str],
    actual_issues: Optional[List[str]] = None,
    critique_quality: float = 0.0,
    hallucination_detected: float = 0.0,
) -> Dict[str, float]:
    """R_judge = 0.5 * verdict_correct + 0.2 * issue_type_accuracy
              + 0.2 * critique_quality + 0.1 * hallucination_detection."""
    # verdict_correct: judge fails iff main was wrong
    if main_was_correct:
        verdict_correct = 1.0 if verdict == "pass" else 0.0
    else:
        verdict_correct = 1.0 if verdict == "fail" else 0.0

    actual = set(actual_issues or [])
    detected = set(detected_issues or [])
    if actual:
        intersection = actual & detected
        issue_acc = len(intersection) / len(actual | detected)
    else:
        issue_acc = 1.0 if not detected else 0.0

    score = (
        0.5 * verdict_correct
        + 0.2 * issue_acc
        + 0.2 * critique_quality
        + 0.1 * hallucination_detected
    )
    return {
        "score": float(score),
        "verdict_correct": float(verdict_correct),
        "issue_accuracy": float(issue_acc),
        "critique_quality": float(critique_quality),
    }
