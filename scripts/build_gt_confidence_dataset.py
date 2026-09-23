#!/usr/bin/env python3
"""Merge Rule-consistency Judge verdicts into four-round Agentic RL data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPORT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPORT_ROOT / "data/train_agenticrl_four_round_clean.jsonl"
DEFAULT_JUDGE_RESULTS = (
    REPORT_ROOT
    / "outputs/kimi_k3_rule_consistency_train_round1_first_gt_20260821/results.jsonl"
)
DEFAULT_OUTPUT = REPORT_ROOT / "data/train_agenticrl_four_round_gt_confidence.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--judge-results", type=Path, default=DEFAULT_JUDGE_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def _identity(row: dict[str, Any]) -> tuple[str, str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else row
    return (
        str(metadata.get("note_id") or ""),
        str(metadata.get("history_id") or ""),
        str(metadata.get("source") or ""),
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(REPORT_ROOT))
    from audit_agentic.rewards.reward_adaptive import (
        extract_four_round_labels,
        human_agreement_confidence,
        rule_support_gate,
    )

    judge_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    judge_statuses = Counter()
    with args.judge_results.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            result = json.loads(line)
            verdict = str(result.get("verdict") or "").strip().lower()
            valid = bool(result.get("success")) and verdict in {
                "supported",
                "ambiguous",
                "unsupported",
            }
            judge_statuses["valid" if valid else "invalid"] += 1
            if not valid:
                continue
            key = _identity(result)
            if key in judge_by_identity:
                raise ValueError(f"Duplicate valid Rule Judge result for {key}")
            judge_by_identity[key] = result

    output_lines: list[str] = []
    missing: list[tuple[str, str, str]] = []
    verdict_counts = Counter()
    confidence_buckets = Counter()
    total_rows = 0
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            metadata = row.setdefault("metadata", {})
            key = _identity(row)
            judge = judge_by_identity.get(key)
            if judge is None:
                missing.append(key)
                continue
            verdict = str(judge["verdict"]).strip().lower()
            human_labels = extract_four_round_labels(metadata)
            human_confidence = human_agreement_confidence(human_labels)
            gate = rule_support_gate(verdict)
            confidence = human_confidence * gate
            metadata["rule_consistency_judge"] = {
                "schema_version": 1,
                "verdict": verdict,
                "issue_type": str(judge.get("issue_type") or ""),
                "key_rule_label": str(judge.get("key_rule_label") or ""),
                "brief_reason": str(judge.get("brief_reason") or ""),
                "judge_source": str(args.judge_results),
            }
            metadata["gt_confidence"] = {
                "schema_version": 1,
                "human_agreement_confidence": human_confidence,
                "rule_support_verdict": verdict,
                "rule_support_gate": gate,
                "value": confidence,
                "formula": "human_agreement_confidence * rule_support_gate",
            }
            verdict_counts[verdict] += 1
            confidence_buckets[f"{confidence:.6f}"] += 1
            output_lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))

    if missing:
        raise ValueError(
            f"Missing valid Rule Judge results for {len(missing)} rows; first={missing[:5]}"
        )
    if len(output_lines) != total_rows:
        raise ValueError(f"Output row count {len(output_lines)} != input row count {total_rows}")

    _atomic_write(args.output, "\n".join(output_lines) + "\n")
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary = {
        "input": str(args.input),
        "judge_results": str(args.judge_results),
        "output": str(args.output),
        "rows": total_rows,
        "judge_statuses": dict(judge_statuses),
        "verdict_counts": dict(verdict_counts),
        "gt_confidence_distribution": dict(
            sorted(confidence_buckets.items(), key=lambda item: float(item[0]), reverse=True)
        ),
    }
    _atomic_write(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
