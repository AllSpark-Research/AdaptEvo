#!/usr/bin/env python3
"""Build the unanimous-human, rule-supported Direct-GT training subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _label_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    return tuple(sorted({str(item).strip() for item in values if str(item).strip()}))


def _four_round_vote(metadata: dict[str, Any]) -> tuple[str, ...] | None:
    block = metadata.get("four_round_labels") or {}
    rounds = block.get("rounds") if isinstance(block, dict) else None
    if not isinstance(rounds, list) or len(rounds) != 4:
        return None
    votes = [
        _label_tuple(item.get("labels")) if isinstance(item, dict) else ()
        for item in rounds
    ]
    if not votes[0] or len(set(votes)) != 1:
        return None
    return votes[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    args = parser.parse_args()

    source_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    selected_note_ids: set[str] = set()
    total = unanimous = supported = selected = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w",
        encoding="utf-8",
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            vote = _four_round_vote(metadata)
            if vote is not None:
                unanimous += 1

            verdict = str(
                (metadata.get("rule_consistency_judge") or {}).get("verdict")
                or ""
            ).strip().lower()
            if verdict == "supported":
                supported += 1
            if vote is None or verdict != "supported":
                continue

            top_level_gt = _label_tuple(row.get("label"))
            audit_gt = _label_tuple(
                (metadata.get("audit_input") or {}).get("gt_labels")
            )
            if top_level_gt != vote or audit_gt != vote:
                raise ValueError(
                    f"GT mismatch at input line {line_number}: "
                    f"label={top_level_gt}, audit_gt={audit_gt}, vote={vote}"
                )

            note_id = str(metadata.get("note_id") or "").strip()
            if not note_id:
                raise ValueError(f"missing note_id at input line {line_number}")
            if note_id in selected_note_ids:
                raise ValueError(f"duplicate selected note_id: {note_id}")
            selected_note_ids.add(note_id)

            source_id = str(metadata.get("source") or "unknown")
            source_counts[source_id] += 1
            label_counts["|".join(vote)] += 1
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            selected += 1

    stats = {
        "input": str(args.input),
        "output": str(args.output),
        "filter": {
            "four_round_labels": "all four normalized label sets are identical",
            "rule_consistency_verdict": "supported",
            "reward_mode": "direct GT hybrid reward; adaptive/process reward disabled",
        },
        "total_input": total,
        "four_round_unanimous": unanimous,
        "rule_supported": supported,
        "selected": selected,
        "unique_note_ids": len(selected_note_ids),
        "source_counts": dict(sorted(source_counts.items())),
        "label_counts": dict(label_counts.most_common()),
    }
    args.stats.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
