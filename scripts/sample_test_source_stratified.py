#!/usr/bin/env python3
"""Sample a fixed number of rows per source while preserving GT proportions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def get_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Row is missing metadata")
    return metadata


def get_source(row: dict[str, Any]) -> str:
    metadata = get_metadata(row)
    source = metadata.get("source")
    if not source:
        audit_input = metadata.get("audit_input") or {}
        source = audit_input.get("source_id")
    if not source:
        raise ValueError("Row is missing source")
    return str(source)


def get_note_id(row: dict[str, Any]) -> str:
    metadata = get_metadata(row)
    note_id = metadata.get("note_id")
    if not note_id:
        audit_input = metadata.get("audit_input") or {}
        note_id = audit_input.get("note_id")
    if not note_id:
        raise ValueError("Row is missing note_id")
    return str(note_id)


def get_gt_labels(row: dict[str, Any]) -> tuple[str, ...]:
    metadata = get_metadata(row)
    audit_input = metadata.get("audit_input") or {}
    labels = audit_input.get("gt_labels")
    if not isinstance(labels, list) or not labels:
        four_round = metadata.get("four_round_labels") or {}
        rounds = four_round.get("rounds") or []
        if len(rounds) == 4:
            labels = rounds[3].get("labels")
    if not isinstance(labels, list) or not labels:
        top_level_label = row.get("label")
        if isinstance(top_level_label, str) and top_level_label.strip():
            labels = [part.strip() for part in top_level_label.split(",")]
    if not isinstance(labels, list) or not labels:
        raise ValueError("Row is missing final GT labels")
    normalized = tuple(sorted({str(label).strip() for label in labels if str(label).strip()}))
    if not normalized:
        raise ValueError("GT labels became empty after normalization")
    return normalized


def gt_key(labels: tuple[str, ...]) -> str:
    return " + ".join(labels)


def binary_gt(labels: tuple[str, ...]) -> str:
    return "pass" if labels == ("通过",) else "violation"


def stable_seed(seed: int, source: str, labels: tuple[str, ...]) -> int:
    payload = f"{seed}\n{source}\n{json.dumps(labels, ensure_ascii=False)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def proportional_allocation(
    counts: Counter[tuple[str, ...]],
    target: int,
) -> dict[tuple[str, ...], int]:
    total = sum(counts.values())
    if target > total:
        raise ValueError(f"Cannot sample {target} rows from a source with {total} rows")

    exact = {labels: count * target / total for labels, count in counts.items()}
    allocated = {labels: math.floor(value) for labels, value in exact.items()}
    remaining = target - sum(allocated.values())

    order = sorted(
        counts,
        key=lambda labels: (
            -(exact[labels] - allocated[labels]),
            -counts[labels],
            gt_key(labels),
        ),
    )
    for labels in order[:remaining]:
        allocated[labels] += 1

    if sum(allocated.values()) != target:
        raise AssertionError("Proportional allocation did not reach the target")
    for labels, amount in allocated.items():
        if amount > counts[labels]:
            raise AssertionError(f"Allocation exceeds available rows for {labels}")
    return allocated


def distribution_payload(
    counts: Counter[tuple[str, ...]],
    total: int,
) -> dict[str, dict[str, float | int]]:
    return {
        gt_key(labels): {
            "count": counts[labels],
            "rate": round(counts[labels] / total, 8) if total else 0.0,
        }
        for labels in sorted(counts, key=gt_key)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    indexed_rows: list[tuple[int, dict[str, Any]]] = list(enumerate(rows))
    rows_by_source: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    input_note_ids: list[str] = []

    for row_idx, row in indexed_rows:
        rows_by_source[get_source(row)].append((row_idx, row))
        input_note_ids.append(get_note_id(row))

    if len(input_note_ids) != len(set(input_note_ids)):
        duplicates = [
            note_id for note_id, count in Counter(input_note_ids).items() if count > 1
        ]
        raise ValueError(f"Input contains duplicate note_id values: {duplicates[:20]}")

    selected_indices: set[int] = set()
    source_summaries: dict[str, Any] = {}

    for source in sorted(rows_by_source):
        source_rows = rows_by_source[source]
        if len(source_rows) < args.per_source:
            raise ValueError(
                f"Source {source!r} has only {len(source_rows)} rows, "
                f"below target {args.per_source}"
            )

        strata: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for row_idx, row in source_rows:
            strata[get_gt_labels(row)].append((row_idx, row))

        before_counts = Counter({labels: len(items) for labels, items in strata.items()})
        allocation = proportional_allocation(before_counts, args.per_source)
        after_counts: Counter[tuple[str, ...]] = Counter()

        for labels in sorted(strata, key=gt_key):
            candidates = list(strata[labels])
            rng = random.Random(stable_seed(args.seed, source, labels))
            chosen = rng.sample(candidates, allocation[labels])
            for row_idx, _ in chosen:
                selected_indices.add(row_idx)
            after_counts[labels] = len(chosen)

        before_binary = Counter()
        after_binary = Counter()
        for labels, count in before_counts.items():
            before_binary[binary_gt(labels)] += count
        for labels, count in after_counts.items():
            after_binary[binary_gt(labels)] += count

        max_rate_delta = max(
            abs(after_counts[labels] / args.per_source - before_counts[labels] / len(source_rows))
            for labels in before_counts
        )
        source_summaries[source] = {
            "input_count": len(source_rows),
            "selected_count": sum(after_counts.values()),
            "gt_strata_count": len(before_counts),
            "max_absolute_gt_rate_delta": round(max_rate_delta, 8),
            "before_gt_distribution": distribution_payload(before_counts, len(source_rows)),
            "selected_gt_distribution": distribution_payload(after_counts, args.per_source),
            "before_binary_distribution": {
                key: {
                    "count": before_binary[key],
                    "rate": round(before_binary[key] / len(source_rows), 8),
                }
                for key in ("pass", "violation")
            },
            "selected_binary_distribution": {
                key: {
                    "count": after_binary[key],
                    "rate": round(after_binary[key] / args.per_source, 8),
                }
                for key in ("pass", "violation")
            },
        }

    selected_rows = [row for row_idx, row in indexed_rows if row_idx in selected_indices]
    expected_total = len(rows_by_source) * args.per_source
    if len(selected_rows) != expected_total:
        raise AssertionError(
            f"Selected row count mismatch: {len(selected_rows)} != {expected_total}"
        )

    output_note_ids = [get_note_id(row) for row in selected_rows]
    if len(output_note_ids) != len(set(output_note_ids)):
        raise AssertionError("Output contains duplicate note_id values")

    output_source_counts = Counter(get_source(row) for row in selected_rows)
    invalid_source_counts = {
        source: count
        for source, count in output_source_counts.items()
        if count != args.per_source
    }
    if invalid_source_counts:
        raise AssertionError(f"Incorrect per-source output counts: {invalid_source_counts}")

    write_jsonl(args.output, selected_rows)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "per_source": args.per_source,
        "input_rows": len(rows),
        "selected_rows": len(selected_rows),
        "source_count": len(rows_by_source),
        "unique_note_ids": len(set(output_note_ids)),
        "duplicate_note_ids": len(output_note_ids) - len(set(output_note_ids)),
        "gt_stratification": "full canonical GT label set",
        "allocation": "largest remainder, then deterministic within-stratum sampling",
        "sources": source_summaries,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "input_rows": len(rows),
        "selected_rows": len(selected_rows),
        "source_count": len(rows_by_source),
        "per_source": args.per_source,
        "unique_note_ids": len(set(output_note_ids)),
        "duplicate_note_ids": len(output_note_ids) - len(set(output_note_ids)),
        "output": str(args.output),
        "summary": str(args.summary),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
