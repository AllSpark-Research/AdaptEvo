"""Step 3: execute planner's tool_calls via ToolExecutor (env query, no LLM).

Reads:
  audit_agentic/traces/step2_planner/rows.jsonl

Writes:
  audit_agentic/traces/step3_tools/rows.jsonl     (per-row observations + planner output)
  audit_agentic/traces/step3_tools/metrics.json   (success/failure breakdown)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = "/path/to/local-assets"
sys.path.insert(0, ROOT)

from audit_agentic.environment import ToolExecutor
from audit_agentic.schemas import ToolCall


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--planner-rows", default=f"{ROOT}/audit_agentic/traces/step2_planner/rows.jsonl")
    p.add_argument("--output-dir", default=f"{ROOT}/audit_agentic/traces/step3_tools")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="append to existing rows and skip completed row_idx/note_id")
    return p.parse_args()


def _row_key(row: dict) -> str:
    if row.get("row_idx") is not None:
        return f"idx:{row['row_idx']}"
    if row.get("note_id") is not None:
        return f"note:{row['note_id']}"
    return ""


def _load_existing_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[tools] skip malformed existing row in {path}", file=sys.stderr)
    return rows


def _rewrite_existing_results(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    executor = ToolExecutor()

    n_rows_with_calls = 0
    n_calls_total = 0
    n_calls_ok = 0
    n_calls_failed = 0
    err_kinds: Counter = Counter()
    tool_status: Counter = Counter()
    per_tool_ok: Counter = Counter()

    planner_rows = []
    for line in open(args.planner_rows):
        row = json.loads(line)
        if not row.get("success"):
            continue
        planner_rows.append(row)

    def _process(row: dict) -> dict:
        nid = row["note_id"]
        src = row.get("source", "")
        gt = row.get("gt_labels") or []
        planner_out = row.get("planner_output") or {}
        tool_calls_raw = planner_out.get("tool_calls") or []

        observations = []
        if tool_calls_raw:
            # bind note_id/source_id for this row
            bound = executor.with_defaults(note_id=nid, source_id=src)
            call_objs = [
                ToolCall(
                    tool_name=tc.get("tool_name", ""),
                    args=tc.get("args") if isinstance(tc.get("args"), dict) else {},
                    reason=str(tc.get("reason", "")),
                )
                for tc in tool_calls_raw
                if isinstance(tc, dict)
            ]
            obs_objs = bound.execute(call_objs)
            for obs in obs_objs:
                observations.append(obs.model_dump())

        return {
            "row_idx": row.get("row_idx"),
            "note_id": nid,
            "source": src,
            "gt_labels": gt,
            "router_shortlist": row.get("router_shortlist"),
            "planner_output": planner_out,
            "tool_observations": observations,
        }

    existing_rows = _load_existing_results(rows_path) if args.resume else []
    if args.resume and rows_path.exists():
        _rewrite_existing_results(rows_path, existing_rows)
    completed = {_row_key(r) for r in existing_rows}
    completed.discard("")
    todo_rows = [r for r in planner_rows if _row_key(r) not in completed]
    if args.resume and existing_rows:
        print(f"[tools] resume: loaded {len(existing_rows)} existing rows, todo={len(todo_rows)}", file=sys.stderr)

    rows_out = list(existing_rows)
    t0 = time.time()
    done = 0
    print_every = max(1, len(todo_rows) // 20) if todo_rows else 1
    workers = max(1, args.concurrency)
    print(f"[tools] rows={len(planner_rows)}  todo={len(todo_rows)}  conc={workers}", file=sys.stderr)

    mode = "a" if args.resume and rows_path.exists() else "w"
    with rows_path.open(mode, encoding="utf-8") as f:
        if workers == 1:
            for row in todo_rows:
                result = _process(row)
                rows_out.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                if done % print_every == 0 or done == len(todo_rows):
                    rate = done / max(0.001, time.time() - t0)
                    print(f"  [{done}/{len(todo_rows)}]  {rate:.1f} rows/s  elapsed={time.time()-t0:.0f}s", file=sys.stderr)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_process, row) for row in todo_rows]
                for fut in as_completed(futs):
                    result = fut.result()
                    rows_out.append(result)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    done += 1
                    if done % print_every == 0 or done == len(todo_rows):
                        rate = done / max(0.001, time.time() - t0)
                        print(f"  [{done}/{len(todo_rows)}]  {rate:.1f} rows/s  elapsed={time.time()-t0:.0f}s", file=sys.stderr)

    for row in rows_out:
        observations = row.get("tool_observations") or []
        if observations:
            n_rows_with_calls += 1
        for obs in observations:
            n_calls_total += 1
            tool_name = obs.get("tool_name", "")
            status = obs.get("status", "")
            tool_status[(tool_name, status)] += 1
            if status == "ok":
                n_calls_ok += 1
                per_tool_ok[tool_name] += 1
            else:
                n_calls_failed += 1
                err_kinds[obs.get("error") or "unknown"] += 1

    metrics = {
        "total_rows": len(rows_out),
        "n_existing_loaded": len(existing_rows),
        "n_new_processed": len(todo_rows),
        "rows_with_calls": n_rows_with_calls,
        "rows_with_calls_rate": n_rows_with_calls / max(1, len(rows_out)),
        "total_tool_calls": n_calls_total,
        "tool_calls_ok": n_calls_ok,
        "tool_calls_failed": n_calls_failed,
        "ok_rate": n_calls_ok / max(1, n_calls_total),
        "wall_seconds": round(time.time() - t0, 2),
        "per_tool_ok_count": dict(per_tool_ok),
        "per_tool_status_count": {f"{k[0]}.{k[1]}": v for k, v in tool_status.items()},
        "failure_kinds": dict(err_kinds),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    print(f"wrote rows  -> {rows_path}")
    print(f"wrote stats -> {metrics_path}")
    print()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
