"""Run main_router stage over a jsonl file and report metrics.

Usage:
  python -m audit_agentic.eval.run_router_eval \\
      --input audit_agentic/data/12_risk_domain.jsonl \\
      --output traces/router_eval.jsonl \\
      --metrics traces/router_eval_metrics.json \\
      --limit 300 \\
      --concurrency 16

Loads model config from audit_agentic/config.json (main_agent section). Runs
each row's note + candidate_labels through the main_router prompt, parses the
<answer>...</answer> output, then scores against metadata.labels (= GT).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

from ..agents import LLMConfig, MainAgent, OpenAIChatClient
from ..prompts.loader import default_loader
from ..schemas import AuditInput
from ..structured_output import router_bypass_max_labels as default_router_bypass_max_labels
from .metrics import compute_router_metrics
from .parsing import parse_label_list


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_INPUT = ROOT / "data" / "12_risk_domain.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=ROOT / "traces" / "router_eval.jsonl")
    p.add_argument("--metrics", type=Path, default=ROOT / "traces" / "router_eval_metrics.json")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--limit", type=int, default=0, help="0 = full file")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-max-tokens", type=int, default=512)
    p.add_argument("--shuffle", action="store_true", help="randomly shuffle before --limit")
    p.add_argument("--resume", action="store_true", help="append to existing rows and skip completed row_idx/note_id")
    return p.parse_args()


def load_rows(path: Path, limit: int, shuffle: bool, seed: int) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if shuffle:
        random.Random(seed).shuffle(rows)
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


def _row_key(row: dict) -> str:
    if row.get("row_idx") is not None:
        return f"idx:{row['row_idx']}"
    if row.get("note_id") is not None:
        return f"note:{row['note_id']}"
    return ""


def _load_existing_results(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[eval] skip malformed existing row in {path}", file=sys.stderr)
    return rows


def _rewrite_existing_results(path: Path, rows: list[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _usage_fields(trace: Dict[str, Any]) -> Dict[str, Any]:
    usage = trace.get("usage") or {}
    return {
        "usage": usage,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
    }


def _add_usage_metrics(metrics: Dict[str, Any], rows: list[Dict[str, Any]]) -> None:
    token_keys = ["prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"]
    usage_rows = [r for r in rows if isinstance(r.get("usage"), dict) and r["usage"]]
    summary: Dict[str, Any] = {"rows_with_usage": len(usage_rows)}
    for key in token_keys:
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        summary[f"{key}_sum"] = int(sum(vals)) if vals else 0
        summary[f"{key}_avg"] = (sum(vals) / len(vals)) if vals else 0.0
    metrics["token_usage"] = summary


def process_row(
    row: dict,
    row_idx: int,
    agent: MainAgent,
    image_max_tokens: int = 512,
    bypass_max_labels: int = 8,
) -> Dict[str, Any]:
    audit_input = AuditInput.from_data_row(row)
    out = {
        "row_idx": row_idx,
        "note_id": audit_input.note_id,
        "source": audit_input.source_id,
        "candidate_labels": audit_input.candidate_labels,
        "gt_labels": audit_input.gt_labels or [],
    }
    if len(audit_input.candidate_labels) <= bypass_max_labels:
        out.update({
            "success": True,
            "router_invoked": False,
            "selection_mode": "all_candidates",
            "predicted_labels": list(audit_input.candidate_labels),
            "answer_text": "候选标签数量较少，跳过 Router 并保留全部候选标签。",
            "parse_source": "bypass_all_candidates",
            "non_candidate_labels": [],
            "fuzzy_mappings": [],
            "raw_response": "",
            "latency_ms": 0.0,
            "error": None,
            "usage": {},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        })
        gt_set = set(parse_label_list(out["gt_labels"]))
        out["overlap_correct"] = bool(gt_set & set(out["predicted_labels"]))
        return out
    try:
        router_out, trace = agent.run_router(audit_input, image_max_tokens=image_max_tokens)
        parsed = trace.get("parsed", {}) if isinstance(trace, dict) else {}
        out["success"] = True
        out["router_invoked"] = True
        out["selection_mode"] = "llm_router"
        out["predicted_labels"] = router_out.shortlist_labels
        out["answer_text"] = parsed.get("answer_text", "")
        out["parse_source"] = parsed.get("parse_source", "")
        out["non_candidate_labels"] = parsed.get("non_candidate_labels", [])
        out["fuzzy_mappings"] = parsed.get("fuzzy_mappings", [])
        out["raw_response"] = trace.get("raw_response", "")
        out["latency_ms"] = trace.get("latency_ms")
        out["error"] = trace.get("error")
        out.update(_usage_fields(trace))
        gt_set = set(parse_label_list(out["gt_labels"]))
        pred_set = set(parse_label_list(out["predicted_labels"]))
        out["overlap_correct"] = bool(gt_set & pred_set)
    except Exception as exc:
        out["success"] = False
        out["predicted_labels"] = []
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["overlap_correct"] = False
        out["parse_source"] = "runner_failure"
    return out


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    llm_config = LLMConfig.from_dict(cfg["main_agent"])
    workflow_cfg = cfg.get("workflow") or {}
    os.environ.setdefault(
        "AUDIT_ROUTER_MIN_LABELS",
        str(workflow_cfg.get("router_min_labels", 8)),
    )
    os.environ.setdefault(
        "AUDIT_ROUTER_MAX_LABELS",
        str(workflow_cfg.get("router_max_labels", 12)),
    )
    bypass_max_labels = int(
        workflow_cfg.get(
            "router_bypass_max_labels",
            default_router_bypass_max_labels(),
        )
    )
    loader = default_loader()

    rows = load_rows(args.input, args.limit, args.shuffle, args.seed)
    print(f"[eval] input={args.input}  rows={len(rows)}  concurrency={args.concurrency}  model={llm_config.model}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)

    indexed_rows = list(enumerate(rows))
    existing_results: list[Dict[str, Any]] = _load_existing_results(args.output) if args.resume else []
    if args.resume and args.output.exists():
        _rewrite_existing_results(args.output, existing_results)
    completed = {_row_key(r) for r in existing_results}
    completed.discard("")
    todo = [(i, r) for i, r in indexed_rows if f"idx:{i}" not in completed and f"note:{AuditInput.from_data_row(r).note_id}" not in completed]
    if args.resume and existing_results:
        print(f"[eval] resume: loaded {len(existing_results)} existing rows, todo={len(todo)}", file=sys.stderr)

    results: list[Dict[str, Any]] = list(existing_results)
    t0 = time.time()
    done = 0
    print_every = max(1, len(todo) // 20) if todo else 1

    # Share ONE client across all workers — httpx.Client is thread-safe.
    # Creating a fresh client per worker wastes TCP handshakes; with 128
    # concurrent workers this severely limits actual in-flight requests.
    shared_client = OpenAIChatClient(llm_config)

    def _worker(row_idx: int, row: dict) -> Dict[str, Any]:
        agent = MainAgent(shared_client, loader)
        return process_row(
            row,
            row_idx,
            agent,
            args.image_max_tokens,
            bypass_max_labels,
        )

    mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_worker, i, row): i for i, row in todo}
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                done += 1
                if done % print_every == 0 or done == len(todo):
                    rate = done / max(0.001, time.time() - t0)
                    print(f"  [{done}/{len(todo)}]  {rate:.1f} req/s  elapsed={time.time()-t0:.0f}s", file=sys.stderr)

    metrics = compute_router_metrics(results)
    metrics["model"] = llm_config.model
    metrics["api_base"] = llm_config.api_base
    metrics["temperature"] = llm_config.temperature
    metrics["input_file"] = str(args.input)
    metrics["n_processed"] = len(results)
    metrics["n_existing_loaded"] = len(existing_results)
    metrics["n_new_processed"] = len(todo)
    metrics["wall_seconds"] = round(time.time() - t0, 1)
    _add_usage_metrics(metrics, results)
    metrics["routing"] = {
        "bypass_max_labels": bypass_max_labels,
        "bypass_count": sum(1 for row in results if not row.get("router_invoked", True)),
        "llm_router_count": sum(1 for row in results if row.get("router_invoked")),
    }
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n[eval] wrote rows  -> {args.output}", file=sys.stderr)
    print(f"[eval] wrote stats -> {args.metrics}", file=sys.stderr)
    print()
    # console summary
    headline_keys = [
        "total", "success_rate", "parse_rate",
        "overlap_accuracy",
        "gt_pass_contains_pass_rate", "gt_violation_contains_gt_rate",
        "gt_violation_pred_pass_only_rate",
    ]
    summary = {k: metrics.get(k) for k in headline_keys}
    summary["pred_label_count_avg"] = round(metrics["pred_label_count"]["avg"], 2)
    summary["shortlist_in_range_rate"] = round(metrics["shortlist_size_vs_target"]["in_range_rate"], 3)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("per-source overlap_accuracy:")
    for src, m in metrics["per_source"].items():
        print(f"  {m['total']:>5}  {m['overlap_accuracy']:.3f}  {src}")


if __name__ == "__main__":
    main()
