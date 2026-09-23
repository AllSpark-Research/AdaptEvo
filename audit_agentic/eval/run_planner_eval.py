"""Step 2 + 3: rule retrieval + planner, building on step1_router results.

Reads:
  audit_agentic/traces/step1_router/rows.jsonl  (each: {note_id, source, gt_labels, candidate_labels, predicted_labels, ...})
  audit_agentic/data/cases_100.jsonl            (each: full row with note text)

Writes:
  audit_agentic/traces/step2_planner/rows.jsonl     (per-case planner output + rules used)
  audit_agentic/traces/step2_planner/metrics.json   (aggregate stats)

Concurrency: rule retrieval loads index once (singleton); planner LLM calls run
in a ThreadPoolExecutor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

ROOT = "/path/to/local-assets"
sys.path.insert(0, ROOT)

from audit_agentic.agents import LLMConfig, OpenAIChatClient, PlannerAgent
from audit_agentic.environment import OnlineRuleRetriever, list_tool_briefs
from audit_agentic.environment.rule_retriever import retrieve_rules_with_id_fallback
from audit_agentic.prompts.loader import default_loader
from audit_agentic.schemas import AuditInput


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--router-rows", default=f"{ROOT}/audit_agentic/traces/step1_router/rows.jsonl")
    p.add_argument("--cases", default=f"{ROOT}/audit_agentic/data/cases_100.jsonl")
    p.add_argument("--output-dir", default=f"{ROOT}/audit_agentic/traces/step2_planner")
    p.add_argument("--config", default=f"{ROOT}/audit_agentic/config.json")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--image-max-tokens", type=int, default=512)
    p.add_argument("--resume", action="store_true", help="append to existing rows and skip completed row_idx/note_id")
    return p.parse_args()


def _row_key(row: dict) -> str:
    if row.get("row_idx") is not None:
        return f"idx:{row['row_idx']}"
    if row.get("note_id") is not None:
        return f"note:{row['note_id']}"
    return ""


def _load_existing_results(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[planner] skip malformed existing row in {path}", file=sys.stderr)
    return rows


def _rewrite_existing_results(path: Path, rows: List[dict]) -> None:
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


def _add_usage_metrics(metrics: Dict[str, Any], rows: List[dict]) -> None:
    token_keys = ["prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"]
    usage_rows = [r for r in rows if isinstance(r.get("usage"), dict) and r["usage"]]
    summary: Dict[str, Any] = {"rows_with_usage": len(usage_rows)}
    for key in token_keys:
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        summary[f"{key}_sum"] = int(sum(vals)) if vals else 0
        summary[f"{key}_avg"] = (sum(vals) / len(vals)) if vals else 0.0
    metrics["token_usage"] = summary


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text())
    llm_cfg = LLMConfig.from_dict(cfg["planner_agent"])

    # 1) load cases by note_id
    cases_by_id: Dict[str, dict] = {}
    for line in open(args.cases):
        r = json.loads(line)
        cases_by_id[r["metadata"]["note_id"]] = r

    # 2) load router output
    router_rows: List[dict] = []
    for line in open(args.router_rows):
        router_rows.append(json.loads(line))
    if args.limit:
        router_rows = router_rows[: args.limit]

    print(f"[planner] cases_by_id={len(cases_by_id)}  router_rows={len(router_rows)}  conc={args.concurrency}", file=sys.stderr)

    # 3) one-time rule index load (singleton)
    print("[planner] loading rule index ...", flush=True, file=sys.stderr)
    retriever = OnlineRuleRetriever()
    loader = default_loader()
    shared_client = OpenAIChatClient(llm_cfg)  # shared across workers (thread-safe)
    briefs = list_tool_briefs()
    print(f"[planner] tools available: {[t['name'] for t in briefs]}", file=sys.stderr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    def _process(router_row: dict) -> dict:
        nid = router_row["note_id"]
        row_idx = router_row.get("row_idx")
        case = cases_by_id.get(nid)
        if not case:
            return {"row_idx": row_idx, "note_id": nid, "success": False, "error": "case_not_in_data_file"}
        audit_input = AuditInput.from_data_row(case)
        available_tools = briefs
        if audit_input.available_tools:
            allowed_tools = set(audit_input.available_tools)
            available_tools = [
                tool
                for tool in briefs
                if str(tool.get("name") or tool.get("tool_name") or "")
                in allowed_tools
            ]
        router_shortlist = router_row.get("predicted_labels") or []
        router_shortlist_fallback = not bool(router_shortlist)
        shortlist = router_shortlist or list(audit_input.candidate_labels)
        gt = audit_input.gt_labels or []

        # Step 2: retrieve rules
        try:
            rules = retrieve_rules_with_id_fallback(retriever, audit_input, shortlist)
        except Exception as e:
            return {"row_idx": row_idx, "note_id": nid, "success": False, "error": f"retrieve_failed: {e}"}

        # Step 3: planner (shared client, thread-safe)
        agent = PlannerAgent(shared_client, loader)
        try:
            t0 = time.time()
            out, trace = agent.run_plan(
                audit_input,
                shortlist_labels=shortlist,
                shortlist_rules=rules,
                available_tools=available_tools,
                image_max_tokens=args.image_max_tokens,
            )
            latency_ms = (time.time() - t0) * 1000
            return {
                "row_idx": row_idx,
                "note_id": nid,
                "source": audit_input.source_id,
                "gt_labels": gt,
                "router_shortlist": shortlist,
                "router_shortlist_fallback": router_shortlist_fallback,
                "rules": [{"label": r.label, "chars": len(r.rule_text)} for r in rules],
                "available_tool_names": [
                    str(t.get("name") or t.get("tool_name") or "")
                    for t in available_tools
                ],
                "planner_output": out.model_dump(),
                "raw_response": trace.get("raw_response", ""),
                "audit_experience_enabled": bool(
                    trace.get("audit_experience_enabled", False)
                ),
                "audit_experience_chars": int(
                    trace.get("audit_experience_chars", 0) or 0
                ),
                "latency_ms": latency_ms,
                "success": True,
                "error": trace.get("error"),
                **_usage_fields(trace),
            }
        except Exception as e:
            return {"row_idx": row_idx, "note_id": nid, "success": False, "error": f"planner_failed: {type(e).__name__}: {e}"}

    # 4) run concurrently
    existing_results = _load_existing_results(rows_path) if args.resume else []
    if args.resume and rows_path.exists():
        _rewrite_existing_results(rows_path, existing_results)
    completed = {_row_key(r) for r in existing_results}
    completed.discard("")
    todo = [r for r in router_rows if _row_key(r) not in completed]
    if args.resume and existing_results:
        print(f"[planner] resume: loaded {len(existing_results)} existing rows, todo={len(todo)}", file=sys.stderr)

    results: List[dict] = list(existing_results)
    t0 = time.time()
    done = 0
    print_every = max(1, len(todo) // 20) if todo else 1
    mode = "a" if args.resume and rows_path.exists() else "w"
    with rows_path.open(mode, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = [pool.submit(_process, r) for r in todo]
            for fut in as_completed(futs):
                result = fut.result()
                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                if done % print_every == 0 or done == len(todo):
                    rate = done / max(0.001, time.time() - t0)
                    print(f"  [{done}/{len(todo)}]  {rate:.1f} req/s  elapsed={time.time()-t0:.0f}s", file=sys.stderr)

    # 6) metrics
    n = len(results)
    ok = [r for r in results if r.get("success")]
    metrics = {
        "total": n,
        "success_rate": len(ok) / max(1, n),
        "wall_seconds": round(time.time() - t0, 1),
        "n_existing_loaded": len(existing_results),
        "n_new_processed": len(todo),
        "config": {
            "model": llm_cfg.model,
            "api_base": llm_cfg.api_base,
            "temperature": llm_cfg.temperature,
            "max_tokens": llm_cfg.max_tokens,
            "audit_experience_mode": os.environ.get(
                "AUDIT_PLANNER_EXPERIENCE", ""
            ),
        },
    }
    _add_usage_metrics(metrics, results)
    metrics["planner_experience_rows"] = sum(
        1
        for r in ok
        if r.get("audit_experience_enabled")
    )
    if ok:
        pl_counts = [len(r["planner_output"]["possible_labels"]) for r in ok]
        tr_counts = [len(r["planner_output"]["tool_required_labels"]) for r in ok]
        tc_counts = [len(r["planner_output"]["tool_calls"]) for r in ok]
        shortlist_sizes = [len(r["router_shortlist"]) for r in ok]
        kept_sizes = [pl + tr for pl, tr in zip(pl_counts, tr_counts)]
        rule_out_counts = [sl - kept for sl, kept in zip(shortlist_sizes, kept_sizes)]

        # GT survival after planner filter (gt label still in possible/tool_required)
        gt_survived = 0
        gt_violation_total = 0
        gt_violation_survived = 0
        for r in ok:
            gt = set(r["gt_labels"] or [])
            pl = {x["label"] for x in r["planner_output"]["possible_labels"]}
            tr = {x["label"] for x in r["planner_output"]["tool_required_labels"]}
            kept = pl | tr
            if gt & kept:
                gt_survived += 1
            if gt and gt != {"通过"}:
                gt_violation_total += 1
                if gt & kept:
                    gt_violation_survived += 1

        # opinion distribution
        opinions = Counter()
        for r in ok:
            for x in r["planner_output"]["possible_labels"]:
                opinions[("possible", x.get("preliminary_opinion", "uncertain"))] += 1
            for x in r["planner_output"]["tool_required_labels"]:
                opinions[("tool_required", x.get("preliminary_opinion", "uncertain"))] += 1

        # tool calls
        tool_call_dist = Counter()
        for r in ok:
            for tc in r["planner_output"]["tool_calls"]:
                tool_call_dist[tc["tool_name"]] += 1

        rows_calling_any_tool = sum(1 for r in ok if r["planner_output"]["tool_calls"])

        # per-source
        per_source = defaultdict(lambda: {"total": 0, "gt_survived": 0, "rule_out_avg": 0.0})
        for r in ok:
            s = r["source"]
            per_source[s]["total"] += 1
            gt = set(r["gt_labels"] or [])
            pl = {x["label"] for x in r["planner_output"]["possible_labels"]}
            tr = {x["label"] for x in r["planner_output"]["tool_required_labels"]}
            if gt & (pl | tr):
                per_source[s]["gt_survived"] += 1
            per_source[s]["rule_out_avg"] += len(r["router_shortlist"]) - len(pl | tr)
        per_source_pretty = {
            s: {
                "total": v["total"],
                "gt_survival_rate": v["gt_survived"] / max(1, v["total"]),
                "rule_out_avg": v["rule_out_avg"] / max(1, v["total"]),
            }
            for s, v in sorted(per_source.items(), key=lambda kv: -kv[1]["total"])
        }

        metrics.update({
            "shortlist_size_avg": sum(shortlist_sizes) / len(shortlist_sizes),
            "possible_labels_avg": sum(pl_counts) / len(pl_counts),
            "tool_required_labels_avg": sum(tr_counts) / len(tr_counts),
            "rule_out_avg": sum(rule_out_counts) / len(rule_out_counts),
            "kept_after_planner_avg": sum(kept_sizes) / len(kept_sizes),
            "tool_calls_avg": sum(tc_counts) / len(tc_counts),
            "rows_calling_any_tool_rate": rows_calling_any_tool / len(ok),
            "gt_survival_rate": gt_survived / len(ok),
            "gt_violation_total": gt_violation_total,
            "gt_violation_survival_rate": gt_violation_survived / max(1, gt_violation_total),
            "tool_call_distribution": dict(tool_call_dist),
            "preliminary_opinion_distribution": {f"{k[0]}.{k[1]}": v for k, v in opinions.items()},
            "per_source": per_source_pretty,
        })

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    # console summary
    print(f"\n[planner] wrote rows  -> {rows_path}", file=sys.stderr)
    print(f"[planner] wrote stats -> {metrics_path}", file=sys.stderr)
    print()
    headline = {k: metrics[k] for k in [
        "total", "success_rate", "wall_seconds",
        "shortlist_size_avg", "rule_out_avg", "kept_after_planner_avg",
        "possible_labels_avg", "tool_required_labels_avg",
        "tool_calls_avg", "rows_calling_any_tool_rate",
        "gt_survival_rate", "gt_violation_survival_rate",
    ] if k in metrics}
    print(json.dumps(headline, ensure_ascii=False, indent=2))
    print()
    print("tool call distribution:", json.dumps(metrics.get("tool_call_distribution", {}), ensure_ascii=False))
    print("opinion distribution:", json.dumps(metrics.get("preliminary_opinion_distribution", {}), ensure_ascii=False))
    print()
    print("per-source:")
    for s, m in metrics.get("per_source", {}).items():
        print(f"  {m['total']:>3}  gt_survival={m['gt_survival_rate']:.3f}  rule_out_avg={m['rule_out_avg']:.1f}  {s}")


if __name__ == "__main__":
    main()
