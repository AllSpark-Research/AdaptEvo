"""Step 4: main_final (continuation) on all 100 cases.

Reads:
  audit_agentic/data/cases_100.jsonl          - for note text / candidate_labels
  audit_agentic/traces/step1_router/rows.jsonl - for router raw_response (assistant turn)
  audit_agentic/traces/step3_tools/rows.jsonl  - for planner output + tool observations

For each case:
  1. AuditInput.from_data_row(row)
  2. Re-render router prompt (deterministic given the same template + inputs)
  3. Build router_trace = {"rendered_prompt": re-rendered, "raw_response": from step1}
  4. ZeusRuleRetriever.retrieve(source, router_shortlist) -> rules
  5. Keep only rules for labels surviving planner filter
  6. main_agent.run_final(router_trace, possible, tool_required, remaining_rules, observations)

Writes:
  audit_agentic/traces/step4_main_final/rows.jsonl
  audit_agentic/traces/step4_main_final/metrics.json
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
from typing import Dict, List

ROOT = "/path/to/local-assets"
sys.path.insert(0, ROOT)

from audit_agentic.agents import LLMConfig, MainAgent, OpenAIChatClient
from audit_agentic.environment import OnlineRuleRetriever
from audit_agentic.environment.rule_retriever import QUEUE_NOTICE_LABEL, retrieve_rules_with_id_fallback
from audit_agentic.prompts.loader import default_loader
from audit_agentic.schemas import AuditInput, FilteredLabel, ToolObservation
from audit_agentic.structured_output import build_final_candidate_labels


PASS_LABEL = "通过"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=f"{ROOT}/audit_agentic/data/cases_100.jsonl")
    p.add_argument("--router-rows", default=f"{ROOT}/audit_agentic/traces/step1_router/rows.jsonl")
    p.add_argument("--step3-rows", default=f"{ROOT}/audit_agentic/traces/step3_tools/rows.jsonl")
    p.add_argument("--output-dir", default=f"{ROOT}/audit_agentic/traces/step4_main_final")
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
                print(f"[step4] skip malformed existing row in {path}", file=sys.stderr)
    return rows


def _rewrite_existing_results(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _usage_fields(trace: Dict) -> Dict:
    usage = trace.get("usage") or {}
    return {
        "usage": usage,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
    }


def _add_usage_metrics(metrics: Dict, rows: List[dict]) -> None:
    token_keys = ["prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"]
    usage_rows = [r for r in rows if isinstance(r.get("usage"), dict) and r["usage"]]
    summary: Dict[str, object] = {"rows_with_usage": len(usage_rows)}
    for key in token_keys:
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        summary[f"{key}_sum"] = int(sum(vals)) if vals else 0
        summary[f"{key}_avg"] = (sum(vals) / len(vals)) if vals else 0.0
    metrics["token_usage"] = summary


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text())
    llm_cfg = LLMConfig.from_dict(cfg["main_agent"])
    loader = default_loader()
    shared_client = OpenAIChatClient(llm_cfg)  # shared across workers (thread-safe)

    case_rows = [json.loads(l) for l in open(args.cases)]
    router_rows = [json.loads(l) for l in open(args.router_rows)]
    cases_by_idx = {idx: row for idx, row in enumerate(case_rows)}
    cases_by_id = {row["metadata"]["note_id"]: row for row in case_rows}
    router_by_idx = {
        row["row_idx"]: row for row in router_rows if row.get("row_idx") is not None
    }
    router_by_id = {row["note_id"]: row for row in router_rows}
    step3_rows = [json.loads(l) for l in open(args.step3_rows)]
    if args.limit:
        step3_rows = step3_rows[: args.limit]
    print(
        f"[step4] cases={len(case_rows)} router={len(router_rows)} "
        f"step3={len(step3_rows)}  conc={args.concurrency}",
        file=sys.stderr,
    )

    print("[step4] loading rule index ...", flush=True, file=sys.stderr)
    retriever = OnlineRuleRetriever()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    def _process(row: dict) -> dict:
        nid = row["note_id"]
        row_idx = row.get("row_idx")
        case = cases_by_idx.get(row_idx) if row_idx is not None else None
        router = router_by_idx.get(row_idx) if row_idx is not None else None
        # Source-filtered eval files are commonly re-indexed from zero. In
        # that case row_idx refers to the filtered file, not the original
        # all-source JSONL. Never accept a positional match for another note.
        if case is not None:
            case_note_id = (case.get("metadata") or {}).get("note_id")
            if case_note_id != nid:
                case = None
        if router is not None and router.get("note_id") != nid:
            router = None
        if case is None:
            case = cases_by_id.get(nid)
        if router is None:
            router = router_by_id.get(nid)
        if not case or not router:
            return {"row_idx": row_idx, "note_id": nid, "success": False, "error": "missing_case_or_router"}
        ai = AuditInput.from_data_row(case)
        shortlist = row.get("router_shortlist") or []
        planner_out = row.get("planner_output") or {}
        observations = row.get("tool_observations") or []
        possible = planner_out.get("possible_labels") or []
        tool_req = planner_out.get("tool_required_labels") or []
        router_invoked = bool(router.get("router_invoked", True))

        # rules
        try:
            rules = retrieve_rules_with_id_fallback(retriever, ai, shortlist)
        except Exception as e:
            return {"row_idx": row_idx, "note_id": nid, "success": False, "error": f"retrieve_failed: {e}"}
        final_candidate_labels = build_final_candidate_labels(
            ai.candidate_labels,
            shortlist,
            possible,
            tool_req,
        )
        surviving = set(final_candidate_labels)
        if os.environ.get("AUDIT_FINAL_KEEP_ALL_ROUTER_RULES", "").strip().lower() in {"1", "true", "yes", "on"}:
            remaining_rules = rules
        else:
            remaining_rules = [r for r in rules if r.label in surviving or r.label == QUEUE_NOTICE_LABEL]

        # router_trace: re-render prompt (deterministic) + use saved raw response
        agent = MainAgent(shared_client, loader)
        router_prompt = loader.render(
            os.environ.get("AUDIT_MAIN_ROUTER_TEMPLATE", "main_router"),
            note=ai.note,
            candidate_labels=ai.candidate_labels,
        )
        router_trace = {
            "rendered_prompt": router_prompt,
            "raw_response": router.get("raw_response", ""),
        }

        try:
            t0 = time.time()
            final_out, trace = agent.run_final(
                ai,
                router_trace=router_trace,
                possible_labels=possible,
                tool_required_labels=tool_req,
                remaining_rules=remaining_rules,
                tool_observations=observations,
                shortlist_labels=shortlist,
                final_candidate_labels=final_candidate_labels,
                standalone=not router_invoked,
                image_max_tokens=args.image_max_tokens,
            )
            latency_ms = (time.time() - t0) * 1000
            trace_error = trace.get("error")
            return {
                "row_idx": row_idx,
                "note_id": nid,
                "source": ai.source_id,
                "gt_labels": ai.gt_labels or [],
                "router_shortlist": shortlist,
                "router_invoked": router_invoked,
                "selection_mode": router.get("selection_mode", "llm_router"),
                "final_candidate_labels": final_candidate_labels,
                "possible_labels": [x["label"] for x in possible],
                "tool_required_labels": [x["label"] for x in tool_req],
                "tools_used_count": len(observations),
                "predict_label": final_out.predict_label,
                "binary_decision": final_out.binary_decision,
                "audit_trace_text": final_out.audit_trace,
                "raw_response": trace.get("raw_response", ""),
                "template_name": trace.get("template_name"),
                "audit_experience_enabled": bool(
                    trace.get("audit_experience_enabled", False)
                ),
                "audit_experience_chars": int(
                    trace.get("audit_experience_chars", 0) or 0
                ),
                "latency_ms": latency_ms,
                "success": not bool(trace_error),
                "error": trace_error,
                **_usage_fields(trace),
            }
        except Exception as e:
            return {"row_idx": row_idx, "note_id": nid, "success": False, "error": f"final_failed: {type(e).__name__}: {e}"}

    existing_results_all = _load_existing_results(rows_path) if args.resume else []
    existing_results = [
        r for r in existing_results_all if r.get("success") and not r.get("error")
    ]
    if args.resume and rows_path.exists():
        _rewrite_existing_results(rows_path, existing_results)
    completed = {_row_key(r) for r in existing_results}
    completed.discard("")
    todo_rows = [r for r in step3_rows if _row_key(r) not in completed]
    if args.resume and existing_results_all:
        dropped = len(existing_results_all) - len(existing_results)
        print(
            f"[step4] resume: loaded {len(existing_results_all)} existing rows, "
            f"kept={len(existing_results)}, dropped_failed={dropped}, todo={len(todo_rows)}",
            file=sys.stderr,
        )

    results = list(existing_results)
    t0 = time.time()
    done = 0
    print_every = max(1, len(todo_rows) // 20) if todo_rows else 1
    mode = "a" if args.resume and rows_path.exists() else "w"
    with rows_path.open(mode, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = [pool.submit(_process, r) for r in todo_rows]
            for fut in as_completed(futs):
                result = fut.result()
                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                if done % print_every == 0 or done == len(todo_rows):
                    rate = done / max(0.001, time.time() - t0)
                    print(f"  [{done}/{len(todo_rows)}]  {rate:.1f} req/s  elapsed={time.time()-t0:.0f}s", file=sys.stderr)

    # ── metrics ──
    # nOA:         预测标签集 == GT标签集 (严格完全一致)
    # wOA:         二分类正确 (通过/违规方向一致即可)
    # overlap_nOA: 预测标签中任一命中GT中任一即算对
    ok = [r for r in results if r.get("success")]
    n = len(results)

    def gt_kind(r):
        gt = set(r["gt_labels"])
        return "pass" if gt == {PASS_LABEL} else "violation"

    def pred_kind(r):
        pred = set(r["predict_label"])
        return "pass" if pred == {PASS_LABEL} or (PASS_LABEL in pred and len(pred) == 1) else "violation"

    def safe(num, den):
        return num / den if den else 0.0

    nOA_count = 0
    wOA_count = 0
    overlap_nOA_count = 0
    tp = tn = fp = fn = 0

    for r in ok:
        gt = set(r["gt_labels"])
        pred = set(r["predict_label"])
        gk = gt_kind(r)
        pk = pred_kind(r)

        # nOA: exact match
        if pred == gt:
            nOA_count += 1

        # wOA: binary direction correct (pass vs violation)
        if gk == pk:
            wOA_count += 1

        # overlap_nOA: any predicted label hits any GT label
        if pred & gt:
            overlap_nOA_count += 1

        # confusion matrix
        if gk == "violation" and pk == "violation":
            tp += 1
        elif gk == "pass" and pk == "pass":
            tn += 1
        elif gk == "pass" and pk == "violation":
            fp += 1
        elif gk == "violation" and pk == "pass":
            fn += 1

    metrics = {
        "total": n,
        "success_rate": safe(len(ok), n),
        "wall_seconds": round(time.time() - t0, 1),
        "n_existing_loaded": len(existing_results),
        "n_new_processed": len(todo_rows),
        "empty_raw_count": sum(1 for r in results if not (r.get("raw_response") or "")),
        "empty_predict_count": sum(1 for r in results if not (r.get("predict_label") or [])),
        "trace_error_count": sum(1 for r in results if r.get("error")),
        "trace_error_distribution": dict(Counter(str(r.get("error"))[:200] for r in results if r.get("error"))),
        "nOA": safe(nOA_count, len(ok)),
        "wOA": safe(wOA_count, len(ok)),
        "overlap_nOA": safe(overlap_nOA_count, len(ok)),
        "confusion": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "violation_recall": safe(tp, tp + fn),
        "violation_precision": safe(tp, tp + fp),
        "pass_recall": safe(tn, tn + fp),
        "pass_precision": safe(tn, tn + fn),
        "balanced_accuracy": (safe(tp, tp + fn) + safe(tn, tn + fp)) / 2,
        "config": {
            "model": llm_cfg.model,
            "temperature": llm_cfg.temperature,
            "max_tokens": llm_cfg.max_tokens,
        },
    }
    _add_usage_metrics(metrics, results)

    # per-source
    per_source = defaultdict(lambda: {"total": 0, "nOA": 0, "wOA": 0, "overlap_nOA": 0,
                                       "viol_tp": 0, "viol_fn": 0, "pass_tn": 0, "pass_fp": 0})
    for r in ok:
        s = r["source"]
        ps = per_source[s]
        ps["total"] += 1
        gt = set(r["gt_labels"])
        pred = set(r["predict_label"])
        gk = gt_kind(r)
        pk = pred_kind(r)
        if pred == gt:
            ps["nOA"] += 1
        if gk == pk:
            ps["wOA"] += 1
        if pred & gt:
            ps["overlap_nOA"] += 1
        if gk == "violation":
            ps["viol_tp" if pk == "violation" else "viol_fn"] += 1
        else:
            ps["pass_tn" if pk == "pass" else "pass_fp"] += 1

    metrics["per_source"] = {
        s: {
            "total": v["total"],
            "nOA": safe(v["nOA"], v["total"]),
            "wOA": safe(v["wOA"], v["total"]),
            "overlap_nOA": safe(v["overlap_nOA"], v["total"]),
            "violation_recall": safe(v["viol_tp"], v["viol_tp"] + v["viol_fn"]),
            "pass_recall": safe(v["pass_tn"], v["pass_tn"] + v["pass_fp"]),
        }
        for s, v in sorted(per_source.items(), key=lambda kv: -kv[1]["total"])
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    print(f"\n[step4] wrote -> {rows_path}", file=sys.stderr)
    print(f"[step4] wrote -> {metrics_path}", file=sys.stderr)
    print()
    print(json.dumps({k: metrics[k] for k in [
        "total", "success_rate", "wall_seconds",
        "nOA", "wOA", "overlap_nOA",
        "violation_recall", "violation_precision",
        "pass_recall", "pass_precision",
        "balanced_accuracy", "confusion",
    ]}, ensure_ascii=False, indent=2))
    print()
    print("per-source:")
    print(f"  {'n':>3}  {'nOA':>5}  {'wOA':>5}  {'ovlp_nOA':>9}  {'viol_R':>6}  {'pass_R':>6}  source")
    for s, m in metrics["per_source"].items():
        print(f"  {m['total']:>3}  {m['nOA']:>5.3f}  {m['wOA']:>5.3f}  {m['overlap_nOA']:>9.3f}  "
              f"{m['violation_recall']:>6.3f}  {m['pass_recall']:>6.3f}  {s}")


if __name__ == "__main__":
    main()
