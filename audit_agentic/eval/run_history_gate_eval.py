"""Step 5: force a Train-history boundary check after Four-stage Main Final.

This runner deliberately keeps history outside the maintained Four-stage
decision itself.  It consumes the saved Step 3/4 artifacts, retrieves balanced
same-source Train precedents, asks a History Analyst to compare the boundary,
then asks an independent Gate Reviewer to keep or revise the Step 4 decision.

History failures are fail-open with respect to the existing model decision:
the row keeps its Step 4 prediction and records the gate error.  This makes a
service timeout observable without silently turning it into an audit change.
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
from typing import Any, Dict, Iterable, List


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from audit_agentic.agents import LLMConfig, OpenAIChatClient
from audit_agentic.agents.base import BaseAgent
from audit_agentic.agents.multimodal import collect_rule_images, make_multimodal_message
from audit_agentic.environment import OnlineRuleRetriever
from audit_agentic.environment.rule_retriever import retrieve_rules_with_id_fallback
from audit_agentic.environment.tool_registry import DEFAULT_TOOL_REGISTRY
from audit_agentic.environment.tool_render import render_observations
from audit_agentic.environment.tools.compare_historical_audit_cases import (
    HISTORY_CONTEXT_FULL_MULTIMODAL,
    HISTORY_CONTEXT_LEGACY_COMPACT,
    HistoricalAuditCasesTool,
)
from audit_agentic.prompts.loader import default_loader
from audit_agentic.schemas import AuditInput, RuleInfo
from audit_agentic.structured_output import json_schema_response_format


PASS_LABEL = "通过"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Force a Train-history comparison gate after Four-stage Step 4."
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--step3-rows", required=True)
    parser.add_argument("--step4-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(AUDIT_ROOT / "config.json"))
    parser.add_argument(
        "--history-data",
        default=str(AUDIT_ROOT / "data" / "train_local.jsonl"),
    )
    parser.add_argument(
        "--history-context-mode",
        choices=[HISTORY_CONTEXT_LEGACY_COMPACT, HISTORY_CONTEXT_FULL_MULTIMODAL],
        default=HISTORY_CONTEXT_LEGACY_COMPACT,
    )
    parser.add_argument("--history-cases-per-class", type=int, default=3)
    parser.add_argument("--history-image-max-tokens", type=int, default=448)
    parser.add_argument("--image-max-tokens", type=int, default=448)
    parser.add_argument("--gate-temperature", type=float)
    parser.add_argument("--gate-max-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def row_key(row: dict) -> str:
    if row.get("row_idx") is not None:
        return f"idx:{row['row_idx']}"
    if row.get("note_id"):
        return f"note:{row['note_id']}"
    return ""


def build_lookup(rows: Iterable[dict]) -> tuple[Dict[int, dict], Dict[str, dict]]:
    by_idx = {
        int(row["row_idx"]): row
        for row in rows
        if row.get("row_idx") is not None
    }
    by_id = {
        str(row["note_id"]): row
        for row in rows
        if row.get("note_id")
    }
    return by_idx, by_id


def matching_row(
    row_idx: int,
    note_id: str,
    by_idx: Dict[int, dict],
    by_id: Dict[str, dict],
) -> dict | None:
    row = by_idx.get(row_idx)
    if row is not None and str(row.get("note_id") or "") != note_id:
        row = None
    return row or by_id.get(note_id)


def dedupe(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        label = str(value or "").strip()
        if label and label not in seen:
            seen.add(label)
            result.append(label)
    return result


def binary_kind(labels: Iterable[str]) -> str:
    values = set(labels)
    return "pass" if values == {PASS_LABEL} else "violation"


def gate_response_format(candidate_labels: List[str]) -> Dict[str, Any]:
    return json_schema_response_format(
        "audit_history_gate_output",
        {
            "type": "object",
            "properties": {
                "gate_action": {
                    "type": "string",
                    "enum": ["keep", "revise"],
                },
                "predict_label": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {"type": "string", "enum": candidate_labels},
                },
                "history_applicability": {
                    "type": "string",
                    "enum": ["usable", "weak", "insufficient"],
                },
                "history_effect": {
                    "type": "string",
                    "enum": [
                        "supports_initial",
                        "supports_revision",
                        "conflicting",
                        "neutral",
                    ],
                },
                "decisive_evidence": {
                    "type": "array",
                    "maxItems": 6,
                    "items": {"type": "string"},
                },
                "audit_trace": {
                    "type": "string",
                    "description": "说明保留或改判所依赖的当前证据、规则边界及历史类比。",
                },
            },
            "required": [
                "gate_action",
                "predict_label",
                "history_applicability",
                "history_effect",
                "decisive_evidence",
                "audit_trace",
            ],
            "additionalProperties": False,
        },
    )


def rule_payload(rules: List[RuleInfo]) -> List[dict]:
    return [
        {
            "label": rule.label,
            "rule_text": rule.rule_text,
            "positive_conditions": rule.positive_conditions,
            "exemption_conditions": rule.exemption_conditions,
            "examples": rule.examples,
            "hard_negatives": rule.hard_negatives,
            "required_tools": rule.required_tools,
        }
        for rule in rules
    ]


def choose_gate_labels(ai: AuditInput, initial: dict) -> List[str]:
    original = set(ai.candidate_labels)
    labels = dedupe(
        list(initial.get("router_shortlist") or [])
        + list(initial.get("final_candidate_labels") or [])
        + list(initial.get("predict_label") or [])
    )
    labels = [label for label in labels if label in original]
    if PASS_LABEL in original and PASS_LABEL not in labels:
        labels.append(PASS_LABEL)
    return labels or list(ai.candidate_labels)


def choose_focus_labels(initial: dict, gate_labels: List[str]) -> List[str]:
    values = dedupe(
        list(initial.get("predict_label") or [])
        + list(initial.get("possible_labels") or [])
        + list(initial.get("tool_required_labels") or [])
        + gate_labels
    )
    return [label for label in values if label != PASS_LABEL][:3]


def usage_fields(trace: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    usage = trace.get("usage") or {}
    return {
        f"{prefix}_usage": usage,
        f"{prefix}_prompt_tokens": usage.get("prompt_tokens"),
        f"{prefix}_completion_tokens": usage.get("completion_tokens"),
        f"{prefix}_total_tokens": usage.get("total_tokens"),
        f"{prefix}_reasoning_tokens": usage.get("reasoning_tokens"),
    }


def safe_ratio(num: int | float, den: int | float) -> float:
    return num / den if den else 0.0


def metric_snapshot(rows: List[dict], prediction_key: str) -> Dict[str, Any]:
    usable = [
        row
        for row in rows
        if row.get("gt_labels") and row.get(prediction_key)
    ]
    exact = binary = overlap = 0
    tp = tn = fp = fn = 0
    for row in usable:
        gt = set(row["gt_labels"])
        pred = set(row[prediction_key])
        gt_kind = binary_kind(gt)
        pred_kind = binary_kind(pred)
        exact += int(pred == gt)
        binary += int(gt_kind == pred_kind)
        overlap += int(bool(pred & gt))
        if gt_kind == "violation" and pred_kind == "violation":
            tp += 1
        elif gt_kind == "pass" and pred_kind == "pass":
            tn += 1
        elif gt_kind == "pass" and pred_kind == "violation":
            fp += 1
        else:
            fn += 1
    total = len(usable)
    return {
        "total": total,
        "nOA": safe_ratio(exact, total),
        "wOA": safe_ratio(binary, total),
        "overlap_nOA": safe_ratio(overlap, total),
        "confusion": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "violation_recall": safe_ratio(tp, tp + fn),
        "violation_precision": safe_ratio(tp, tp + fp),
        "pass_recall": safe_ratio(tn, tn + fp),
        "pass_precision": safe_ratio(tn, tn + fn),
        "balanced_accuracy": (
            safe_ratio(tp, tp + fn) + safe_ratio(tn, tn + fp)
        )
        / 2,
    }


def per_source_metrics(rows: List[dict]) -> Dict[str, Any]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source") or "")].append(row)
    output: Dict[str, Any] = {}
    for source, source_rows in sorted(grouped.items()):
        initial = metric_snapshot(source_rows, "initial_predict_label")
        gate = metric_snapshot(source_rows, "predict_label")
        output[source] = {
            "total": len(source_rows),
            "initial": initial,
            "history_gate": gate,
            "delta_nOA": gate["nOA"] - initial["nOA"],
            "delta_wOA": gate["wOA"] - initial["wOA"],
            "delta_violation_recall": (
                gate["violation_recall"] - initial["violation_recall"]
            ),
            "delta_pass_recall": gate["pass_recall"] - initial["pass_recall"],
        }
    return output


def build_metrics(rows: List[dict], wall_seconds: float, config: dict) -> Dict[str, Any]:
    initial = metric_snapshot(rows, "initial_predict_label")
    gate = metric_snapshot(rows, "predict_label")
    exact_changed = binary_changed = 0
    exact_fix = exact_harm = binary_fix = binary_harm = 0
    for row in rows:
        gt = set(row.get("gt_labels") or [])
        initial_pred = set(row.get("initial_predict_label") or [])
        gate_pred = set(row.get("predict_label") or [])
        if not gt or not initial_pred or not gate_pred:
            continue
        if initial_pred != gate_pred:
            exact_changed += 1
        if binary_kind(initial_pred) != binary_kind(gate_pred):
            binary_changed += 1
        initial_exact = initial_pred == gt
        gate_exact = gate_pred == gt
        exact_fix += int(not initial_exact and gate_exact)
        exact_harm += int(initial_exact and not gate_exact)
        initial_binary = binary_kind(initial_pred) == binary_kind(gt)
        gate_binary = binary_kind(gate_pred) == binary_kind(gt)
        binary_fix += int(not initial_binary and gate_binary)
        binary_harm += int(initial_binary and not gate_binary)

    token_keys = ["prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"]
    token_usage: Dict[str, Any] = {}
    for prefix in ("history", "gate"):
        for key in token_keys:
            values = [
                row.get(f"{prefix}_{key}")
                for row in rows
                if isinstance(row.get(f"{prefix}_{key}"), (int, float))
            ]
            token_usage[f"{prefix}_{key}_sum"] = int(sum(values)) if values else 0
            token_usage[f"{prefix}_{key}_avg"] = safe_ratio(sum(values), len(values))

    return {
        "total": len(rows),
        "wall_seconds": round(wall_seconds, 1),
        "gate_success_rate": safe_ratio(
            sum(1 for row in rows if row.get("gate_success")), len(rows)
        ),
        "fallback_to_initial_count": sum(
            1 for row in rows if row.get("fallback_to_initial")
        ),
        "history_applicability": dict(
            Counter(str(row.get("history_applicability") or "unknown") for row in rows)
        ),
        "gate_actions": dict(
            Counter(str(row.get("gate_action") or "unknown") for row in rows)
        ),
        "error_distribution": dict(
            Counter(str(row.get("error"))[:240] for row in rows if row.get("error"))
        ),
        "initial": initial,
        "history_gate": gate,
        "delta": {
            "nOA": gate["nOA"] - initial["nOA"],
            "wOA": gate["wOA"] - initial["wOA"],
            "violation_recall": gate["violation_recall"] - initial["violation_recall"],
            "pass_recall": gate["pass_recall"] - initial["pass_recall"],
            "balanced_accuracy": gate["balanced_accuracy"] - initial["balanced_accuracy"],
        },
        "changes": {
            "exact_changed": exact_changed,
            "binary_changed": binary_changed,
            "exact_fix": exact_fix,
            "exact_harm": exact_harm,
            "binary_fix": binary_fix,
            "binary_harm": binary_harm,
        },
        "token_usage": token_usage,
        "config": config,
        "per_source": per_source_metrics(rows),
    }


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("ERROR: --limit must be >= 0")
    if args.concurrency < 1:
        raise SystemExit("ERROR: --concurrency must be >= 1")
    if args.gate_max_tokens < 1:
        raise SystemExit("ERROR: --gate-max-tokens must be >= 1")
    if not 1 <= args.history_cases_per_class <= 4:
        raise SystemExit("ERROR: --history-cases-per-class must be in [1, 4]")

    cases_path = Path(args.cases).expanduser().resolve()
    step3_path = Path(args.step3_rows).expanduser().resolve()
    step4_path = Path(args.step4_rows).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    history_path = Path(args.history_data).expanduser().resolve()
    for name, path in (
        ("cases", cases_path),
        ("step3 rows", step3_path),
        ("step4 rows", step4_path),
        ("config", config_path),
        ("history data", history_path),
    ):
        if not path.is_file():
            raise SystemExit(f"ERROR: {name} not found: {path}")

    os.environ["AUDIT_HISTORY_DATA_PATH"] = str(history_path)
    os.environ["AUDIT_HISTORY_CONTEXT_MODE"] = args.history_context_mode
    os.environ["AUDIT_HISTORY_CASES_PER_CLASS"] = str(args.history_cases_per_class)
    os.environ["AUDIT_HISTORY_IMAGE_MAX_TOKENS"] = str(args.history_image_max_tokens)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    llm_config = LLMConfig.from_dict(config["main_agent"])
    gate_temperature = (
        llm_config.temperature if args.gate_temperature is None else args.gate_temperature
    )
    # The History Analyst sets its own deterministic overrides. These defaults
    # therefore apply only to the independent Gate Reviewer call.
    llm_config.temperature = gate_temperature
    llm_config.max_tokens = args.gate_max_tokens
    client = OpenAIChatClient(llm_config)
    history_tool = HistoricalAuditCasesTool(client)
    gate_agent = BaseAgent(client, default_loader(), "history_gate")
    retriever = OnlineRuleRetriever()

    case_rows = read_jsonl(cases_path)
    if args.limit:
        case_rows = case_rows[: args.limit]
    step3_rows = read_jsonl(step3_path)
    step4_rows = read_jsonl(step4_path)
    step3_by_idx, step3_by_id = build_lookup(step3_rows)
    step4_by_idx, step4_by_id = build_lookup(step4_rows)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    metrics_path = output_dir / "metrics.json"
    run_config_path = output_dir / "run_config.json"

    run_config = {
        "cases": str(cases_path),
        "step3_rows": str(step3_path),
        "step4_rows": str(step4_path),
        "config": str(config_path),
        "model": llm_config.model,
        "gate_temperature": gate_temperature,
        "gate_max_tokens": args.gate_max_tokens,
        "history_data": str(history_path),
        "history_context_mode": args.history_context_mode,
        "history_cases_per_class": args.history_cases_per_class,
        "history_image_max_tokens": args.history_image_max_tokens,
        "image_max_tokens": args.image_max_tokens,
        "concurrency": args.concurrency,
        "limit": args.limit,
        "resume": args.resume,
    }
    run_config_path.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    work_items: List[tuple[int, dict, dict | None, dict | None]] = []
    for row_idx, case in enumerate(case_rows):
        ai = AuditInput.from_data_row(case)
        note_id = str(ai.note_id or "")
        initial = matching_row(row_idx, note_id, step4_by_idx, step4_by_id)
        step3 = matching_row(row_idx, note_id, step3_by_idx, step3_by_id)
        work_items.append((row_idx, case, initial, step3))

    existing_all = read_jsonl(rows_path) if args.resume and rows_path.exists() else []
    existing = [row for row in existing_all if row.get("gate_success")]
    completed = {row_key(row) for row in existing}
    completed.discard("")
    todo = [item for item in work_items if f"idx:{item[0]}" not in completed]
    if args.resume and rows_path.exists():
        with rows_path.open("w", encoding="utf-8") as handle:
            for row in existing:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"[history-gate] cases={len(case_rows)} step3={len(step3_rows)} "
        f"step4={len(step4_rows)} existing={len(existing)} todo={len(todo)} "
        f"context={args.history_context_mode} concurrency={args.concurrency}",
        file=sys.stderr,
    )

    def process(item: tuple[int, dict, dict | None, dict | None]) -> dict:
        row_idx, case, initial, step3 = item
        ai = AuditInput.from_data_row(case)
        note_id = str(ai.note_id or "")
        gt_labels = list(ai.gt_labels or [])
        initial_labels = dedupe((initial or {}).get("predict_label") or [])

        base_result: Dict[str, Any] = {
            "row_idx": row_idx,
            "note_id": note_id,
            "source": ai.source_id,
            "gt_labels": gt_labels,
            "initial_success": bool((initial or {}).get("success")),
            "initial_predict_label": initial_labels,
            "initial_binary_decision": binary_kind(initial_labels) if initial_labels else "unknown",
            "initial_audit_trace": str((initial or {}).get("audit_trace_text") or ""),
            "initial_final_candidate_labels": list(
                (initial or {}).get("final_candidate_labels") or []
            ),
            "history_call_forced": True,
            "history_context_mode": args.history_context_mode,
        }
        if initial is None:
            return {
                **base_result,
                "success": False,
                "gate_success": False,
                "fallback_to_initial": False,
                "predict_label": [],
                "error": "missing_step4_result",
            }
        if not initial_labels:
            return {
                **base_result,
                "success": False,
                "gate_success": False,
                "fallback_to_initial": False,
                "predict_label": [],
                "error": f"invalid_step4_result: {initial.get('error') or 'empty prediction'}",
            }

        gate_labels = choose_gate_labels(ai, initial)
        focus_labels = choose_focus_labels(initial, gate_labels)
        observations = list((step3 or {}).get("tool_observations") or [])
        try:
            rules = retrieve_rules_with_id_fallback(retriever, ai, gate_labels)
            rules_data = rule_payload(rules)
            context = {
                "source_id": ai.source_id,
                "note_id": note_id,
                "note": ai.note,
                "images": list(ai.images or []),
                "candidate_labels": gate_labels,
                "rules": rules_data,
                "observations": observations,
            }
            history_args = {
                "comparison_axis": "binary_boundary",
                "focus_labels": focus_labels,
                "current_case_summary": (
                    f"Four-stage 初判={initial_labels}。"
                    f"初判依据：{str(initial.get('audit_trace_text') or '')[:1200]}"
                ),
                "evidence_question": (
                    "结合当前完整证据与同 source 的 Train 通过/违规先例，"
                    "判断 Four-stage 初判是否跨越了真实违规边界。"
                ),
                "_audit_context": context,
            }
            history_started = time.time()
            history_result = history_tool.run(history_args)
            history_latency_ms = (time.time() - history_started) * 1000
            history_analysis = dict(history_result.get("analysis") or {})
            history_applicability = str(
                history_analysis.get("applicability") or "insufficient"
            )

            tool_section, tool_images = render_observations(
                observations,
                DEFAULT_TOOL_REGISTRY,
            )
            rule_images = collect_rule_images(rules)
            prompt = gate_agent.build_prompt(
                "history_gate_reviewer",
                note=ai.note,
                source_id=ai.source_id,
                candidate_labels=gate_labels,
                rules=rules_data,
                tool_section=tool_section,
                initial_predict_label=initial_labels,
                initial_audit_trace=str(initial.get("audit_trace_text") or ""),
                history_analysis=history_analysis,
                history_cases=list(history_result.get("history_cases") or []),
                history_context_mode=args.history_context_mode,
            )
            messages = [
                make_multimodal_message(
                    "user",
                    prompt,
                    list(ai.images or []) + rule_images + tool_images,
                    image_max_tokens=args.image_max_tokens,
                )
            ]
            parsed, trace = gate_agent.run_messages(
                messages,
                role_tag="history_gate",
                template_name="history_gate_reviewer",
                response_format=gate_response_format(gate_labels),
            )
            trace_data = trace.model_dump()
            parsed_labels = dedupe(parsed.get("predict_label") or [])
            valid_labels = [label for label in parsed_labels if label in set(gate_labels)]
            if trace.error or len(valid_labels) != 1:
                raise RuntimeError(trace.error or "invalid gate prediction")

            final_labels = valid_labels
            computed_action = "keep" if final_labels == initial_labels else "revise"
            history_usage = dict(history_result.get("analyst_usage") or {})
            return {
                **base_result,
                "success": True,
                "gate_success": True,
                "fallback_to_initial": False,
                "gate_candidate_labels": gate_labels,
                "focus_labels": focus_labels,
                "predict_label": final_labels,
                "binary_decision": binary_kind(final_labels),
                "gate_action": computed_action,
                "model_gate_action": parsed.get("gate_action"),
                "history_applicability": history_applicability,
                "history_effect": parsed.get("history_effect"),
                "decisive_evidence": list(parsed.get("decisive_evidence") or []),
                "audit_trace_text": str(parsed.get("audit_trace") or ""),
                "history_analysis": history_analysis,
                "history_cases": list(history_result.get("history_cases") or []),
                "history_snapshot": history_result.get("history_snapshot"),
                "history_latency_ms": history_latency_ms,
                "history_analyst_image_count": history_result.get("analyst_image_count", 0),
                "gate_latency_ms": trace.latency_ms,
                "gate_raw_response": trace.raw_response,
                "error": None,
                "history_usage": history_usage,
                "history_prompt_tokens": history_usage.get("prompt_tokens"),
                "history_completion_tokens": history_usage.get("completion_tokens"),
                "history_total_tokens": history_usage.get("total_tokens"),
                "history_reasoning_tokens": history_usage.get("reasoning_tokens"),
                **usage_fields(trace_data, "gate"),
            }
        except Exception as exc:
            return {
                **base_result,
                "success": True,
                "gate_success": False,
                "fallback_to_initial": True,
                "predict_label": initial_labels,
                "binary_decision": binary_kind(initial_labels),
                "gate_action": "keep",
                "history_applicability": "unknown",
                "audit_trace_text": str(initial.get("audit_trace_text") or ""),
                "error": f"history_gate_failed: {type(exc).__name__}: {exc}",
            }

    results = list(existing)
    started = time.time()
    done = 0
    print_every = max(1, len(todo) // 100) if todo else 1
    mode = "a" if args.resume and rows_path.exists() else "w"
    with rows_path.open(mode, encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(process, item) for item in todo]
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                done += 1
                if done % print_every == 0 or done == len(todo):
                    elapsed = time.time() - started
                    rate = done / max(elapsed, 0.001)
                    revised = sum(1 for value in results if value.get("gate_action") == "revise")
                    failed = sum(1 for value in results if not value.get("gate_success"))
                    print(
                        f"  [{done}/{len(todo)}] {rate:.2f} rows/s "
                        f"revised={revised} gate_failed={failed} elapsed={elapsed:.0f}s",
                        file=sys.stderr,
                    )

    metrics = build_metrics(results, time.time() - started, run_config)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[history-gate] wrote {rows_path}", file=sys.stderr)
    print(f"[history-gate] wrote {metrics_path}", file=sys.stderr)
    print(
        json.dumps(
            {
                "total": metrics["total"],
                "gate_success_rate": metrics["gate_success_rate"],
                "initial": metrics["initial"],
                "history_gate": metrics["history_gate"],
                "delta": metrics["delta"],
                "changes": metrics["changes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
