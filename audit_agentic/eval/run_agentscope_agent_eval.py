"""Evaluate audit cases through AgentScope's native agent runtime.

The runner intentionally reuses the existing audit rule retriever, experience
loader, tool registry, ToolExecutor and local caches. It changes only the
native reasoning/tool-call runtime so results can be compared with the current
four-stage workflow and the hand-written native loop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIT_ROOT.parent
AGENTICRL_ROOT = PROJECT_ROOT.parent
AGENTSCOPE_ROOT = Path(os.environ.get("AGENTSCOPE_ROOT", str(PROJECT_ROOT / "agentscope")))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(AGENTSCOPE_ROOT / "src"))

PASS_LABEL = "通过"
DEFAULT_IMAGE_CACHE = Path(
    os.environ.get("IMAGE_CACHE_DIR", str(PROJECT_ROOT / "local_artifacts/images"))
)
DEFAULT_BLOB_INDEX = Path(
    os.environ.get("BLOB_INDEX_PATH", str(PROJECT_ROOT / "local_artifacts/blob_index.pkl"))
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate audit JSONL with AgentScope 2.0 native tools.",
    )
    parser.add_argument("--input", default=str(AUDIT_ROOT / "data" / "test_local.jsonl"))
    parser.add_argument("--config", default=str(AUDIT_ROOT / "config.example.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--row-indices",
        default="",
        help="Comma-separated original JSONL row indices. Applied before --limit.",
    )
    parser.add_argument(
        "--sources",
        default="",
        help="Comma-separated source allowlist. Applied before --limit.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--structured-output-grace-iters", type=int, default=2)
    parser.add_argument(
        "--router-mode",
        choices=["none", "agentscope"],
        default="none",
        help="Optional AgentScope Router before rule injection.",
    )
    parser.add_argument("--router-bypass-max-labels", type=int, default=8)
    parser.add_argument(
        "--rule-loading-mode",
        choices=["all", "preview_tool"],
        default="all",
    )
    parser.add_argument("--rule-preview-max-chars", type=int, default=700)
    parser.add_argument("--detail-rule-max-labels", type=int, default=6)
    parser.add_argument("--parallel-tool-calls", action="store_true")
    parser.add_argument(
        "--context-compression",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compression-trigger-tokens",
        type=int,
        default=0,
        help="0 keeps AgentScope's default 80%% trigger.",
    )
    parser.add_argument(
        "--compression-profile",
        choices=["default", "audit"],
        default="default",
        help="Compression prompt/schema profile used after the trigger.",
    )
    parser.add_argument("--image-max-tokens", type=int, default=448)
    parser.add_argument(
        "--rule-fetch-concurrency",
        type=int,
        default=8,
        help="Concurrent per-source Jupiter fetches in online data-access mode.",
    )
    parser.add_argument("--online-service-timeout", type=float, default=20.0)
    parser.add_argument("--online-service-max-attempts", type=int, default=3)
    parser.add_argument("--online-service-retry-wait", type=float, default=1.0)
    parser.add_argument("--online-http-pool-size", type=int, default=32)
    parser.add_argument(
        "--data-access-mode",
        choices=["cache", "online"],
        default="cache",
        help="cache keeps current local artifacts; online calls live rule/tool services.",
    )
    parser.add_argument(
        "--rule-access-mode",
        choices=["cache", "online"],
        default="cache",
        help="Load rules from local cache or live Jupiter independently of tools.",
    )
    parser.add_argument(
        "--experience",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--experience-path",
        default=str(
            AUDIT_ROOT
            / "environment"
            / "cache"
            / "experience"
            / "audit_experience.json"
        ),
    )
    parser.add_argument(
        "--rule-cache",
        default=str(AUDIT_ROOT / "environment" / "cache" / "rules" / "test"),
    )
    parser.add_argument(
        "--tool-cache",
        default=str(AUDIT_ROOT / "environment" / "cache" / "test_jhx"),
    )
    parser.add_argument(
        "--tool-cache-version",
        default="tool-cache-v2-example_user",
        help="Version string embedded in the deterministic tool-cache key.",
    )
    parser.add_argument(
        "--tool-cache-identity-mode",
        choices=["default", "note_history"],
        default="default",
        help=(
            "default preserves the existing public tool args; note_history "
            "matches caches keyed only by note_id and history_id."
        ),
    )
    parser.add_argument("--full-trace", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[tuple[int, Dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row_idx, row


def load_existing(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows: Dict[int, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "row_idx" in row:
                rows[int(row["row_idx"])] = row
    return rows


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["AUDIT_DATA_ACCESS_MODE"] = args.data_access_mode
    os.environ["AUDIT_RULE_ACCESS_MODE"] = args.rule_access_mode
    os.environ["AUDIT_RULE_CACHE_PATH"] = str(Path(args.rule_cache).resolve())
    os.environ["AUDIT_TOOL_RESULT_CACHE_PATH"] = str(Path(args.tool_cache).resolve())
    os.environ["AUDIT_TOOL_CACHE_VERSION"] = args.tool_cache_version
    os.environ["AUDIT_TOOL_CACHE_IDENTITY_MODE"] = args.tool_cache_identity_mode
    os.environ["AUDIT_EXPERIENCE_PATH"] = str(Path(args.experience_path).resolve())
    if args.data_access_mode == "online":
        os.environ["AUDIT_ENABLE_RULE_IMAGES"] = "1"
        os.environ["AUDIT_DISABLE_LOCAL_IMAGE_RESIZE"] = "0"
        os.environ["AUDIT_SKIP_UNAVAILABLE_IMAGES"] = "1"
        os.environ["AUDIT_TOOL_RESULT_CACHE_DISABLE"] = "1"
        os.environ["AUDIT_ONLINE_RULE_CACHE_DISABLE"] = (
            "1" if args.rule_access_mode == "online" else "0"
        )
        os.environ["AUDIT_DISABLE_ONLINE_SERVICES"] = "0"
        os.environ["AUDIT_ONLINE_SERVICE_TIMEOUT"] = str(
            max(0.1, args.online_service_timeout),
        )
        os.environ["AUDIT_ONLINE_SERVICE_MAX_ATTEMPTS"] = str(
            max(1, args.online_service_max_attempts),
        )
        os.environ["AUDIT_ONLINE_SERVICE_RETRY_WAIT"] = str(
            max(0.0, args.online_service_retry_wait),
        )
        os.environ["AUDIT_ONLINE_HTTP_POOL_SIZE"] = str(
            max(1, args.online_http_pool_size),
        )
    else:
        os.environ.setdefault("AUDIT_ENABLE_RULE_IMAGES", "0")
        os.environ.setdefault("AUDIT_DISABLE_LOCAL_IMAGE_RESIZE", "1")
    os.environ.setdefault("AUDIT_TOOL_IMAGE_LIMIT", "-1")
    os.environ.setdefault("BLOB_INDEX_PATH", str(DEFAULT_BLOB_INDEX))
    os.environ.setdefault("IMAGE_CACHE_DIR", str(DEFAULT_IMAGE_CACHE))
    os.environ.setdefault(
        "PLAGIARISM_CACHE_PATH",
        str(AUDIT_ROOT / "environment" / "cache" / "disabled_plagiarism_cache.pkl"),
    )
    os.environ.setdefault("RECENT_NOTES_MAX_NOTES", "5")
    os.environ.setdefault("RECENT_NOTES_MAX_IMAGES_PER_NOTE", "4")
    os.environ.setdefault("COMMERCIAL_DETAIL_MAX_IMAGES_PER_ITEM", "2")


def build_model(args: argparse.Namespace, config: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
    from agentscope.credential import OpenAICredential
    from agentscope.model import OpenAIChatModel

    cfg = dict(config["main_agent"])
    if cfg.get("api_format") == "maas-gemini":
        from .maas_gemini_model import build_maas_gemini_model

        return build_maas_gemini_model(args, cfg)
    if cfg.get("api_format", "openai") == "maas-bedrock":
        from .maas_bedrock_model import build_maas_model

        return build_maas_model(args, cfg)
    temperature = cfg.get("temperature", 0.7) if args.temperature is None else args.temperature
    max_tokens = cfg.get("max_tokens", 4096) if args.max_tokens is None else args.max_tokens
    extra_body = dict(cfg.get("extra_payload") or {})
    chat_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_kwargs["enable_thinking"] = bool(args.thinking)
    extra_body["chat_template_kwargs"] = chat_kwargs
    api_key = cfg.get("api_key") or os.environ.get(cfg.get("api_key_env") or "OPENAI_API_KEY") or "EMPTY"

    model = OpenAIChatModel(
        credential=OpenAICredential(
            api_key=api_key,
            base_url=cfg["api_base"],
        ),
        model=cfg["model"],
        parameters=OpenAIChatModel.Parameters(
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_enable=bool(args.thinking),
            parallel_tool_calls=bool(args.parallel_tool_calls),
        ),
        stream=False,
        max_retries=int(cfg.get("max_retries", 3)),
        retry_delay=float(cfg.get("retry_delay", 1.0)),
        client_kwargs={"timeout": args.timeout},
        extra_body=extra_body,
    )
    public = {
        "api_base": cfg["api_base"],
        "model": cfg["model"],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": bool(args.thinking),
        "parallel_tool_calls": bool(args.parallel_tool_calls),
        "timeout": args.timeout,
    }
    if extra_body.get("reasoning_effort") is not None:
        public["reasoning_effort"] = extra_body["reasoning_effort"]
    return model, public


def binary_kind(labels: List[str]) -> str:
    return "pass" if set(labels) == {PASS_LABEL} else "violation"


def compute_metrics(rows: List[Dict[str, Any]], wall_seconds: float) -> Dict[str, Any]:
    def safe(num: int | float, den: int | float) -> float:
        return num / den if den else 0.0

    ok = [row for row in rows if row.get("success")]
    exact = binary = tp = tn = fp = fn = 0
    by_source = defaultdict(lambda: {"total": 0, "exact": 0, "binary": 0})
    for row in ok:
        gt = list(row.get("gt_labels") or [])
        pred = list(row.get("predict_label") or [])
        exact_hit = set(gt) == set(pred)
        binary_hit = binary_kind(gt) == binary_kind(pred)
        exact += int(exact_hit)
        binary += int(binary_hit)
        source = by_source[str(row.get("source") or "<unknown>")]
        source["total"] += 1
        source["exact"] += int(exact_hit)
        source["binary"] += int(binary_hit)
        if binary_kind(gt) == "violation" and binary_kind(pred) == "violation":
            tp += 1
        elif binary_kind(gt) == "pass" and binary_kind(pred) == "pass":
            tn += 1
        elif binary_kind(gt) == "pass":
            fp += 1
        else:
            fn += 1

    tool_counts = Counter(
        name
        for row in rows
        for name in row.get("used_tools") or []
    )
    latencies = [float(row.get("latency_seconds") or 0.0) for row in rows]
    router_rows = [row for row in ok if row.get("router_invoked")]
    router_gt_survival = sum(
        bool(set(row.get("gt_labels") or []) & set(row.get("router_shortlist_labels") or []))
        for row in router_rows
    )
    router_shortlist_sizes = [
        len(row.get("router_shortlist_labels") or [])
        for row in router_rows
    ]
    preview_rows = [row for row in ok if row.get("rule_loading_mode") == "preview_tool"]
    detail_rows = [row for row in preview_rows if row.get("detail_rule_call_count")]
    violation_without_detail = sum(
        row.get("binary_decision") == "violation"
        and not row.get("final_label_detail_loaded")
        for row in preview_rows
    )
    compression_rows = [
        row
        for row in ok
        if int(row.get("context_compression_count") or 0) > 0
    ]
    compression_count = sum(
        int(row.get("context_compression_count") or 0)
        for row in ok
    )
    return {
        "framework": "agentscope-2.0.5",
        "total": len(rows),
        "success": len(ok),
        "success_rate": safe(len(ok), len(rows)),
        "nOA": safe(exact, len(ok)),
        "wOA": safe(binary, len(ok)),
        "confusion": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "violation_recall": safe(tp, tp + fn),
        "pass_recall": safe(tn, tn + fp),
        "balanced_accuracy": (safe(tp, tp + fn) + safe(tn, tn + fp)) / 2,
        "wall_seconds": round(wall_seconds, 3),
        "latency_seconds_avg": safe(sum(latencies), len(latencies)),
        "tool_rows": sum(bool(row.get("used_tools")) for row in rows),
        "tool_calls_by_name": dict(tool_counts.most_common()),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "context_compression": {
            "enabled": next(
                (
                    bool(row.get("context_compression_enabled"))
                    for row in ok
                    if "context_compression_enabled" in row
                ),
                False,
            ),
            "profile": next(
                (
                    str(row.get("context_compression_profile") or "default")
                    for row in ok
                    if "context_compression_profile" in row
                ),
                "default",
            ),
            "trigger_tokens": next(
                (
                    int(row.get("context_compression_trigger_tokens") or 0)
                    for row in ok
                    if row.get("context_compression_trigger_tokens") is not None
                ),
                0,
            ),
            "compressed_rows": len(compression_rows),
            "compressed_row_rate": safe(len(compression_rows), len(ok)),
            "compression_count": compression_count,
            "compressions_per_compressed_row_avg": safe(
                compression_count,
                len(compression_rows),
            ),
        },
        "router": {
            "mode": next(
                (str(row.get("router_mode")) for row in ok if row.get("router_mode")),
                "none",
            ),
            "invoked_rows": len(router_rows),
            "bypassed_rows": len(ok) - len(router_rows),
            "shortlist_size_avg": safe(
                sum(router_shortlist_sizes),
                len(router_shortlist_sizes),
            ),
            "gt_survival_rate": safe(router_gt_survival, len(router_rows)),
        },
        "rule_loading": {
            "mode": next(
                (
                    str(row.get("rule_loading_mode"))
                    for row in ok
                    if row.get("rule_loading_mode")
                ),
                "all",
            ),
            "preview_rows": len(preview_rows),
            "detail_rule_rows": len(detail_rows),
            "detail_rule_row_rate": safe(len(detail_rows), len(preview_rows)),
            "detail_rule_calls": sum(
                int(row.get("detail_rule_call_count") or 0)
                for row in preview_rows
            ),
            "loaded_rule_labels_avg": safe(
                sum(len(row.get("loaded_rule_labels") or []) for row in preview_rows),
                len(preview_rows),
            ),
            "violation_without_detail": violation_without_detail,
        },
        "per_source": {
            source: {
                "total": values["total"],
                "nOA": safe(values["exact"], values["total"]),
                "wOA": safe(values["binary"], values["total"]),
            }
            for source, values in sorted(by_source.items())
        },
    }


async def async_main(args: argparse.Namespace) -> None:
    configure_environment(args)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    metrics_path = output_dir / "metrics.json"
    run_config_path = output_dir / "run_config.json"

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model, public_model_config = build_model(args, config)

    from audit_agentic.agents.agentscope_audit_agent import AgentScopeAuditAgent
    from audit_agentic.environment.rule_retriever import (
        OnlineRuleRetriever,
        retrieve_rules_with_id_fallback,
    )
    from audit_agentic.environment.tool_executor import ToolExecutor
    from audit_agentic.environment.tool_registry import DEFAULT_TOOL_REGISTRY
    from audit_agentic.prompts.loader import default_loader
    from audit_agentic.schemas import AuditInput

    selected_indices = {
        int(value.strip())
        for value in args.row_indices.split(",")
        if value.strip()
    }
    selected_sources = {
        value.strip()
        for value in args.sources.split(",")
        if value.strip()
    }
    indexed: List[tuple[int, Dict[str, Any], Any]] = []
    for row_idx, row in iter_jsonl(input_path):
        audit_input = AuditInput.from_data_row(row)
        if selected_indices and row_idx not in selected_indices:
            continue
        if selected_sources and audit_input.source_id not in selected_sources:
            continue
        indexed.append((row_idx, row, audit_input))
    if args.limit:
        indexed = indexed[: args.limit]

    retriever = OnlineRuleRetriever()
    labels_by_source: Dict[str, List[str]] = defaultdict(list)
    for _, _, audit_input in indexed:
        labels_by_source[audit_input.source_id].extend(audit_input.candidate_labels)
    labels_by_source = {
        source: list(dict.fromkeys(labels))
        for source, labels in labels_by_source.items()
    }
    if args.rule_access_mode == "cache":
        for source, labels in labels_by_source.items():
            retriever.retrieve(source, labels)
    else:
        rule_fetch_semaphore = asyncio.Semaphore(max(1, args.rule_fetch_concurrency))

        async def prefetch_online_rules(source: str, labels: List[str]) -> None:
            async with rule_fetch_semaphore:
                await asyncio.to_thread(retriever.retrieve, source, labels)

        await asyncio.gather(*(
            prefetch_online_rules(source, labels)
            for source, labels in labels_by_source.items()
        ))

    prepared = []
    for row_idx, row, audit_input in indexed:
        rules = retrieve_rules_with_id_fallback(
            retriever,
            audit_input,
            audit_input.candidate_labels,
        )
        prepared.append((row_idx, row, audit_input, rules))

    existing = load_existing(rows_path) if args.resume else {}
    done_indices = {
        row_idx
        for row_idx, row in existing.items()
        if not (args.retry_failed and not row.get("success"))
    }
    todo = [item for item in prepared if item[0] not in done_indices]
    print(
        f"[agentscope] total={len(prepared)} existing={len(existing)} "
        f"todo={len(todo)} concurrency={args.concurrency} model={public_model_config['model']}",
        file=sys.stderr,
    )

    run_config = {
        "framework": "agentscope-2.0.5",
        "agentscope_root": str(AGENTSCOPE_ROOT),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "concurrency": args.concurrency,
        "limit": args.limit,
        "row_indices": sorted(selected_indices),
        "sources": sorted(selected_sources),
        "experience": args.experience,
        "experience_path": str(Path(args.experience_path).resolve()),
        "rule_cache": str(Path(args.rule_cache).resolve()),
        "tool_cache": str(Path(args.tool_cache).resolve()),
        "tool_cache_version": args.tool_cache_version,
        "tool_cache_identity_mode": args.tool_cache_identity_mode,
        "data_access_mode": args.data_access_mode,
        "rule_access_mode": args.rule_access_mode,
        "rule_fetch_concurrency": args.rule_fetch_concurrency,
        "online_service_timeout": args.online_service_timeout,
        "online_service_max_attempts": args.online_service_max_attempts,
        "online_service_retry_wait": args.online_service_retry_wait,
        "online_http_pool_size": args.online_http_pool_size,
        "image_max_tokens": args.image_max_tokens,
        "max_iters": args.max_iters,
        "structured_output_grace_iters": args.structured_output_grace_iters,
        "router_mode": args.router_mode,
        "router_bypass_max_labels": args.router_bypass_max_labels,
        "rule_loading_mode": args.rule_loading_mode,
        "rule_preview_max_chars": args.rule_preview_max_chars,
        "detail_rule_max_labels": args.detail_rule_max_labels,
        "context_compression": args.context_compression,
        "compression_trigger_tokens": args.compression_trigger_tokens,
        "compression_profile": args.compression_profile,
        **public_model_config,
    }
    run_config_path.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    loader = default_loader()
    executor = ToolExecutor(DEFAULT_TOOL_REGISTRY)

    async def process(item: tuple[int, Dict[str, Any], Any, List[Any]]) -> Dict[str, Any]:
        row_idx, _, audit_input, rules = item
        queued_at = time.time()
        async with semaphore:
            started = time.time()
            queue_wait_seconds = round(started - queued_at, 3)
            try:
                agent = AgentScopeAuditAgent(
                    model=model,
                    prompt_loader=loader,
                    tool_executor=executor,
                    tool_registry=DEFAULT_TOOL_REGISTRY,
                    max_iters=args.max_iters,
                    structured_output_grace_iters=args.structured_output_grace_iters,
                    experience_enabled=args.experience,
                    router_mode=args.router_mode,
                    router_bypass_max_labels=args.router_bypass_max_labels,
                    rule_loading_mode=args.rule_loading_mode,
                    rule_preview_max_chars=args.rule_preview_max_chars,
                    detail_rule_max_labels=args.detail_rule_max_labels,
                    compression_enabled=args.context_compression,
                    compression_trigger_tokens=args.compression_trigger_tokens,
                    compression_profile=args.compression_profile,
                    full_trace=args.full_trace,
                )
                result, trace = await agent.run(
                    audit_input,
                    audit_input.candidate_labels,
                    rules,
                    image_max_tokens=args.image_max_tokens,
                )
                row_result: Dict[str, Any] = {
                    "row_idx": row_idx,
                    "note_id": audit_input.note_id,
                    "source": audit_input.source_id,
                    "gt_labels": list(audit_input.gt_labels or []),
                    "candidate_labels": list(audit_input.candidate_labels),
                    "predict_label": list(result.predict_label or []),
                    "binary_decision": result.binary_decision,
                    "audit_trace": result.audit_trace,
                    "used_rules": list(result.used_rules or []),
                    "used_tools": list(result.used_tools or []),
                    "success": bool(result.predict_label),
                    "framework": trace.get("mode"),
                    "finished_reason": trace.get("finished_reason"),
                    "model_calls": int(trace.get("model_calls") or 0),
                    "prompt_tokens": int(trace.get("prompt_tokens") or 0),
                    "completion_tokens": int(trace.get("completion_tokens") or 0),
                    "tool_calls": list(trace.get("tool_calls") or []),
                    "available_tool_names": list(trace.get("available_tool_names") or []),
                    "experience_enabled": bool(trace.get("experience_enabled")),
                    "experience_chars": int(trace.get("experience_chars") or 0),
                    "router_mode": str(trace.get("router_mode") or "none"),
                    "router_invoked": bool(trace.get("router_invoked")),
                    "router_shortlist_labels": list(
                        trace.get("router_shortlist_labels") or [],
                    ),
                    "router_trace": dict(trace.get("router") or {}),
                    "rule_loading_mode": str(trace.get("rule_loading_mode") or "all"),
                    "rule_preview_chars": int(trace.get("rule_preview_chars") or 0),
                    "detail_rule_call_count": int(
                        trace.get("detail_rule_call_count") or 0,
                    ),
                    "loaded_rule_labels": list(trace.get("loaded_rule_labels") or []),
                    "final_label_detail_loaded": bool(
                        trace.get("final_label_detail_loaded"),
                    ),
                    "context_compression_enabled": bool(
                        trace.get("context_compression_enabled"),
                    ),
                    "context_compression_profile": str(
                        trace.get("context_compression_profile") or "default",
                    ),
                    "context_compression_trigger_tokens": int(
                        trace.get("context_compression_trigger_tokens") or 0,
                    ),
                    "context_compression_count": int(
                        trace.get("context_compression_count") or 0,
                    ),
                    "queue_wait_seconds": queue_wait_seconds,
                    "latency_seconds": round(time.time() - started, 3),
                    "errors": [],
                }
                if args.full_trace:
                    row_result["agentscope_trace"] = trace
                return row_result
            except Exception as exc:  # noqa: BLE001
                return {
                    "row_idx": row_idx,
                    "note_id": audit_input.note_id,
                    "source": audit_input.source_id,
                    "gt_labels": list(audit_input.gt_labels or []),
                    "candidate_labels": list(audit_input.candidate_labels),
                    "predict_label": [],
                    "binary_decision": "",
                    "audit_trace": "",
                    "used_rules": [],
                    "used_tools": [],
                    "success": False,
                    "framework": "agentscope_native",
                    "queue_wait_seconds": queue_wait_seconds,
                    "latency_seconds": round(time.time() - started, 3),
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }

    started = time.time()
    results_by_idx = dict(existing)
    mode = "a" if args.resume and rows_path.exists() else "w"
    with rows_path.open(mode, encoding="utf-8") as handle:
        tasks = [asyncio.create_task(process(item)) for item in todo]
        for done, task in enumerate(asyncio.as_completed(tasks), 1):
            row = await task
            results_by_idx[int(row["row_idx"])] = row
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[agentscope] {done}/{len(todo)} row={row['row_idx']} "
                f"success={row['success']} tools={row.get('used_tools') or []} "
                f"latency={row['latency_seconds']}s",
                file=sys.stderr,
            )

    rows = [
        results_by_idx[row_idx]
        for row_idx, _, _, _ in prepared
        if row_idx in results_by_idx
    ]
    metrics = compute_metrics(rows, time.time() - started)
    metrics["config"] = run_config
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[agentscope] rows -> {rows_path}", file=sys.stderr)
    print(f"[agentscope] metrics -> {metrics_path}", file=sys.stderr)
    client = getattr(model, "client", None)
    if client is not None:
        await client.close()


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
