"""Relax managed-agent entry for the AgentScope native audit loop.

The process is launched once per Relax managed session. Every model call is
routed through ``RELAX_BASE_URL`` with ``RELAX_SESSION_ID`` as the API key, so
Relax records the generated reasoning, native tool calls, and final structured
output as one trainable chat trajectory.

The first training integration intentionally uses one AgentScope main agent
(``router_mode=none``) and Relax's implicit export. This avoids reconstructing
AgentScope messages for explicit state-hash matching and keeps the rollout
forest linear.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

from agentscope.credential import OpenAICredential
from audit_agentic.agents.agentscope_audit_agent import AgentScopeAuditAgent
from audit_agentic.environment.rule_retriever import (
    OnlineRuleRetriever,
    retrieve_rules_with_id_fallback,
)
from audit_agentic.environment.tool_executor import ToolExecutor
from audit_agentic.environment.tool_registry import DEFAULT_TOOL_REGISTRY
from audit_agentic.prompts.loader import default_loader
from audit_agentic.relax_app.protocol_openai_model import (
    ProtocolAwareOpenAIChatModel,
)
from audit_agentic.relax_app.protocol_reward import score_agentscope_protocol
from audit_agentic.rewards.process_judge_client import (
    ProcessJudgeClient,
    process_judge_enabled,
)
from audit_agentic.rewards.reward_adaptive import (
    compute_final_reward,
    extract_four_round_labels,
)
from audit_agentic.schemas import AuditInput


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _protocol_exception_types(exc: Exception) -> list[str]:
    """Recognize model-output protocol failures without hiding infra errors."""
    message = str(exc or "").lower()
    error_types: list[str] = []
    if "not defined in the tools list" in message:
        error_types.append("unknown_tool_call")
    if "structured output" in message or "structured_output" in message:
        error_types.append("structured_output_error")
    if "literal_eval" in message or "tool arguments" in message:
        error_types.append("invalid_tool_arguments")
    return error_types


def _is_infrastructure_exception(exc: Exception) -> bool:
    """Keep transport/server failures retryable at the Relax group level."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    module = exc.__class__.__module__.lower()
    if module.startswith(("httpx", "openai")):
        return True
    message = str(exc or "").lower()
    return any(
        marker in message
        for marker in (
            "500 internal server error",
            "502 bad gateway",
            "503 service unavailable",
            "504 gateway timeout",
            "connection refused",
            "connection reset",
            "request is disconnected",
            "duplicate request ids",
            "timed out",
        )
    )


def _thinking_enabled() -> bool:
    raw = os.getenv("AUDIT_AGENTSCOPE_THINKING")
    if raw is not None:
        return _env_bool("AUDIT_AGENTSCOPE_THINKING", False)
    template_raw = os.getenv("AUDIT_CHAT_TEMPLATE_KWARGS") or os.getenv(
        "APPLY_CHAT_TEMPLATE_KWARGS",
        "",
    )
    try:
        parsed = json.loads(template_raw) if template_raw else {}
    except json.JSONDecodeError:
        parsed = {}
    return bool(parsed.get("enable_thinking", False))


def _max_tokens() -> int:
    raw = (
        os.getenv("AUDIT_AGENTSCOPE_MAX_TOKENS")
        or os.getenv("AUDIT_LLM_MAX_TOKENS")
        or os.getenv("ROLLOUT_MAX_RESPONSE_LEN")
        or "4096"
    )
    return int(raw)


def _protocol_metadata(
    model: ProtocolAwareOpenAIChatModel,
    *,
    invalid_final_output: bool,
    final_protocol_error_count: int,
) -> Dict[str, float]:
    diagnostics = model.protocol_diagnostics
    wire_stats = model.wire_history_stats
    return {
        "agentscope/invalid_final_output": float(invalid_final_output),
        "agentscope/final_protocol_error_count": float(
            final_protocol_error_count,
        ),
        "agentscope/unknown_tool_call_count": float(
            diagnostics.get("unknown_tool_call_count") or 0,
        ),
        "agentscope/tool_name_format_error_count": float(
            diagnostics.get("tool_name_format_error_count") or 0,
        ),
        "agentscope/invalid_tool_arguments_count": float(
            diagnostics.get("invalid_tool_arguments_count") or 0,
        ),
        "agentscope/repaired_tool_call_count": float(
            diagnostics.get("repaired_tool_call_count") or 0,
        ),
        "agentscope/wire_history_message_count": float(
            wire_stats.get("message_count") or 0,
        ),
        "agentscope/wire_history_canonicalized_turns": float(
            wire_stats.get("canonicalized_turns") or 0,
        ),
        "agentscope/wire_history_rejected_turns": float(
            wire_stats.get("rejected_turns") or 0,
        ),
    }


def _adaptive_reward_enabled() -> bool:
    return _env_bool("AUDIT_ADAPTIVE_REWARD_ENABLED", False)


def _rule_consistency_verdict(audit_input: AuditInput) -> Any:
    metadata = audit_input.metadata or {}
    judge = metadata.get("rule_consistency_judge") or metadata.get("rule_judge")
    if isinstance(judge, dict):
        return judge.get("verdict")
    confidence = metadata.get("gt_confidence")
    if isinstance(confidence, dict):
        return confidence.get("rule_support_verdict")
    return None


def _empty_process_judge_result(*, enabled: bool, error: str = "") -> Dict[str, Any]:
    return {
        "enabled": enabled,
        "success": False,
        "scores": {},
        "process_reward": None,
        "latency_seconds": 0.0,
        "retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "source_post_image_count": 0,
        "source_tool_image_count": 0,
        "post_image_count": 0,
        "tool_image_count": 0,
        "final_image_count": 0,
        "dropped_post_image_count": 0,
        "dropped_tool_image_count": 0,
        "error": error,
    }


async def _evaluate_process_reward(
    *,
    audit_input: AuditInput,
    result: Any,
    trace: Dict[str, Any],
) -> Dict[str, Any]:
    if not process_judge_enabled():
        return _empty_process_judge_result(enabled=False)
    if trace.get("invalid_final_output"):
        return _empty_process_judge_result(
            enabled=True,
            error="invalid final output; Process Judge skipped",
        )
    try:
        evaluated = await ProcessJudgeClient.from_env().evaluate(
            trace=trace,
            prediction=list(result.predict_label or []),
            binary_decision=str(result.binary_decision or ""),
            audit_trace=str(result.audit_trace or ""),
            used_rules=list(result.used_rules or []),
            used_tools=list(result.used_tools or []),
            candidate_labels=list(audit_input.candidate_labels or []),
            human_cot=str((audit_input.metadata or {}).get("human_cot") or ""),
            source_id=str(audit_input.source_id or ""),
            note_id=str(audit_input.note_id or ""),
            protocol_issues=list(trace.get("final_protocol_error_types") or []),
        )
        evaluated["enabled"] = True
        return evaluated
    except Exception as exc:
        return _empty_process_judge_result(
            enabled=True,
            error=f"{type(exc).__name__}: {exc}",
        )


def _adaptive_reward(
    *,
    audit_input: AuditInput,
    prediction: list[str],
    process_judge: Dict[str, Any],
    invalid_final_output: bool,
    diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    if _env_bool("AUDIT_PROCESS_JUDGE_REQUIRE_SCORE", False) and not invalid_final_output:
        value = process_judge.get("process_reward")
        if (
            not process_judge.get("enabled")
            or not process_judge.get("success")
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            # A transport failure is not a process score. Fail this session before export.
            raise RuntimeError(
                "Required Process Judge score unavailable: "
                f"note_id={audit_input.note_id}; "
                f"error={str(process_judge.get('error') or 'missing/invalid score')[:500]}"
            )
    return compute_final_reward(
        human_labels=extract_four_round_labels(audit_input.metadata or {}),
        rule_verdict=_rule_consistency_verdict(audit_input),
        gt=list(audit_input.gt_labels or []),
        prediction=list(prediction or []),
        process_reward_value=(
            process_judge.get("process_reward")
            if process_judge.get("success")
            else None
        ),
        invalid_final_output=invalid_final_output,
        unknown_tool_call_count=int(
            diagnostics.get("unknown_tool_call_count") or 0,
        ),
        tool_name_format_error_count=int(
            diagnostics.get("tool_name_format_error_count") or 0,
        ),
        invalid_tool_arguments_count=int(
            diagnostics.get("invalid_tool_arguments_count") or 0,
        ),
    )


def _adaptive_metadata(
    adaptive: Dict[str, Any],
    process_judge: Dict[str, Any],
) -> Dict[str, float]:
    scores = process_judge.get("scores") or {}
    metadata = {
        "reward/adaptive_enabled": 1.0,
        "reward/human_confidence": float(adaptive["human_confidence"]),
        "reward/rule_gate": float(adaptive["rule_gate"]),
        "reward/gt_confidence": float(adaptive["gt_confidence"]),
        "reward/outcome": float(adaptive["outcome_reward"]),
        "reward/process": float(adaptive["process_reward"]),
        "reward/process_available": float(adaptive["process_reward_available"]),
        "reward/outcome_weight": float(adaptive["outcome_weight"]),
        "reward/process_weight": float(adaptive["process_weight"]),
        "reward/bounded_weight_enabled": float(
            adaptive["bounded_weight_enabled"],
        ),
        "reward/gt_weight_min": float(adaptive["gt_weight_min"]),
        "reward/gt_weight_max": float(adaptive["gt_weight_max"]),
        "reward/outcome_component": float(adaptive["outcome_component"]),
        "reward/process_component": float(adaptive["process_component"]),
        "reward/mixed_before_format": float(adaptive["mixed_reward_before_format"]),
        "reward/format_error_count": float(adaptive["format_error_count"]),
        "reward/format_penalty": float(adaptive["format_penalty"]),
        "reward/advantage_weight": float(adaptive["advantage_weight"]),
        "process_judge/enabled": float(bool(process_judge.get("enabled"))),
        "process_judge/success": float(bool(process_judge.get("success"))),
        "process_judge/latency_seconds": float(
            process_judge.get("latency_seconds") or 0,
        ),
        "process_judge/retry_count": float(process_judge.get("retry_count") or 0),
        "process_judge/prompt_tokens": float(process_judge.get("prompt_tokens") or 0),
        "process_judge/completion_tokens": float(
            process_judge.get("completion_tokens") or 0,
        ),
        "process_judge/reasoning_tokens": float(
            process_judge.get("reasoning_tokens") or 0,
        ),
        "process_judge/post_image_count": float(
            process_judge.get("post_image_count") or 0,
        ),
        "process_judge/tool_image_count": float(
            process_judge.get("tool_image_count") or 0,
        ),
        "process_judge/final_image_count": float(
            process_judge.get("final_image_count") or 0,
        ),
        "process_judge/dropped_post_image_count": float(
            process_judge.get("dropped_post_image_count") or 0,
        ),
        "process_judge/dropped_tool_image_count": float(
            process_judge.get("dropped_tool_image_count") or 0,
        ),
    }
    for name in (
        "factual_grounding",
        "rule_fidelity",
        "evidence_coverage",
        "tool_use",
        "decision_consistency",
    ):
        metadata[f"process_judge/{name}"] = float(scores.get(name) or 0)
    return metadata


def _build_model() -> ProtocolAwareOpenAIChatModel:
    base_url = os.getenv("RELAX_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError("RELAX_BASE_URL or OPENAI_BASE_URL is required")
    session_id = os.getenv("RELAX_SESSION_ID") or os.getenv("OPENAI_API_KEY")
    if not session_id:
        raise RuntimeError("RELAX_SESSION_ID or OPENAI_API_KEY is required")

    thinking = _thinking_enabled()
    temperature = float(
        os.getenv("AUDIT_AGENTSCOPE_TEMPERATURE")
        or os.getenv("AUDIT_LLM_TEMPERATURE")
        or os.getenv("ROLLOUT_TEMPERATURE")
        or "0.7"
    )
    extra_body = {
        "chat_template_kwargs": {
            "enable_thinking": thinking,
        },
    }
    return ProtocolAwareOpenAIChatModel(
        credential=OpenAICredential(
            api_key=session_id,
            base_url=base_url,
        ),
        model=os.getenv("RELAX_MODEL", "model"),
        parameters=ProtocolAwareOpenAIChatModel.Parameters(
            temperature=temperature,
            max_tokens=_max_tokens(),
            thinking_enable=thinking,
            parallel_tool_calls=_env_bool(
                "AUDIT_AGENTSCOPE_PARALLEL_TOOL_CALLS",
                False,
            ),
        ),
        stream=False,
        max_retries=int(os.getenv("AUDIT_AGENTSCOPE_MAX_RETRIES", "2")),
        retry_delay=float(os.getenv("AUDIT_AGENTSCOPE_RETRY_DELAY", "1")),
        client_kwargs={
            "timeout": float(
                os.getenv("AUDIT_AGENTSCOPE_CLIENT_TIMEOUT")
                or os.getenv("AUDIT_AGENT_CLIENT_TIMEOUT")
                or "900"
            ),
        },
        extra_body=extra_body,
    )


async def run_session(session_input: Dict[str, Any]) -> Dict[str, Any]:
    audit_input = AuditInput.from_data_row(session_input)
    router_mode = os.getenv("AUDIT_AGENTSCOPE_ROUTER_MODE", "none").strip().lower()
    if router_mode != "none":
        raise RuntimeError(
            "The first Relax-AgentScope integration requires "
            "AUDIT_AGENTSCOPE_ROUTER_MODE=none. A separate router creates "
            "multiple Relax export leaves and needs explicit export records."
        )

    retriever = OnlineRuleRetriever()
    rules = retrieve_rules_with_id_fallback(
        retriever,
        audit_input,
        audit_input.candidate_labels,
    )
    model = _build_model()
    agent = AgentScopeAuditAgent(
        model=model,
        prompt_loader=default_loader(),
        tool_executor=ToolExecutor(DEFAULT_TOOL_REGISTRY),
        tool_registry=DEFAULT_TOOL_REGISTRY,
        max_iters=int(os.getenv("AUDIT_AGENTSCOPE_MAX_ITERS", "8")),
        structured_output_grace_iters=int(
            os.getenv("AUDIT_AGENTSCOPE_STRUCTURED_OUTPUT_GRACE_ITERS", "2")
        ),
        experience_enabled=_env_bool("AUDIT_AGENTSCOPE_EXPERIENCE", True),
        router_mode=router_mode,
        router_bypass_max_labels=int(
            os.getenv("AUDIT_AGENTSCOPE_ROUTER_BYPASS_MAX_LABELS", "8")
        ),
        rule_loading_mode=os.getenv(
            "AUDIT_AGENTSCOPE_RULE_LOADING_MODE",
            "preview_tool",
        ),
        rule_preview_max_chars=int(
            os.getenv("AUDIT_AGENTSCOPE_RULE_PREVIEW_MAX_CHARS", "700")
        ),
        detail_rule_max_labels=int(
            os.getenv("AUDIT_AGENTSCOPE_DETAIL_RULE_MAX_LABELS", "6")
        ),
        compression_enabled=_env_bool(
            "AUDIT_AGENTSCOPE_COMPRESSION_ENABLED",
            True,
        ),
        compression_trigger_tokens=int(
            os.getenv("AUDIT_AGENTSCOPE_COMPRESSION_TRIGGER_TOKENS", "0"),
        ),
        full_trace=process_judge_enabled(),
    )
    try:
        result, trace = await agent.run(
            audit_input,
            audit_input.candidate_labels,
            rules,
            image_max_tokens=int(os.getenv("IMAGE_MAX_TOKENS", "448")),
        )
    except Exception as exc:
        if _is_infrastructure_exception(exc):
            raise
        inferred_errors = _protocol_exception_types(exc)
        diagnostics = model.protocol_diagnostics
        if not inferred_errors and not diagnostics.get("protocol_error_count"):
            raise

        final_errors = list(
            dict.fromkeys(inferred_errors + ["incomplete_final_output"]),
        )
        scored = score_agentscope_protocol(
            prediction=[],
            gt_labels=list(audit_input.gt_labels or []),
            invalid_final_output=True,
            unknown_tool_call_count=int(
                diagnostics.get("unknown_tool_call_count") or 0,
            ),
            tool_name_format_error_count=int(
                diagnostics.get("tool_name_format_error_count") or 0,
            ),
            invalid_tool_arguments_count=int(
                diagnostics.get("invalid_tool_arguments_count") or 0,
            ),
            final_protocol_error_count=len(final_errors),
        )
        process_judge = _empty_process_judge_result(
            enabled=process_judge_enabled(),
            error="protocol exception; Process Judge skipped",
        )
        adaptive = None
        overall = float(scored["score"])
        if _adaptive_reward_enabled():
            adaptive = _adaptive_reward(
                audit_input=audit_input,
                prediction=[],
                process_judge=process_judge,
                invalid_final_output=True,
                diagnostics=diagnostics,
            )
            overall = float(adaptive["final_reward"])
        metadata = {
            "score/overall": overall,
            "score/main": overall,
            "score/classification": scored["classification_score"],
            "score/protocol_penalty": scored["protocol_penalty"],
            "score/tool_protocol_penalty": scored["tool_protocol_penalty"],
            "reward/main/final": scored["classification_score"],
            "reward/detail_rule_penalty": 0.0,
            "final/nOA": scored["nOA"],
            "final/wOA": scored["wOA"],
            "final/overlap": scored["overlap"],
            "final/label_f1": scored["label_f1"],
            "agentscope/protocol_error_count": scored["protocol_error_count"],
            "agentscope/tool_protocol_error_count": scored[
                "tool_protocol_error_count"
            ],
            "agentscope/protocol_exception": 1.0,
            "agentscope/model_calls": float(model.model_call_count),
            "agentscope/thinking_enabled": float(_thinking_enabled()),
            "agentscope/gt_label_count": float(len(audit_input.gt_labels or [])),
            "agentscope/multi_gt": float(len(audit_input.gt_labels or []) > 1),
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
        }
        metadata.update(
            _protocol_metadata(
                model,
                invalid_final_output=True,
                final_protocol_error_count=len(final_errors),
            ),
        )
        if adaptive is not None:
            metadata.update(_adaptive_metadata(adaptive, process_judge))
        return {
            "metadata": metadata,
            "reward": overall,
        }

    invalid_final_output = bool(trace.get("invalid_final_output"))
    final_protocol_errors = list(trace.get("final_protocol_error_types") or [])
    diagnostics = model.protocol_diagnostics
    scored = score_agentscope_protocol(
        prediction=list(result.predict_label or []),
        gt_labels=list(audit_input.gt_labels or []),
        invalid_final_output=invalid_final_output,
        unknown_tool_call_count=int(
            diagnostics.get("unknown_tool_call_count") or 0,
        ),
        tool_name_format_error_count=int(
            diagnostics.get("tool_name_format_error_count") or 0,
        ),
        invalid_tool_arguments_count=int(
            diagnostics.get("invalid_tool_arguments_count") or 0,
        ),
        final_protocol_error_count=len(final_protocol_errors),
    )
    predicted_label = (result.predict_label or [""])[0]
    detail_rule_missing = float(
        not invalid_final_output
        and predicted_label != "通过"
        and not bool(trace.get("final_label_detail_loaded"))
    )
    detail_penalty = min(
        0.1,
        max(
            0.0,
            float(os.getenv("AUDIT_AGENTSCOPE_DETAIL_RULE_PENALTY", "0")),
        ),
    ) * detail_rule_missing
    process_judge = await _evaluate_process_reward(
        audit_input=audit_input,
        result=result,
        trace=trace,
    )
    adaptive = None
    if _adaptive_reward_enabled():
        adaptive = _adaptive_reward(
            audit_input=audit_input,
            prediction=list(result.predict_label or []),
            process_judge=process_judge,
            invalid_final_output=invalid_final_output,
            diagnostics=diagnostics,
        )
        reward_before_detail = float(adaptive["final_reward"])
    else:
        reward_before_detail = float(scored["score"])
    overall = max(
        _env_float("AUDIT_AGENTSCOPE_REWARD_FLOOR", -1.0),
        reward_before_detail - detail_penalty,
    )

    # Keep metadata scalar-only. Relax materializes these values into tensors.
    metadata = {
        "score/overall": overall,
        "score/main": overall,
        "score/classification": scored["classification_score"],
        "score/protocol_penalty": scored["protocol_penalty"],
        "score/tool_protocol_penalty": scored["tool_protocol_penalty"],
        "reward/main/final": scored["classification_score"],
        "reward/detail_rule_penalty": detail_penalty,
        "final/nOA": scored["nOA"],
        "final/wOA": scored["wOA"],
        "final/overlap": scored["overlap"],
        "final/label_f1": scored["label_f1"],
        "agentscope/protocol_error_count": scored["protocol_error_count"],
        "agentscope/tool_protocol_error_count": scored[
            "tool_protocol_error_count"
        ],
        "agentscope/protocol_exception": 0.0,
        "agentscope/model_calls": float(trace.get("model_calls") or 0),
        "agentscope/tool_call_count": float(len(trace.get("tool_calls") or [])),
        "agentscope/detail_rule_call_count": float(
            trace.get("detail_rule_call_count") or 0
        ),
        "agentscope/final_label_detail_loaded": float(
            bool(trace.get("final_label_detail_loaded"))
        ),
        "agentscope/detail_rule_missing": detail_rule_missing,
        "agentscope/experience_enabled": float(
            bool(trace.get("experience_enabled"))
        ),
        "agentscope/experience_chars": float(trace.get("experience_chars") or 0),
        "agentscope/thinking_enabled": float(_thinking_enabled()),
        "agentscope/context_compression_enabled": float(
            bool(trace.get("context_compression_enabled"))
        ),
        "agentscope/context_compression_trigger_tokens": float(
            trace.get("context_compression_trigger_tokens") or 0
        ),
        "agentscope/context_compression_count": float(
            trace.get("context_compression_count") or 0
        ),
        "agentscope/gt_label_count": float(len(audit_input.gt_labels or [])),
        "agentscope/multi_gt": float(len(audit_input.gt_labels or []) > 1),
        "prompt_tokens": float(trace.get("prompt_tokens") or 0),
        "completion_tokens": float(trace.get("completion_tokens") or 0),
    }
    metadata.update(
        _protocol_metadata(
            model,
            invalid_final_output=invalid_final_output,
            final_protocol_error_count=len(final_protocol_errors),
        ),
    )
    if adaptive is not None:
        metadata.update(_adaptive_metadata(adaptive, process_judge))
    # No explicit messages are written. Relax implicitly exports the unique
    # leaf in the managed session's captured chat forest.
    return {
        "metadata": metadata,
        "reward": overall,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one AgentScope audit session under Relax.",
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = asyncio.run(run_session(_read_json(args.input_json)))
    _write_json(args.output_json, output)


if __name__ == "__main__":
    main()
