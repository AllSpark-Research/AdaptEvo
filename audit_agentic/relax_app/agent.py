"""Relax agent app for audit_agentic (aligned with current inference flow).

Mirrors examples/dapo_ma/app/agent.py: one managed session reads
RELAX_INPUT_JSON, runs the audit workflow against the Relax agentic chat
endpoint (via AsyncOpenAI), and writes explicit JSONL export records to
RELAX_OUTPUT_JSON.

Two exported agents. Router and final share the same main agent only when the
candidate set is larger than the bypass threshold:

  * ``main``    — large queues use router + final continuation; queues with at
    most ``AUDIT_ROUTER_BYPASS_MAX_LABELS`` candidates skip router and export
    only the standalone final turn.

  * ``planner`` — single-turn:
      [seed..., user(planner_prompt), asst(planner_json)]
    Score = r_planner.

Pipeline (matches the inference pipeline used to produce traces/full_5999):
  1. Build AuditInput.from_data_row(session_input)
  2. Bypass router for small queues; otherwise main router selects 8-12 labels
  3. ZeusRuleRetriever.retrieve(source, shortlist) → rules
  4. planner with note + shortlist + full rules + tool briefs
     → tool_required_labels(required_tools) / possible_labels
  5. Aggregate required_tools and execute the deduplicated tool calls
     → tool_observations
  6. Filter rules to "surviving" labels (possible + tool_required)
  7. main agent turn 2 (final) — CONTINUATION of router turn

Env vars:
  OPENAI_BASE_URL   - relax agentic chat endpoint
  OPENAI_API_KEY    - (defaults to "dummy")
  RELAX_MODEL       - model id (defaults to "model")
  AUDIT_MAIN_LLM_MAX_TOKENS / AUDIT_PLANNER_LLM_MAX_TOKENS
                    - role-specific max tokens for internal chat calls.
  APPLY_CHAT_TEMPLATE_KWARGS / AUDIT_CHAT_TEMPLATE_KWARGS
                    - forwarded to SGLang chat_template_kwargs; defaults to {"enable_thinking": false}.
  AUDIT_LLM_MAX_TOKENS / ROLLOUT_MAX_RESPONSE_LEN
                    - fallback max tokens for each internal chat call (default 8192)
  RELAX_VERBOSE     - "1" to print per-stage timing to stderr (default off)
  AUDIT_TOOL_CONCURRENCY - per-session tool call concurrency (default 4)
  AUDIT_STRUCTURED_OUTPUT - use dynamic JSON Schema outputs (default on)
  AUDIT_ROUTER_BYPASS_MAX_LABELS - bypass threshold (default 8)
  AUDIT_LLM_TEMPERATURE - internal role-call sampling temperature (default 0.7)
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from openai import APIStatusError, AsyncOpenAI

# Re-use audit_agentic core
from audit_agentic.agents.base import _extract_first_json_object, _strip_json_fence
from audit_agentic.environment.rule_retriever import (
    OnlineRuleRetriever,
    QUEUE_NOTICE_LABEL,
    retrieve_rules_with_id_fallback,
)
from audit_agentic.environment.audit_experience import (
    load_audit_experience,
    planner_experience_enabled,
)
from audit_agentic.environment.tool_executor import ToolExecutor
from audit_agentic.environment.tool_registry import DEFAULT_TOOL_REGISTRY, list_tool_briefs
from audit_agentic.environment.tool_render import render_observations
from audit_agentic.eval.parsing import parse_answer_labels
from audit_agentic.prompts.loader import default_loader
from audit_agentic.rewards import (
    combine_main_score,
    reward_final,
    reward_planner_v5,
    reward_recall_v5,
)
from audit_agentic.agents.multimodal import make_multimodal_message, DEFAULT_IMAGE_MAX_TOKENS, collect_rule_images
from audit_agentic.schemas import AuditInput, RuleInfo, ToolCall
from audit_agentic.structured_output import (
    build_final_candidate_labels,
    final_output_quality,
    final_response_format,
    planner_output_quality,
    planner_response_format,
    router_bypass_max_labels,
    router_max_labels,
    router_min_labels,
    router_output_quality,
    router_response_format,
    structured_output_enabled,
)


# ---- helpers ---------------------------------------------------------------


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_jsonl(path: str | Path, records: List[Dict[str, Any]]) -> None:
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    Path(path).write_text(payload + "\n", encoding="utf-8")


def _planner_bucket_slice(text: str, key: str, end_keys: List[str]) -> str:
    start = text.find(f'"{key}"')
    if start < 0:
        start = text.find(key)
    if start < 0:
        return ""
    ends = [text.find(f'"{k}"', start + len(key)) for k in end_keys]
    ends += [text.find(k, start + len(key)) for k in end_keys]
    ends = [x for x in ends if x >= 0]
    end = min(ends) if ends else len(text)
    return text[start:end]


def _salvage_planner_labels(text: str) -> Dict[str, Any]:
    """Best-effort planner parse when evidence strings break JSON quoting.

    This intentionally recovers only labels/opinions for reward/downstream
    routing. The caller still marks json_valid=0 via the _parse_fallback flag.
    """

    def entries(segment: str) -> List[Dict[str, Any]]:
        matches = list(re.finditer(r'"label"\s*:\s*"([^"]+)"', segment))
        out: List[Dict[str, Any]] = []
        seen = set()
        for i, match in enumerate(matches):
            label = match.group(1).strip()
            if not label or label in seen:
                continue
            next_start = matches[i + 1].start() if i + 1 < len(matches) else len(segment)
            chunk = segment[match.end():next_start]
            opinion = ""
            opinion_match = re.search(r'"preliminary_opinion"\s*:\s*"([^"]+)"', chunk)
            if opinion_match:
                opinion = opinion_match.group(1).strip()
            item: Dict[str, Any] = {"label": label}
            if opinion:
                item["preliminary_opinion"] = opinion
            out.append(item)
            seen.add(label)
        return out

    possible = entries(_planner_bucket_slice(text, "possible_labels", ["tool_required_labels", "tool_calls"]))
    tool_required = entries(_planner_bucket_slice(text, "tool_required_labels", ["tool_calls"]))
    if not possible and not tool_required:
        return {}
    return {
        "_parse_fallback": "planner_label_salvage",
        "filtered_labels": {
            "possible_labels": possible,
            "tool_required_labels": tool_required,
        },
        "tool_calls": [],
    }


def _parse_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    candidate = _strip_json_fence(text).replace("<|im_end|>", "").strip()
    try:
        return json.loads(candidate)
    except Exception:
        pass
    obj = _extract_first_json_object(candidate)
    if obj:
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict):
                parsed.setdefault("_parse_fallback", "json_object_extract")
            return parsed
        except Exception:
            salvage = _salvage_planner_labels(obj)
            if salvage:
                return salvage
            return {}
    salvage = _salvage_planner_labels(candidate)
    if salvage:
        return salvage
    return {}


def _planner_label(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("label", "")).strip()


def _dedupe_planner_label_sections(
    possible_labels: List[Dict[str, Any]],
    tool_required_labels: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep each planner label once before feeding main_final.

    Tool-required entries win over possible entries for the same label because
    they indicate the final turn should inspect tool evidence for that label.
    """
    seen = set()
    dedup_tool_required: List[Dict[str, Any]] = []
    for entry in tool_required_labels:
        label = _planner_label(entry)
        if not label or label in seen:
            continue
        dedup_tool_required.append(entry)
        seen.add(label)

    dedup_possible: List[Dict[str, Any]] = []
    for entry in possible_labels:
        label = _planner_label(entry)
        if not label or label in seen:
            continue
        dedup_possible.append(entry)
        seen.add(label)

    return dedup_possible, dedup_tool_required


def _max_tokens_for_role(role: str) -> int:
    if role == "planner":
        raw = os.getenv("AUDIT_PLANNER_LLM_MAX_TOKENS")
    else:
        raw = os.getenv("AUDIT_MAIN_LLM_MAX_TOKENS")
    raw = raw or os.getenv("AUDIT_LLM_MAX_TOKENS") or os.getenv("ROLLOUT_MAX_RESPONSE_LEN") or "8192"
    return int(raw)


def _chat_template_kwargs() -> Dict[str, Any]:
    raw = os.getenv("AUDIT_CHAT_TEMPLATE_KWARGS") or os.getenv("APPLY_CHAT_TEMPLATE_KWARGS")
    if raw is None:
        return {"enable_thinking": False}
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {"enable_thinking": False}
    return parsed if isinstance(parsed, dict) else {"enable_thinking": False}


async def _chat_once(
    client: AsyncOpenAI,
    messages: List[Dict[str, Any]],
    *,
    role: str = "main",
    response_format: Optional[Dict[str, Any]] = None,
) -> tuple[str, Dict[str, Any]]:
    model = os.getenv("RELAX_MODEL", "model")
    max_tokens = _max_tokens_for_role(role)
    chat_kwargs = _chat_template_kwargs()
    extra_body = {"chat_template_kwargs": chat_kwargs} if chat_kwargs else None
    call_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": float(os.getenv("AUDIT_LLM_TEMPERATURE", "0.7")),
        "extra_body": extra_body,
    }
    if response_format is not None:
        call_kwargs["response_format"] = response_format
    resp = await client.chat.completions.create(
        **call_kwargs,
    )
    usage = resp.usage.model_dump() if resp.usage is not None else {}
    content = resp.choices[0].message.content or ""
    return content, usage


# ---- per-role wrappers -----------------------------------------------------


class RoleAgent:
    """Each role keeps its own messages history. For main_final we pre-seed
    the router round into its history so its export sample is a continuation.
    """

    def __init__(self, *, name: str, client: AsyncOpenAI, seed_messages: List[Dict[str, Any]]):
        self.name = name
        self.client = client
        self.messages: List[Dict[str, Any]] = copy.deepcopy(seed_messages)
        self.last_usage: Dict[str, Any] = {}
        self.raw: str = ""

    async def call(
        self,
        prompt: str,
        *,
        images: list | None = None,
        image_max_tokens: int = DEFAULT_IMAGE_MAX_TOKENS,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        msg = make_multimodal_message("user", prompt, images, image_max_tokens)
        self.messages.append(msg)
        try:
            content, usage = await _chat_once(
                self.client,
                self.messages,
                role=self.name,
                response_format=response_format,
            )
        except APIStatusError:
            self.messages.pop()
            raise
        self.messages.append({"role": "assistant", "content": content})
        self.raw = content
        self.last_usage = usage
        return content

    def record(self, **metadata: Any) -> Dict[str, Any]:
        # IMPORTANT: relax's dict_to_tensordict will torch.tensor() every
        # metadata value. Only put scalars (int/float/None) here — NO dicts,
        # lists, or strings (except 'role' which relax handles specially).
        # The original `usage` dict is flattened to scalar fields.
        usage = self.last_usage or {}
        meta = {
            "role": self.name,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            **metadata,
        }
        record = {"name": self.name, "messages": self.messages, "metadata": meta}
        score_key = "score/overall" if "score/overall" in meta else f"score/{self.name}"
        if score_key in meta:
            # reward MUST be a float, not {"score": float} — relax's
            # get_reward_value returns self.reward as-is when reward_key is
            # unset, and dict_to_tensordict chokes on dict rewards.
            record["reward"] = float(meta[score_key])
        return record


# ---- main session ----------------------------------------------------------


async def run_session(
    session_input: Dict[str, Any],
    *,
    retriever: Optional[OnlineRuleRetriever] = None,
    loader=None,
    client: Optional[AsyncOpenAI] = None,
    tool_registry: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """One audit session = router → retrieve rules → planner → tools → main_final.

    Optional shared resources (recommended for batch benchmarks):
      * ``retriever`` — ZeusRuleRetriever instance with warmed cache. If None,
        a fresh one is created (and will lazy-load the rule index on first
        retrieve(), ~30s). Sharing avoids re-loading 348K trees N times.
      * ``loader`` — PromptTemplateLoader (cheap to construct, but sharing saves
        some Jinja env init per call).
      * ``client`` — AsyncOpenAI client. Sharing pools HTTP connections.
      * ``tool_registry`` — defaults to DEFAULT_TOOL_REGISTRY.

    session_input expected shapes (both supported):

    1) Cleaned-data row layout (preferred):
       ``{"note": "...", "images": [...],
          "metadata": {"note_id": ..., "source": ..., "candidate_labels": [...],
                       "labels": [...], ...}}``
       Will be passed through ``AuditInput.from_data_row``.

    2) Pre-built input wrapper:
       ``{"messages": [...seed messages...],
          "metadata": {"audit_input": {...AuditInput fields...}}}``
    """
    import time as _t
    import sys as _sys
    _t0 = _t.time()
    _verbose = os.environ.get("RELAX_VERBOSE", "0").lower() in ("1", "true", "yes")
    def _log(msg):
        if _verbose:
            print(f"[relax_app +{_t.time()-_t0:6.1f}s] {msg}", file=_sys.stderr, flush=True)

    audit_input = AuditInput.from_data_row(session_input)
    seed_messages = copy.deepcopy(session_input.get("messages") or [])
    _log(f"audit_input built: note_id={audit_input.note_id} source={audit_input.source_id} candidates={len(audit_input.candidate_labels)}")

    if loader is None:
        loader = default_loader()
    if client is None:
        client_timeout = float(os.environ.get("AUDIT_AGENT_CLIENT_TIMEOUT", "900"))
        connect_timeout = float(os.environ.get("AUDIT_AGENT_CONNECT_TIMEOUT", "30"))
        client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            timeout=httpx.Timeout(timeout=client_timeout, connect=connect_timeout),
        )
    if tool_registry is None:
        tool_registry = DEFAULT_TOOL_REGISTRY
    if retriever is None:
        _log("creating OnlineRuleRetriever (Mars Jupiter API) ...")
        retriever = OnlineRuleRetriever()
    executor = ToolExecutor(tool_registry).with_defaults(
        note_id=audit_input.note_id,
        source_id=audit_input.source_id,
    )
    _log("rule retriever + tool executor ready")

    # ───────────── stage 1: conditional main-agent router turn ─────────────
    use_schema = structured_output_enabled()
    router_invoked = len(audit_input.candidate_labels) > router_bypass_max_labels()
    image_max_tokens = int(os.environ.get("IMAGE_MAX_TOKENS", str(DEFAULT_IMAGE_MAX_TOKENS)))
    main = RoleAgent(name="main", client=client, seed_messages=seed_messages)
    cand_set = set(audit_input.candidate_labels)
    router_raw = ""
    router_usage: Dict[str, Any] = {}
    answer_text = ""
    non_cand: List[str] = []
    fuzzy: List[Dict[str, Any]] = []
    router_quality = router_output_quality("", {}, [], 0)
    if router_invoked:
        _log("STAGE 1: main agent router turn")
        router_template = os.environ.get(
            "AUDIT_MAIN_ROUTER_TEMPLATE",
            "main_router_schema" if use_schema else "main_router",
        )
        router_prompt = loader.render(
            router_template,
            note=audit_input.note,
            candidate_labels=audit_input.candidate_labels,
        )
        await main.call(
            router_prompt,
            images=audit_input.images,
            image_max_tokens=image_max_tokens,
            response_format=(
                router_response_format(audit_input.candidate_labels)
                if use_schema
                else None
            ),
        )
        router_raw = main.raw
        router_usage = main.last_usage
        _log(f"  router done, raw_chars={len(router_raw)}")
        if use_schema:
            router_parsed = _parse_json(router_raw)
            answer_text = str(router_parsed.get("brief_analysis", ""))
            raw_router_labels = list(router_parsed.get("shortlist_labels") or [])
            parsed_labels = raw_router_labels
            non_cand = [label for label in parsed_labels if label not in cand_set]
            shortlist = list(dict.fromkeys(label for label in parsed_labels if label in cand_set))
            parse_source = "json_schema" if router_parsed else "json_parse_failed"
            router_quality = router_output_quality(
                router_raw,
                router_parsed,
                raw_router_labels,
                min(router_min_labels(), len(audit_input.candidate_labels)),
            )
        else:
            parsed_labels, answer_text, parse_source, non_cand, fuzzy = parse_answer_labels(
                router_raw, audit_input.candidate_labels
            )
            shortlist = [label for label in parsed_labels if label in cand_set]
        if not shortlist:
            shortlist = list(audit_input.candidate_labels[: router_max_labels()])
            parse_source = "router_fallback_first_candidates"
    else:
        _log("STAGE 1: bypass router and keep all candidate labels")
        shortlist = list(audit_input.candidate_labels)
        answer_text = "候选标签数量较少，跳过 Router 并保留全部候选标签。"
        parse_source = "bypass_all_candidates"

    # ───────────── stage 2: retrieve rules (env, not LLM) ─────────────
    _log("STAGE 2: retrieve rules (will lazy-load index if first call)")
    rules: List[RuleInfo] = retrieve_rules_with_id_fallback(retriever, audit_input, shortlist)
    _log(f"  rules retrieved, n={len(rules)} total_chars={sum(len(r.rule_text) for r in rules)}")

    # ───────────── stage 3: planner ─────────────
    _log("STAGE 3: planner LLM call")
    available_tools = list_tool_briefs()
    if audit_input.available_tools:
        allowed = set(audit_input.available_tools)
        available_tools = [t for t in available_tools if t["name"] in allowed]

    use_planner_experience = planner_experience_enabled(audit_input.source_id)
    planner_audit_experience = (
        load_audit_experience(
            audit_input.source_id,
            include_all_labels=True,
        )
        if use_planner_experience
        else ""
    )
    planner_prompt = loader.render(
        os.environ.get(
            "AUDIT_PLANNER_TEMPLATE",
            "planner_schema" if use_schema else "planner",
        ),
        note=audit_input.note,
        shortlist_labels=shortlist,
        shortlist_rules=[r.model_dump() for r in rules],
        available_tools=available_tools,
        audit_experience=planner_audit_experience,
    )
    planner = RoleAgent(name="planner", client=client, seed_messages=seed_messages)
    await planner.call(
        planner_prompt,
        images=list(audit_input.images or []) + collect_rule_images(rules),
        image_max_tokens=image_max_tokens,
        response_format=(
            planner_response_format(shortlist, [tool["name"] for tool in available_tools])
            if use_schema
            else None
        ),
    )
    _log(f"  planner done, raw_chars={len(planner.raw)}")

    planner_parsed = _parse_json(planner.raw)
    planner_quality = (
        planner_output_quality(planner.raw, planner_parsed)
        if use_schema
        else planner_output_quality("", {})
    )
    filtered = planner_parsed.get("filtered_labels") or {}
    possible_labels_raw = filtered.get("possible_labels") or []
    tool_required_labels_raw = filtered.get("tool_required_labels") or []
    possible_labels_raw, tool_required_labels_raw = _dedupe_planner_label_sections(
        possible_labels_raw,
        tool_required_labels_raw,
    )
    tool_names = {t["name"] for t in available_tools}
    tool_calls: List[ToolCall] = []
    seen_tool_names = set()
    for label_entry in tool_required_labels_raw:
        if not isinstance(label_entry, dict):
            continue
        label = str(label_entry.get("label", "")).strip()
        for tool_name in label_entry.get("required_tools") or []:
            if tool_name not in tool_names or tool_name in seen_tool_names:
                continue
            seen_tool_names.add(tool_name)
            tool_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    args={},
                    reason=f"required by label: {label}",
                )
            )

    # Legacy non-schema prompts may still emit the old top-level tool_calls.
    if not use_schema and not tool_calls:
        for call in planner_parsed.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            tool_name = call.get("tool_name")
            if tool_name not in tool_names or tool_name in seen_tool_names:
                continue
            seen_tool_names.add(tool_name)
            tool_calls.append(ToolCall(tool_name=tool_name, args={}))

    # ───────────── stage 4: tools (env, args auto-injected) ─────────────
    _log("STAGE 4: tools (local)")
    tool_concurrency = int(os.environ.get("AUDIT_TOOL_CONCURRENCY", "4"))
    observations = executor.execute(tool_calls, concurrency=tool_concurrency)
    _log(f"  tool_calls={len(tool_calls)} observations={len(observations)} concurrency={tool_concurrency}")

    # ───────────── stage 5: main agent — final turn (continuation) ─────────────
    final_candidate_labels = build_final_candidate_labels(
        audit_input.candidate_labels,
        shortlist,
        possible_labels_raw,
        tool_required_labels_raw,
    )
    surviving = set(final_candidate_labels)
    if os.environ.get("AUDIT_FINAL_KEEP_ALL_ROUTER_RULES", "").strip().lower() in {"1", "true", "yes", "on"}:
        remaining_rules = rules
    else:
        remaining_rules = [r for r in rules if r.label in surviving or r.label == QUEUE_NOTICE_LABEL]

    # Render tool observations → (tool_section text with <image>, aligned images).
    # Tool images ride their own <image> placeholders inside the continuation
    # message so the main agent actually SEES them (previously they were dropped
    # by replacing <image> with the literal "图片").
    tool_section, tool_images = render_observations(observations, tool_registry)
    rule_images = collect_rule_images(remaining_rules)

    final_audit_experience = load_audit_experience(
        audit_input.source_id,
        labels=final_candidate_labels,
    )
    continuation_prompt = loader.render(
        os.environ.get(
            "AUDIT_MAIN_FINAL_TEMPLATE",
            "main_final_schema" if use_schema else "main_final",
        ),
        standalone_mode=not router_invoked,
        note=audit_input.note,
        final_candidate_labels=final_candidate_labels,
        possible_labels=possible_labels_raw,
        tool_required_labels=tool_required_labels_raw,
        remaining_rules=[r.model_dump() for r in remaining_rules],
        tool_section=tool_section,
        judge_feedback=None,
        audit_experience=final_audit_experience,
    )
    _log(
        "STAGE 5: main agent final turn "
        + ("(continuation)" if router_invoked else "(standalone)")
    )
    final_images = rule_images + tool_images
    if not router_invoked:
        final_images = list(audit_input.images or []) + final_images
    await main.call(
        continuation_prompt,
        images=final_images,
        image_max_tokens=image_max_tokens,
        response_format=(
            final_response_format(final_candidate_labels)
            if use_schema
            else None
        ),
    )
    final_raw = main.raw
    final_usage = main.last_usage
    _log(f"  final done, raw_chars={len(final_raw)}")

    final_parsed = _parse_json(final_raw)
    final_quality = (
        final_output_quality(final_raw, final_parsed)
        if use_schema
        else final_output_quality("", {})
    )

    # ───────────── rewards (latest: final-only score minus parse penalties) ─────────────
    gt = audit_input.gt_labels or []
    r_recall = reward_recall_v5(
        shortlist=shortlist,
        gt_labels=gt,
        candidate_labels=audit_input.candidate_labels,
    )
    r_final = reward_final(
        predict_label=final_parsed.get("predict_label", []),
        gt_labels=gt,
        mode="hybrid",
    )
    r_main = combine_main_score(r_recall, r_final)

    # V5: pass raw dicts (for preliminary_opinion check) + format info
    _planner_json_valid = bool(planner_parsed) and not planner_parsed.get("_parse_fallback")
    _planner_has_filtered = "filtered_labels" in planner_parsed
    # Structured mode derives calls from required_tools and intentionally has
    # no redundant top-level tool_calls field.
    _planner_has_tool_calls = use_schema or "tool_calls" in planner_parsed

    r_planner = reward_planner_v5(
        possible_labels_raw=possible_labels_raw,
        tool_required_labels_raw=tool_required_labels_raw,
        shortlist=shortlist,
        gt_labels=gt,
        tools_called=len(observations),
        tools_total_available=len(available_tools),
        json_valid=_planner_json_valid,
        has_filtered=_planner_has_filtered,
        has_tool_calls=_planner_has_tool_calls,
    )
    expected_router_parse = "json_schema" if use_schema else "answer_tag"
    router_format_error = float(
        router_invoked
        and (parse_source != expected_router_parse or bool(non_cand) or bool(fuzzy))
    )
    planner_format_error = float((not planner_parsed) or bool(planner_parsed.get("_parse_fallback")))
    final_format_error = float((not final_parsed) or bool(final_parsed.get("_parse_fallback")))
    format_error_count = router_format_error + planner_format_error + final_format_error
    parse_format_penalty = min(0.3, 0.1 * format_error_count)
    router_quality_penalty = router_quality["penalty"] if router_invoked else 0.0
    planner_quality_penalty = planner_quality["penalty"]
    final_quality_penalty = final_quality["penalty"]
    structured_quality_penalty = min(
        0.05,
        router_quality_penalty + planner_quality_penalty + final_quality_penalty,
    )
    format_penalty = parse_format_penalty + structured_quality_penalty
    r_overall = max(0.0, min(1.0, float(r_final["score"]) - format_penalty))

    # ───────────── export 2 role samples (main + planner) ─────────────
    # IMPORTANT: every metadata value must be a flat scalar (int / float / None).
    # relax's dict_to_tensordict calls torch.tensor() on each value; dicts,
    # strings, or lists of dicts crash with "Could not infer dtype of dict".
    records = [
        main.record(
            **{
                "score/overall": r_overall,
                "score/main": r_overall,
                "reward/main/combined_legacy": r_main["score"],
                "reward/main/recall_v5": r_recall["score"],
                "reward/main/final": r_final["score"],
                "reward/main/final_after_format": r_overall,
                "reward/format_penalty": format_penalty,
                "reward/parse_format_penalty": parse_format_penalty,
                "reward/router_quality_penalty": router_quality_penalty,
                "reward/planner_quality_penalty": planner_quality_penalty,
                "reward/final_quality_penalty": final_quality_penalty,
                "reward/structured_quality_penalty": structured_quality_penalty,
                "reward/format_error_count": format_error_count,
                "format/router_error": router_format_error,
                "format/planner_error": planner_format_error,
                "format/final_error": final_format_error,
                "format/router_parse_source_fallback": float(
                    router_invoked and parse_source != expected_router_parse
                ),
                "router/invoked": float(router_invoked),
                "format/router_non_candidate_count": float(len(non_cand)),
                "format/router_fuzzy_count": float(len(fuzzy)),
                "format/router_duplicate_excess": router_quality["duplicate_excess"],
                "format/router_missing_unique_count": router_quality["missing_unique_count"],
                "format/router_raw_item_count": router_quality["raw_item_count"],
                "format/router_unique_item_count": router_quality["unique_item_count"],
                "format/router_abnormal_whitespace": router_quality["abnormal_whitespace"],
                "format/router_max_newline_run": router_quality["max_newline_run"],
                "format/planner_duplicate_excess": planner_quality["duplicate_excess"],
                "format/planner_cross_bucket_duplicate_count": planner_quality["cross_bucket_duplicate_count"],
                "format/planner_abnormal_whitespace": planner_quality["abnormal_whitespace"],
                "format/planner_max_newline_run": planner_quality["max_newline_run"],
                "format/final_duplicate_excess": final_quality["duplicate_excess"],
                "format/final_abnormal_whitespace": final_quality["abnormal_whitespace"],
                "format/final_max_newline_run": final_quality["max_newline_run"],
                # router sub-metrics (V5)
                "router/is_gt_pass": r_recall["is_gt_pass"],
                "router/has_pass": r_recall["has_pass"],
                "router/gt_recall": r_recall["gt_recall"],
                "router/gt_any_hit": r_recall["gt_any_hit"],
                "router/pass_hit": r_recall["pass_hit"],
                "router/pass_rank_bonus": r_recall["pass_rank_bonus"],
                "router/pass_penalty": r_recall["pass_penalty"],
                "router/size_penalty": r_recall["size_penalty"],
                "router/extra_size_penalty": r_recall["extra_size_penalty"],
                "router/too_few_penalty": r_recall["too_few_penalty"],
                "router/invalid_cnt": r_recall["invalid_cnt"],
                "router/invalid_penalty": r_recall["invalid_penalty"],
                "router/shortlist_size": r_recall["shortlist_size"],
                "router/out_min": r_recall["out_min"],
                "router/out_max": r_recall["out_max"],
                "router/gt_rank_bonus": r_recall["gt_rank_bonus"],
                "router/raw_score": r_recall["raw_score"],
                "router/raw_max": r_recall["raw_max"],
                "router/raw_min": r_recall["raw_min"],
                # final sub-metrics (unchanged from V2)
                "final/nOA": r_final["nOA"],
                "final/wOA": r_final["wOA"],
                "final/overlap": r_final["overlap"],
                "final/label_f1": r_final["label_f1"],
                # token accounting
                "completion_tokens/router_sub": router_usage.get("completion_tokens"),
                "completion_tokens/final_sub": final_usage.get("completion_tokens"),
                "experience/final_enabled": float(bool(final_audit_experience)),
                "experience/final_chars": float(len(final_audit_experience)),
                "experience/planner_enabled": float(bool(planner_audit_experience)),
                "experience/planner_chars": float(len(planner_audit_experience)),
            },
        ),
        planner.record(
            **{
                "score/overall": r_overall,
                "score/planner": r_planner["score"],
                "reward/main/final": r_final["score"],
                "reward/main/final_after_format": r_overall,
                "reward/format_penalty": format_penalty,
                "reward/parse_format_penalty": parse_format_penalty,
                "reward/router_quality_penalty": router_quality_penalty,
                "reward/planner_quality_penalty": planner_quality_penalty,
                "reward/final_quality_penalty": final_quality_penalty,
                "reward/structured_quality_penalty": structured_quality_penalty,
                "reward/format_error_count": format_error_count,
                "format/router_error": router_format_error,
                "format/planner_error": planner_format_error,
                "format/final_error": final_format_error,
                # V5: diagnostic + format
                "planner/is_gt_pass": r_planner["is_gt_pass"],
                "planner/format_score": r_planner["format_score"],
                "planner/json_valid": r_planner["json_valid"],
                # shared counters
                "planner/possible_cnt": r_planner["possible_cnt"],
                "planner/tool_required_cnt": r_planner["tool_required_cnt"],
                "planner/rule_out_cnt": r_planner["rule_out_cnt"],
                "planner/kept_cnt": r_planner["kept_cnt"],
                "planner/shortlist_cnt": r_planner["shortlist_cnt"],
                "planner/tools_called": r_planner["tools_called"],
                # pass-case sub-metrics (V5)
                "planner/possible_violation_cnt": r_planner.get("possible_violation_cnt", 0.0),
                "planner/non_support_violation_cnt": r_planner.get("non_support_violation_cnt", 0.0),
                "planner/support_violation_cnt": r_planner["support_violation_cnt"],
                "planner/support_penalty": r_planner["support_penalty"],
                "planner/possible_violation_penalty": r_planner.get("possible_violation_penalty", r_planner["support_penalty"]),
                # violation-case sub-metrics (V5)
                "planner/gt_survival": r_planner["gt_survival"],
                "planner/pass_rule_out_bonus": r_planner["pass_rule_out_bonus"],
                "planner/pass_kept_penalty": r_planner["pass_kept_penalty"],
                "planner/compact_kept": r_planner["compact_kept"],
                "planner/raw_score": r_planner["raw_score"],
                "planner/raw_max": r_planner["raw_max"],
                "planner/raw_min": r_planner["raw_min"],
                "experience/final_enabled": float(bool(final_audit_experience)),
                "experience/final_chars": float(len(final_audit_experience)),
                "experience/planner_enabled": float(bool(planner_audit_experience)),
                "experience/planner_chars": float(len(planner_audit_experience)),
            },
        ),
    ]
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one audit_agentic Relax session.")
    p.add_argument("--input-json", required=True)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = asyncio.run(run_session(_read_json(args.input_json)))
    _write_jsonl(args.output_json, records)


if __name__ == "__main__":
    main()
