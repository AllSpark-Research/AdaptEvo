"""AgentScope runtime adapter for the audit agent.

This module keeps the existing audit tool registry, ToolExecutor, cache and
multimodal rendering contract intact. AgentScope only owns the native
reasoning/tool-call loop and final structured output protocol.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Type

from pydantic import BaseModel, Field, create_model

from agentscope.agent import Agent, ContextConfig, InjectionConfig, ReActConfig
from agentscope.message import (
    Base64Source,
    DataBlock,
    TextBlock,
    ToolResultState,
    URLSource,
    UserMsg,
)
from agentscope.permission import (
    PermissionBehavior,
    PermissionContext,
    PermissionDecision,
)
from agentscope.state import AgentState
from agentscope.tool import ToolBase, ToolChunk, Toolkit

from ..environment.audit_experience import (
    load_audit_experience,
    load_audit_experience_for_rule_previews,
)
from ..environment.rule_retriever import QUEUE_NOTICE_LABEL
from ..environment.tool_executor import ToolExecutor
from ..environment.tool_registry import DEFAULT_TOOL_REGISTRY
from ..environment.tools.base_tool import BaseTool as AuditBaseTool
from ..schemas import AuditInput, MainFinalOutput, RuleInfo, ToolCall
from ..structured_output import normalize_decision_basis
from .multimodal import (
    DEFAULT_IMAGE_MAX_TOKENS,
    build_multimodal_content,
    collect_rule_images,
    prepare_accessible_multimodal_inputs,
    resilient_image_preparation_enabled,
)


SYSTEM_PROMPT = """你是小红书内容审核 Audit Agent。
请严格依据当前帖子、当前审核规则、审核经验和工具返回的事实完成判断。
你可以自由分析并自主调用必要工具；工具事实优先于未经证实的常识。
当证据充分后，必须使用系统提供的结构化输出工具提交最终结果。
最终依据应简洁、可复核，不要在最终字段中输出完整思维过程。
decision_basis 必须是一段普通文本，可以换行分点，不要输出 JSON 数组。"""

ROUTER_SYSTEM_PROMPT = """你是小红书内容审核 Router Agent。
你的任务是高召回筛选候选标签，不做最终裁决，也不调用业务工具。
只允许从当前候选标签中选择；边界不清、需要规则或工具确认的标签应保留。
完成分析后，必须使用系统提供的结构化输出工具提交 shortlist。"""

AUDIT_COMPRESSION_PROMPT = """<system-hint>
你正在压缩一条单 case 内容审核 Agent 的历史上下文。
这不是重新审核，也不是生成最终结论；请生成准确、可追溯、可供后续推理继续使用的审核状态快照。压缩后，较早的原始消息、图片和工具结果可能永久离开上下文，因此必须保留所有可能影响最终判定的信息。

请严格遵守：
1. 只记录上下文中实际出现的证据，不补充常识、GT、猜测或隐含事实。保留关键原文、OCR、数值、否定词、限定词、比较对象和行为主体。
2. 明确区分帖子可见证据、正式审核规则、工具客观返回、审核经验和模型暂定判断。不得把模型推测改写成帖子、规则或工具事实。
3. “未发现/未查询到/未展示”不等于确定不存在；工具失败、工具无结果和工具明确否定必须分别记录。
4. 对重点标签同时保留命中条件与豁免/排除条件，不得只摘要严格侧或宽松侧。经验只能用于尺度提醒，不能覆盖规则和工具事实。
5. 工具证据必须保留工具名、决定性参数、成功/失败状态、关键返回和相关图片证据。仍在等待或仍需补充的调用必须写明。
6. 图片即将离开上下文时，按可识别对象记录与审核有关的文字、OCR、行为、数值和视觉特征。无法确认的内容标记为不确定，不得自行补全。
7. 当前二分类和标签判断只能写成暂定状态；保留已排除标签及原因、尚未解决的冲突和仍需核验的事项，避免后续模型盲从旧结论。
8. 删除重复、寒暄和无关过程，但冲突证据必须两边都保留。摘要必须自包含，使用明确的 source_id、note_id、row_idx、标签名和工具名，禁止使用无法脱离旧上下文理解的指代。
</system-hint>"""

AUDIT_COMPRESSION_TEMPLATE = """<system-info>
以下是当前单条审核 case 的历史状态快照。它用于恢复上下文，不等同于最终结论；后续判断仍须依据当前帖子、正式规则和工具事实重新核验。

# Case Anchor
{case_anchor}

# Visible Evidence
{visible_evidence}

# Rule Boundaries
{rule_boundaries}

# Tool Evidence
{tool_evidence}

# Provisional Decision State
{decision_state}

# Unresolved Conflicts
{unresolved_conflicts}

# Required Next Actions
{next_actions}
</system-info>"""


class AuditCompressionSummary(BaseModel):
    """Structured state preserved when compressing an audit trajectory."""

    case_anchor: str = Field(
        description=(
            "source_id, note_id, row_idx when present; audit subject, complete "
            "candidate labels, and output constraints."
        ),
    )
    visible_evidence: str = Field(
        description=(
            "Decision-relevant title, body, OCR, image, video, comment, topic, "
            "and account evidence. Preserve key wording, qualifiers and numbers."
        ),
    )
    rule_boundaries: str = Field(
        description=(
            "For each reviewed or important label: applicable hit conditions, "
            "exemptions/exclusions, and whether current evidence supports either."
        ),
    )
    tool_evidence: str = Field(
        description=(
            "Tool name, decisive arguments, success/failure/no-result state, key "
            "objective returns, image evidence, and pending tool work."
        ),
    )
    decision_state: str = Field(
        description=(
            "Provisional binary tendency and label candidates, plus excluded "
            "labels and exact reasons. Never present this as a final verdict."
        ),
    )
    unresolved_conflicts: str = Field(
        description=(
            "Unresolved conflicts among visible evidence, rules, exemptions, "
            "tools, experience, and prior model assumptions; use 'none' if absent."
        ),
    )
    next_actions: str = Field(
        description=(
            "Specific rules, tools, evidence, or boundary questions that still "
            "must be checked before the final structured audit result."
        ),
    )


def parse_final_structured_output(
    raw_structured: Any,
    candidate_labels: List[str],
) -> tuple[Dict[str, Any], str, List[str]]:
    """Parse final output without turning model protocol errors into infra errors."""
    if isinstance(raw_structured, dict):
        structured = dict(raw_structured)
    elif hasattr(raw_structured, "model_dump"):
        try:
            structured = dict(raw_structured.model_dump())
        except (TypeError, ValueError):
            structured = {}
    else:
        structured = {}

    label = str(structured.get("predict_label") or "").strip()
    error_types: List[str] = []
    if not structured:
        error_types.append("missing_or_malformed_structured_output")
    if not label:
        error_types.append("missing_predict_label")
    elif label not in candidate_labels:
        error_types.append("invalid_predict_label")
    return structured, label, error_types


def _media_type(value: str) -> str:
    path = value[len("file://") :] if value.startswith("file://") else value
    guessed = mimetypes.guess_type(path)[0]
    return guessed if guessed and guessed.startswith("image/") else "image/jpeg"


def _data_url_source(url: str) -> Base64Source:
    header, data = url.split(",", 1)
    media_type = header[5:].split(";", 1)[0] or "image/jpeg"
    return Base64Source(data=data, media_type=media_type)


def to_agentscope_blocks(
    text: str,
    images: List[str],
    image_max_tokens: int = DEFAULT_IMAGE_MAX_TOKENS,
) -> List[TextBlock | DataBlock]:
    """Convert the existing aligned ``<image>`` representation to blocks."""
    if resilient_image_preparation_enabled():
        prepared = prepare_accessible_multimodal_inputs(text, images, image_max_tokens)
        text, images = prepared.text, prepared.images
    content = build_multimodal_content(text, images, image_max_tokens)
    blocks: List[TextBlock | DataBlock] = []
    for item in content:
        if item.get("type") == "text":
            value = str(item.get("text") or "")
            if value.strip():
                blocks.append(TextBlock(text=value))
            continue

        if item.get("type") != "image_url":
            continue
        url = str((item.get("image_url") or {}).get("url") or "")
        if not url:
            continue
        if url.startswith("data:"):
            source = _data_url_source(url)
        else:
            source = URLSource(url=url, media_type=_media_type(url))
        blocks.append(DataBlock(source=source))
    return blocks or [TextBlock(text=text)]


async def to_agentscope_blocks_async(
    text: str,
    images: List[str],
    image_max_tokens: int = DEFAULT_IMAGE_MAX_TOKENS,
) -> List[TextBlock | DataBlock]:
    """Prepare/download/resize images without blocking the async agent loop."""
    return await asyncio.to_thread(
        to_agentscope_blocks,
        text,
        images,
        image_max_tokens,
    )


@dataclass
class ToolRunRecorder:
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def append(self, value: Dict[str, Any]) -> None:
        self.calls.append(value)


def build_compression_context_config(
    model: Any,
    trigger_tokens: int = 0,
    profile: str = "default",
) -> ContextConfig:
    """Build AgentScope compression config from an absolute token threshold.

    ``trigger_tokens <= 0`` preserves AgentScope's default 80% threshold.
    AgentScope requires at least 10% of the context for the summary response,
    so an enabled custom threshold must stay between 10% and 90%.
    """
    profile = str(profile or "default").strip().lower()
    if profile not in {"default", "audit"}:
        raise ValueError(f"Unsupported compression profile: {profile!r}")
    config_kwargs: Dict[str, Any] = {}
    if profile == "audit":
        config_kwargs.update(
            compression_prompt=AUDIT_COMPRESSION_PROMPT,
            summary_template=AUDIT_COMPRESSION_TEMPLATE,
            summary_schema=AuditCompressionSummary.model_json_schema(),
        )
    if int(trigger_tokens or 0) <= 0:
        return ContextConfig(**config_kwargs)
    context_size = int(getattr(model, "context_size", 0) or 0)
    if context_size <= 0:
        raise ValueError("Model context_size is required for token-based compression")
    ratio = int(trigger_tokens) / context_size
    if not 0.1 < ratio < 0.9:
        raise ValueError(
            "compression_trigger_tokens must be between 10% and 90% of "
            f"context_size; got {trigger_tokens}/{context_size}",
        )
    return ContextConfig(trigger_ratio=ratio, **config_kwargs)


class ConfigurableCompressionAgent(Agent):
    """AgentScope Agent with an explicit compression on/off switch."""

    def __init__(self, *args: Any, compression_enabled: bool = True, **kwargs: Any):
        self.compression_enabled = bool(compression_enabled)
        self.compression_count = 0
        super().__init__(*args, **kwargs)

    async def compress_context(self, *args: Any, **kwargs: Any) -> None:
        if not self.compression_enabled:
            return
        previous_summary = self.state.summary
        await super().compress_context(*args, **kwargs)
        if self.state.summary != previous_summary:
            self.compression_count += 1


class AuditToolBridge(ToolBase):
    """Expose one existing audit ``BaseTool`` as an AgentScope tool."""

    is_concurrency_safe = True
    is_read_only = True
    is_external_tool = False
    is_state_injected = False

    def __init__(
        self,
        audit_tool: AuditBaseTool,
        executor: ToolExecutor,
        recorder: ToolRunRecorder,
        image_max_tokens: int,
    ) -> None:
        super().__init__()
        self.audit_tool = audit_tool
        self.executor = executor
        self.recorder = recorder
        self.image_max_tokens = image_max_tokens
        self.name = audit_tool.name
        self.description = audit_tool.function_entry()["function"]["description"]
        self.input_schema = dict(audit_tool.public_input_schema or {})
        self.input_schema.setdefault("type", "object")
        self.input_schema.setdefault("properties", {})
        self.input_schema.setdefault("additionalProperties", False)

    async def check_permissions(
        self,
        tool_input: dict[str, Any],
        context: PermissionContext,
    ) -> PermissionDecision:
        del tool_input, context
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="Audit evidence tools are read-only.",
            decision_reason="Read-only audit evidence lookup",
        )

    async def call(self, **kwargs: Any) -> ToolChunk:
        started = time.perf_counter()
        call = ToolCall(
            tool_name=self.name,
            args=dict(kwargs),
            reason="AgentScope native tool call",
        )
        observations = await asyncio.to_thread(self.executor.execute, [call], 1)
        observation = observations[0]
        record = {
            "tool_name": self.name,
            "args": dict(kwargs),
            "status": observation.status,
            "error": observation.error,
            "cache_bound_args": observation.call.args if observation.call else {},
        }

        if observation.status != "ok":
            record["latency_seconds"] = round(time.perf_counter() - started, 3)
            self.recorder.append(record)
            return ToolChunk(
                content=[
                    TextBlock(
                        text=json.dumps(
                            {
                                "status": "failed",
                                "tool_name": self.name,
                                "error": observation.error,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                ],
                state=ToolResultState.ERROR,
                metadata=record,
            )

        rendered = self.audit_tool.render(observation.result or {})
        record["image_count"] = len(rendered.images)
        record["text_chars"] = len(rendered.text)
        content = await to_agentscope_blocks_async(
            rendered.text,
            rendered.images,
            self.image_max_tokens,
        )
        record["latency_seconds"] = round(time.perf_counter() - started, 3)
        self.recorder.append(record)
        return ToolChunk(
            content=content,
            metadata=record,
        )


def build_rule_preview(rule: RuleInfo, max_chars: int = 700) -> str:
    """Extract a compact navigation view without rewriting rule semantics."""
    text = str(rule.rule_text or "").strip()
    if not text:
        return f"## {rule.label}\n（无规则文本）"
    if len(text) <= max_chars:
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    selected: List[str] = [f"## {rule.label}"]
    section_titles: List[str] = []
    exemption_lines: List[str] = []
    capture = False
    captured_chars = 0
    field_pattern = re.compile(
        r"^\*\*(风险等级|标签定义|审核前提|管控前提|适用范围)\*\*",
    )
    for line in lines:
        if line.startswith("###"):
            section_titles.append(line.lstrip("# ").strip())
            capture = False
            continue
        if field_pattern.match(line):
            selected.append(line)
            capture = True
            captured_chars += len(line)
            continue
        if line.startswith("**"):
            capture = False
        if capture and captured_chars < 360:
            selected.append(line)
            captured_chars += len(line)
        if any(token in line for token in ("豁免", "不管控", "无需管控")):
            exemption_lines.append(line)

    if section_titles:
        selected.append("**规则章节**: " + "；".join(dict.fromkeys(section_titles[:10])))
    if exemption_lines:
        selected.append("**豁免提示**: " + "；".join(dict.fromkeys(exemption_lines[:3])))

    preview = "\n".join(selected).strip()
    if len(preview) < 80:
        preview = text[:max_chars]
    if len(preview) > max_chars:
        preview = preview[: max(0, max_chars - 12)].rstrip() + "\n（预览截断）"
    return preview


class DetailRuleTool(ToolBase):
    """Return exact full rule text for model-selected candidate labels."""

    is_concurrency_safe = True
    is_read_only = True
    is_external_tool = False
    is_state_injected = False

    def __init__(
        self,
        rules: List[RuleInfo],
        recorder: ToolRunRecorder,
        image_max_tokens: int,
        max_labels: int = 6,
    ) -> None:
        super().__init__()
        self.name = "get_detail_rule"
        self.description = (
            "读取当前审核 case 中一个或多个候选标签的完整规则原文。"
            "当规则预览显示某标签可能相关、需要检查完整命中条件或豁免条件时调用。"
            "可以多次调用；source 已由运行时绑定，只需提供 labels。"
        )
        self.rules_by_label = {
            rule.label: rule
            for rule in rules
            if rule.label not in {QUEUE_NOTICE_LABEL, "通过"}
        }
        self.recorder = recorder
        self.image_max_tokens = image_max_tokens
        self.max_labels = max(1, int(max_labels))
        labels = list(self.rules_by_label)
        self.input_schema = {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": min(self.max_labels, max(1, len(labels))),
                    "items": {"type": "string", "enum": labels},
                    "description": "需要读取完整规则的候选标签列表。",
                },
            },
            "required": ["labels"],
            "additionalProperties": False,
        }

    async def check_permissions(
        self,
        tool_input: dict[str, Any],
        context: PermissionContext,
    ) -> PermissionDecision:
        del tool_input, context
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="Audit rules are read-only local resources.",
            decision_reason="Read-only rule lookup",
        )

    async def call(self, labels: List[str]) -> ToolChunk:
        started = time.perf_counter()
        requested = list(dict.fromkeys(str(label).strip() for label in labels if str(label).strip()))
        selected = [self.rules_by_label[label] for label in requested if label in self.rules_by_label]
        missing = [label for label in requested if label not in self.rules_by_label]
        record: Dict[str, Any] = {
            "tool_name": self.name,
            "args": {"labels": requested},
            "status": "ok" if selected else "failed",
            "error": "" if selected else f"No matching rules for labels: {missing}",
            "loaded_rule_labels": [rule.label for rule in selected],
            "missing_rule_labels": missing,
        }
        self.recorder.append(record)
        if not selected:
            record["latency_seconds"] = round(time.perf_counter() - started, 3)
            return ToolChunk(
                content=[TextBlock(text=json.dumps(record, ensure_ascii=False))],
                state=ToolResultState.ERROR,
                metadata=record,
            )

        text_parts: List[str] = ["# 完整审核规则原文"]
        images: List[str] = []
        for rule in selected:
            text_parts.append(rule.rule_text)
            images.extend(rule.rule_images or [])
        if missing:
            text_parts.append("未找到规则的标签：" + "、".join(missing))
        text = "\n\n".join(text_parts)
        record["text_chars"] = len(text)
        record["image_count"] = len(images)
        content = await to_agentscope_blocks_async(
            text,
            images,
            self.image_max_tokens,
        )
        record["latency_seconds"] = round(time.perf_counter() - started, 3)
        return ToolChunk(
            content=content,
            metadata=record,
        )


def build_result_schema(candidate_labels: List[str]) -> Type[BaseModel]:
    """Build a per-case schema whose label field is a strict enum."""
    labels = list(dict.fromkeys(label for label in candidate_labels if label))
    enum_values = {f"LABEL_{index}": label for index, label in enumerate(labels)}
    audit_label = Enum("AuditLabel", enum_values, type=str)
    return create_model(
        "AgentScopeAuditResult",
        predict_label=(audit_label, ...),
        decision_basis=(
            str,
            Field(
                min_length=1,
                description=(
                    "用一段简洁、可复核的文本说明最终判定依据；"
                    "可以换行分点，但不要输出 JSON 数组。"
                ),
            ),
        ),
        used_evidence_ids=(List[str], Field(default_factory=list)),
    )


def build_router_schema(candidate_labels: List[str]) -> Type[BaseModel]:
    """Build a strict high-recall shortlist schema for one case."""
    labels = list(dict.fromkeys(label for label in candidate_labels if label))
    enum_values = {f"LABEL_{index}": label for index, label in enumerate(labels)}
    router_label = Enum("RouterLabel", enum_values, type=str)
    minimum = min(4, len(labels))
    maximum = min(12, len(labels))
    return create_model(
        "AgentScopeRouterResult",
        brief_analysis=(str, ...),
        shortlist_labels=(
            List[router_label],
            Field(min_length=minimum, max_length=maximum),
        ),
    )


def _usage(agent: Agent) -> Dict[str, int]:
    input_tokens = 0
    output_tokens = 0
    model_calls = 0
    for message in agent.state.context:
        if message.usage is None:
            continue
        model_calls += 1
        input_tokens += int(message.usage.input_tokens or 0)
        output_tokens += int(message.usage.output_tokens or 0)
    return {
        "model_calls": model_calls,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
    }


class AgentScopeAuditAgent:
    """Run one audit case through AgentScope's native agent loop."""

    def __init__(
        self,
        model: Any,
        prompt_loader: Any,
        tool_executor: ToolExecutor,
        tool_registry: Dict[str, AuditBaseTool] | None = None,
        max_iters: int = 8,
        structured_output_grace_iters: int = 2,
        experience_enabled: bool = True,
        router_mode: str = "none",
        router_bypass_max_labels: int = 8,
        rule_loading_mode: str = "all",
        rule_preview_max_chars: int = 700,
        detail_rule_max_labels: int = 6,
        compression_enabled: bool = True,
        compression_trigger_tokens: int = 0,
        compression_profile: str = "default",
        full_trace: bool = False,
    ) -> None:
        self.model = model
        self.prompt_loader = prompt_loader
        self.tool_executor = tool_executor
        self.tool_registry = tool_registry or DEFAULT_TOOL_REGISTRY
        self.max_iters = max(1, int(max_iters))
        self.structured_output_grace_iters = max(
            1,
            int(structured_output_grace_iters),
        )
        self.experience_enabled = experience_enabled
        self.router_mode = str(router_mode or "none").strip().lower()
        if self.router_mode not in {"none", "agentscope"}:
            raise ValueError(f"Unsupported AgentScope router mode: {router_mode!r}")
        self.router_bypass_max_labels = max(0, int(router_bypass_max_labels))
        self.rule_loading_mode = str(rule_loading_mode or "all").strip().lower()
        if self.rule_loading_mode not in {"all", "preview_tool"}:
            raise ValueError(f"Unsupported rule loading mode: {rule_loading_mode!r}")
        self.rule_preview_max_chars = max(200, int(rule_preview_max_chars))
        self.detail_rule_max_labels = max(1, int(detail_rule_max_labels))
        self.compression_enabled = bool(compression_enabled)
        self.compression_trigger_tokens = max(
            0,
            int(compression_trigger_tokens or 0),
        )
        self.compression_profile = str(compression_profile or "default").strip().lower()
        if self.compression_profile not in {"default", "audit"}:
            raise ValueError(
                f"Unsupported compression profile: {compression_profile!r}",
            )
        self.full_trace = full_trace

    async def _run_router(
        self,
        audit_input: AuditInput,
        candidate_labels: List[str],
        image_max_tokens: int,
    ) -> tuple[List[str], Dict[str, Any]]:
        original = list(dict.fromkeys(label for label in candidate_labels if label))
        should_route = (
            self.router_mode == "agentscope"
            and len(original) > self.router_bypass_max_labels
        )
        if not should_route:
            return original, {
                "mode": self.router_mode,
                "invoked": False,
                "selection_mode": "all_candidates",
                "shortlist_labels": original,
                "model_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        prompt = self.prompt_loader.render(
            "main_router_schema",
            note=audit_input.note,
            candidate_labels=original,
        )
        content = await to_agentscope_blocks_async(
            prompt,
            list(audit_input.images or []),
            image_max_tokens,
        )
        agent = Agent(
            name="audit_router",
            system_prompt=ROUTER_SYSTEM_PROMPT,
            model=self.model,
            toolkit=Toolkit(tools=[]),
            state=AgentState(),
            react_config=ReActConfig(
                max_iters=1,
                structured_output_grace_iters=self.structured_output_grace_iters,
            ),
            injection_config=InjectionConfig(inject_runtime_state=False),
        )
        final_message = await agent.reply(
            UserMsg(name="router_request", content=content),
            structured_schema=build_router_schema(original),
        )
        structured = dict(final_message.structured_output or {})
        candidate_set = set(original)
        shortlist = [
            str(label).strip()
            for label in structured.get("shortlist_labels") or []
            if str(label).strip() in candidate_set
        ]
        shortlist = list(dict.fromkeys(shortlist))
        model_shortlist = list(shortlist)
        fallback = not shortlist
        if fallback:
            shortlist = list(original)
        if "通过" in candidate_set and "通过" not in shortlist:
            shortlist.append("通过")

        trace: Dict[str, Any] = {
            "mode": self.router_mode,
            "invoked": True,
            "selection_mode": "agentscope_router",
            "structured_output": structured,
            "model_shortlist_labels": model_shortlist,
            "shortlist_labels": shortlist,
            "fallback_to_all_candidates": fallback,
            "finished_reason": str(final_message.finished_reason or ""),
            **_usage(agent),
        }
        if self.full_trace:
            trace["context"] = [
                message.model_dump(mode="json")
                for message in agent.state.context
            ]
        return shortlist, trace

    async def run(
        self,
        audit_input: AuditInput,
        candidate_labels: List[str],
        rules: List[RuleInfo],
        image_max_tokens: int = DEFAULT_IMAGE_MAX_TOKENS,
    ) -> tuple[MainFinalOutput, Dict[str, Any]]:
        original_candidate_labels = list(
            dict.fromkeys(label for label in candidate_labels if label),
        )
        candidate_labels, router_trace = await self._run_router(
            audit_input,
            original_candidate_labels,
            image_max_tokens,
        )
        selected_label_set = set(candidate_labels)
        selected_rules = [
            rule
            for rule in rules
            if rule.label == QUEUE_NOTICE_LABEL or rule.label in selected_label_set
        ]
        if not selected_rules:
            selected_rules = list(rules)
            candidate_labels = list(original_candidate_labels)
            router_trace["rule_filter_fallback"] = True
        rules = selected_rules

        allowed_names = set(self.tool_registry)
        if audit_input.available_tools:
            allowed_names &= set(audit_input.available_tools)
        registry = {
            name: tool
            for name, tool in self.tool_registry.items()
            if name in allowed_names
        }

        audit_context = {
            "note_id": audit_input.note_id,
            "source_id": audit_input.source_id,
            "note": audit_input.note,
            "images": list(audit_input.images or []),
            "candidate_labels": candidate_labels,
            "rules": [rule.model_dump() for rule in rules],
            "observations": [],
        }
        runtime_defaults: Dict[str, Any] = {
            "note_id": audit_input.note_id,
            "source_id": audit_input.source_id,
            "_audit_context": audit_context,
        }
        if os.environ.get("AUDIT_TOOL_CACHE_IDENTITY_MODE", "").lower() == "note_history":
            runtime_defaults["history_id"] = audit_input.history_id

        executor = ToolExecutor(
            registry,
            default_args=self.tool_executor.default_args,
        ).with_defaults(**runtime_defaults)
        recorder = ToolRunRecorder()
        bridges = [
            AuditToolBridge(
                audit_tool=tool,
                executor=executor,
                recorder=recorder,
                image_max_tokens=image_max_tokens,
            )
            for tool in registry.values()
        ]

        queue_notice_rules = [rule for rule in rules if rule.label == QUEUE_NOTICE_LABEL]
        label_rules = [rule for rule in rules if rule.label != QUEUE_NOTICE_LABEL]
        rule_previews = [
            {
                "label": rule.label,
                "preview": build_rule_preview(rule, self.rule_preview_max_chars),
            }
            for rule in label_rules
        ]
        if self.rule_loading_mode == "preview_tool":
            bridges.append(
                DetailRuleTool(
                    rules=label_rules,
                    recorder=recorder,
                    image_max_tokens=image_max_tokens,
                    max_labels=self.detail_rule_max_labels,
                ),
            )

        experience = (
            load_audit_experience(audit_input.source_id, labels=candidate_labels)
            if self.experience_enabled
            else ""
        )
        if self.experience_enabled and self.rule_loading_mode == "preview_tool":
            experience, rule_previews = load_audit_experience_for_rule_previews(
                audit_input.source_id, candidate_labels, rule_previews,
            )
        if self.rule_loading_mode == "preview_tool":
            prompt = self.prompt_loader.render(
                "agentscope_audit_agent_preview",
                note=audit_input.note,
                candidate_labels=candidate_labels,
                queue_notices=[rule.model_dump() for rule in queue_notice_rules],
                rule_previews=rule_previews,
                audit_experience=experience,
            )
            initial_rule_images = collect_rule_images(queue_notice_rules)
        else:
            prompt = self.prompt_loader.render(
                "agentscope_audit_agent",
                note=audit_input.note,
                candidate_labels=candidate_labels,
                rules=[rule.model_dump() for rule in rules],
                audit_experience=experience,
            )
            initial_rule_images = collect_rule_images(rules)
        images = list(audit_input.images or []) + initial_rule_images
        user_content = await to_agentscope_blocks_async(
            prompt,
            images,
            image_max_tokens,
        )

        compression_config = build_compression_context_config(
            self.model,
            self.compression_trigger_tokens,
            self.compression_profile,
        )
        agent = ConfigurableCompressionAgent(
            name="audit_agent",
            system_prompt=SYSTEM_PROMPT,
            model=self.model,
            toolkit=Toolkit(tools=bridges),
            state=AgentState(),
            react_config=ReActConfig(
                max_iters=self.max_iters,
                structured_output_grace_iters=self.structured_output_grace_iters,
            ),
            context_config=compression_config,
            injection_config=InjectionConfig(inject_runtime_state=False),
            compression_enabled=self.compression_enabled,
        )
        result_schema = build_result_schema(candidate_labels)
        final_message = await agent.reply(
            UserMsg(name="audit_request", content=user_content),
            structured_schema=result_schema,
        )
        structured, label, final_protocol_errors = parse_final_structured_output(
            final_message.structured_output,
            candidate_labels,
        )
        valid_final_output = not final_protocol_errors
        raw_basis = structured.get("decision_basis")
        basis = normalize_decision_basis(raw_basis)
        used_tools = list(
            dict.fromkeys(
                record["tool_name"]
                for record in recorder.calls
                if record.get("status") == "ok"
            ),
        )
        loaded_rule_labels = list(
            dict.fromkeys(
                label
                for record in recorder.calls
                if record.get("tool_name") == "get_detail_rule"
                and record.get("status") == "ok"
                for label in record.get("loaded_rule_labels") or []
            ),
        )
        output = MainFinalOutput(
            predict_label=[label] if valid_final_output else [],
            binary_decision=(
                "pass"
                if label == "通过"
                else "violation"
                if valid_final_output
                else "invalid"
            ),
            audit_trace="；".join(basis),
            used_rules=(
                loaded_rule_labels
                if self.rule_loading_mode == "preview_tool"
                else [rule.label for rule in rules]
            ),
            used_tools=used_tools,
        )
        audit_usage = _usage(agent)
        trace: Dict[str, Any] = {
            "mode": (
                "agentscope_native_router"
                if router_trace.get("invoked")
                else "agentscope_native"
            ),
            "framework_version": "2.0.5",
            "structured_output": structured,
            "valid_final_output": valid_final_output,
            "invalid_final_output": not valid_final_output,
            "raw_predict_label": label,
            "final_protocol_error_types": final_protocol_errors,
            "decision_basis_input_type": type(raw_basis).__name__,
            "finished_reason": str(final_message.finished_reason or ""),
            "tool_calls": recorder.calls,
            "available_tool_names": sorted(bridge.name for bridge in bridges),
            "experience_enabled": bool(experience) or any(
                item.get("label_experience") for item in rule_previews
            ),
            "experience_chars": len(experience),
            "label_experience_labels": [
                item["label"] for item in rule_previews if item.get("label_experience")
            ],
            "label_experience_chars": sum(
                len(item.get("label_experience", "")) for item in rule_previews
            ),
            "rule_loading_mode": self.rule_loading_mode,
            "rule_preview_chars": sum(len(item["preview"]) for item in rule_previews),
            "detail_rule_call_count": sum(
                record.get("tool_name") == "get_detail_rule"
                for record in recorder.calls
            ),
            "context_compression_enabled": self.compression_enabled,
            "context_compression_profile": self.compression_profile,
            "context_compression_trigger_tokens": int(
                compression_config.trigger_ratio * self.model.context_size,
            ),
            "context_compression_count": int(agent.compression_count),
            "context_compressed": bool(agent.compression_count),
            "loaded_rule_labels": loaded_rule_labels,
            "final_label_detail_loaded": (
                valid_final_output
                and (
                    self.rule_loading_mode == "all"
                    or label in loaded_rule_labels
                    or (label == "通过" and bool(loaded_rule_labels))
                )
            ),
            "router": router_trace,
            "router_mode": self.router_mode,
            "router_invoked": bool(router_trace.get("invoked")),
            "router_shortlist_labels": list(candidate_labels),
            "original_candidate_label_count": len(original_candidate_labels),
            "candidate_label_count": len(candidate_labels),
            "rule_count": len(rules),
            "model_calls": (
                int(router_trace.get("model_calls") or 0)
                + int(audit_usage.get("model_calls") or 0)
            ),
            "prompt_tokens": (
                int(router_trace.get("prompt_tokens") or 0)
                + int(audit_usage.get("prompt_tokens") or 0)
            ),
            "completion_tokens": (
                int(router_trace.get("completion_tokens") or 0)
                + int(audit_usage.get("completion_tokens") or 0)
            ),
            "audit_model_calls": int(audit_usage.get("model_calls") or 0),
            "audit_prompt_tokens": int(audit_usage.get("prompt_tokens") or 0),
            "audit_completion_tokens": int(
                audit_usage.get("completion_tokens") or 0,
            ),
        }
        if self.full_trace:
            trace["system_prompt"] = SYSTEM_PROMPT
            trace["tool_schemas"] = [
                {
                    "name": bridge.name,
                    "description": bridge.description,
                    "input_schema": bridge.input_schema,
                }
                for bridge in bridges
            ]
            trace["structured_output_schema"] = result_schema.model_json_schema()
            trace["context"] = [
                message.model_dump(mode="json")
                for message in agent.state.context
            ]
        return output, trace
