"""统一工具框架核心抽象。

每个工具是一个自包含的 py，实现一个 ``BaseTool`` 子类：
  * 元数据用类属性声明（name / description / brief / when / ...），framework
    自动派生 planner 看到的 brief。
  * ``run(args) -> dict`` 取数，返回工具私有的结构化 dict。
  * ``render(result) -> ToolRenderResult`` 把结果组装成 prompt 片段 + 图片列表。
    简单工具设 ``RESULT_TEMPLATE`` 走默认实现即可；复杂工具 override render()
    （可在里面走 tool_subagent）。

对外唯一契约是 ``ToolRenderResult(text, images)`` —— 不管工具内部 inline 拼图
还是走 subagent 出摘要，下游拿到的都是这一个形状。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from jinja2 import Environment

IMAGE_TAG = "<image>"

# 工具结果模板用一个宽松的 jinja2 env：缺字段渲染成空串而不是报错
# （各工具 result dict 字段不一，不能用 StrictUndefined）。
_TOOL_ENV = Environment(autoescape=False, keep_trailing_newline=True)


def render_jinja(template_str: str, **ctx: Any) -> str:
    """用宽松 env 渲染一个 jinja2 模板串。"""
    return _TOOL_ENV.from_string(template_str).render(**ctx)


@dataclass
class ToolRenderResult:
    """工具 → prompt 层的标准化输出（下游唯一依赖的契约）。

    - ``text``：prompt 片段，可含 ``<image>`` 占位符。
    - ``images``：与 text 中 ``<image>`` **严格对齐**（等量、同序）的图片路径。
    """

    text: str = ""
    images: List[str] = field(default_factory=list)


def align_images(rr: ToolRenderResult) -> ToolRenderResult:
    """保证 text 中 ``<image>`` 数量 == len(images)，多退少补。

    - 占位符比图多：多出来的 ``<image>`` 替换成"图片"（从后往前）。
    - 图比占位符多：多出来的图截断丢弃。

    这是最大正确性保障：下游 make_multimodal_message 按位置映射 <image>→image，
    数量不齐会整体错位（把 A 工具的图配到 B 工具的占位符上）。
    """
    text = rr.text or ""
    images = list(rr.images or [])
    n_tags = text.count(IMAGE_TAG)

    if n_tags == len(images):
        return ToolRenderResult(text=text, images=images)

    if n_tags > len(images):
        # 从后往前把多余的 <image> 替换成"图片"，保留前 len(images) 个占位符
        keep = len(images)
        # 用一个占位标记逐个替换：先全部拆分再重组
        parts = text.split(IMAGE_TAG)
        # parts 之间有 n_tags 个分隔点；前 keep 个保留为 <image>，其余用"图片"
        rebuilt = parts[0]
        for i, seg in enumerate(parts[1:]):
            rebuilt += (IMAGE_TAG if i < keep else "图片") + seg
        return ToolRenderResult(text=rebuilt, images=images)

    # 图比占位符多：截断图片
    return ToolRenderResult(text=text, images=images[:n_tags])


class BaseTool(ABC):
    """所有工具的统一接口。framework 只依赖本类。"""

    # ── 元数据（类属性；子类覆盖）──
    name: str = ""
    description: str = ""
    brief: str = ""          # planner 看到的一句话功能简介
    when: str = ""           # planner 看到的使用场景
    input_schema: Dict[str, Any] = {"note_id": "待审笔记ID"}
    # OpenAI/Qwen native function-calling parameters exposed to the model.
    # Runtime-bound fields such as note_id/source_id must not be public.
    public_input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    output_schema: Dict[str, Any] = {"text": "string"}
    cost: int = 1
    # Dynamic tools whose result depends on session context or an inner LLM
    # must opt out of the generic file cache.
    cacheable: bool = True
    image_mode: str = "inline"       # "inline" | "summarize"
    images_field: str = "display_images"   # run() 结果里图片列表的字段名

    # 默认结果外壳；简单工具只需覆盖这一行，复杂工具 override render()
    RESULT_TEMPLATE: str = "### {{ tool_name }} 结果\n{{ text }}"

    # ── 取数（子类必须实现）──
    @abstractmethod
    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """给 args（至少含 note_id），返回工具私有的结构化 dict。

        约定：返回 dict 里 ``text`` 中的 ``<image>`` 占位符数量应与
        ``images_field`` 列表长度对齐（render 时还会用 align_images 兜底）。
        """
        raise NotImplementedError

    # ── 组装 prompt（默认实现，子类可 override）──
    def render(self, result: Dict[str, Any]) -> ToolRenderResult:
        """把 run() 的结果渲染成标准化的 (text, images)。"""
        result = result or {}
        text = render_jinja(self.RESULT_TEMPLATE, tool_name=self.name, **result)
        images = list(result.get(self.images_field, []) or [])
        rr = ToolRenderResult(text=text, images=images)

        if self.image_mode == "summarize":
            # 走 tool_subagent 出摘要（当前未实现 → 安全 fallback 成纯文本）
            from ...agents.tool_subagent import subagent_or_fallback
            return subagent_or_fallback(self, rr)

        return align_images(rr)

    # ── planner brief（自动派生，无需子类实现）──
    def brief_entry(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "brief": self.brief or self.description,
            "when": self.when,
            "args": self.input_schema,
        }

    def spec_entry(self) -> Dict[str, Any]:
        """给下游/文档的完整描述（无 runner）。"""
        return {
            "tool_name": self.name,
            "description": self.description,
            "brief": self.brief,
            "when": self.when,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "cost": self.cost,
            "cacheable": self.cacheable,
            "image_mode": self.image_mode,
        }

    def function_entry(self) -> Dict[str, Any]:
        """Return an OpenAI-compatible native function tool definition."""
        description = self.description or self.brief
        if self.when:
            description = f"{description}\n适用场景：{self.when}"
        parameters = dict(self.public_input_schema or {})
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        parameters.setdefault("additionalProperties", False)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": parameters,
            },
        }

    @property
    def needs_subagent(self) -> bool:
        """是否需要 tool_subagent 处理结果（当前 == image_mode 为 summarize）。"""
        return self.image_mode == "summarize"

    def catalog_entry(self) -> Dict[str, Any]:
        """给 tools_catalog.json 的机器可读条目（自动生成，勿手改）。"""
        return {
            "name": self.name,
            "description": self.description,
            "brief": self.brief,
            "when": self.when,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "cost": self.cost,
            "cacheable": self.cacheable,
            "image_mode": self.image_mode,
            "needs_subagent": self.needs_subagent,
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
        }
