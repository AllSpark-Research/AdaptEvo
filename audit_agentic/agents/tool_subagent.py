"""Tool subagent 接口（本次只设计接口 + 安全 fallback，暂不实现）。

设计意图：某些工具返回的信息量大 / 图片多，直接 inline 进 main_final 会占用
大量 image token。这类工具可声明 ``image_mode = "summarize"``，把 (text, images)
先交给一个独立的 tool_subagent，让它看图 + 读文，压成一段文字 tool_summary，
再作为纯文本喂给 main agent（不带图）。

当前状态：接口已定义，具体 subagent 调用未实现；``subagent_or_fallback`` 会安全
退化成"把 <image> 换成'图片'文字、不带图"的纯文本片段，保证 image_mode=summarize
的工具即使在 subagent 未接线时也能正常跑通。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:  # 避免运行时循环 import
    from ..environment.tools.base_tool import BaseTool, ToolRenderResult


class ToolSubagent(Protocol):
    """把工具结果（文字+图）压成一段文字摘要的 agent。"""

    def summarize(self, tool_name: str, text: str, images: list[str]) -> str:
        ...


# 全局可选注入点：训练/推理启动时可 set_tool_subagent(实例) 接线真实实现。
_ACTIVE_SUBAGENT: Optional[ToolSubagent] = None


def set_tool_subagent(agent: Optional[ToolSubagent]) -> None:
    global _ACTIVE_SUBAGENT
    _ACTIVE_SUBAGENT = agent


def subagent_or_fallback(tool: "BaseTool", rr: "ToolRenderResult") -> "ToolRenderResult":
    """image_mode='summarize' 分支：有 subagent 则出摘要；否则安全退化成纯文本。"""
    from ..environment.tools.base_tool import ToolRenderResult

    if _ACTIVE_SUBAGENT is not None:
        summary = _ACTIVE_SUBAGENT.summarize(tool.name, rr.text, rr.images)
        return ToolRenderResult(text=summary, images=[])

    # TODO(subagent 未实现): 退化为纯文本，<image> 占位符换"图片"，不带图
    return ToolRenderResult(text=(rr.text or "").replace("<image>", "图片"), images=[])
