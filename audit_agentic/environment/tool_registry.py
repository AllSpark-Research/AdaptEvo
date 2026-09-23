"""Tool registry: declares tool metadata used by Planner / Executor.

Registry 的值是 ``BaseTool`` 实例（不再是 dict SPEC）。framework 只依赖
``BaseTool`` 接口。

Active tools:
  * ``check_plagiarism_video`` — 视频搬运检测
  * ``check_plagiarism_image`` — 图文搬运检测
  * ``get_commercial_detail`` — 商业详情（商品SKU）
  * ``get_user_qualification`` — 用户资质（交易类目 + 医疗资质 + 行业投放资质 + 蒲公英）
  * ``get_recent_notes`` — 用户近期笔记
  * ``get_comments`` — 完整评论列表
  * ``get_user_records`` — 用户资料修改记录
  * ``get_report_history`` — 用户举报记录
  * ``get_user_comments`` — 用户在其他笔记下的评论
"""

from __future__ import annotations

from typing import Any, Dict, List

from .tools import ACTIVE_TOOLS as _ACTIVE_TOOLS
from .tools import load_tool_briefs
from .tools.base_tool import BaseTool

# Production registry: name -> BaseTool instance.
DEFAULT_TOOL_REGISTRY: Dict[str, BaseTool] = dict(_ACTIVE_TOOLS)


def list_tool_specs(registry: Dict[str, BaseTool] | None = None) -> List[Dict[str, Any]]:
    """Public tool descriptions for downstream code (no runner)."""
    registry = registry or DEFAULT_TOOL_REGISTRY
    return [tool.spec_entry() for tool in registry.values()]


def list_tool_briefs() -> List[Dict[str, Any]]:
    """Planner-facing brief list, auto-derived from each tool's class attrs."""
    return load_tool_briefs(only_active=True)


def list_openai_tools(
    registry: Dict[str, BaseTool] | None = None,
    allowed_names: set[str] | None = None,
) -> List[Dict[str, Any]]:
    """Return native OpenAI/Qwen function definitions for active tools."""
    registry = registry or DEFAULT_TOOL_REGISTRY
    return [
        tool.function_entry()
        for name, tool in registry.items()
        if allowed_names is None or name in allowed_names
    ]
