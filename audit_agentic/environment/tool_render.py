"""把工具调用结果（List[ToolObservation]）聚合成 main_final continuation 用的
(tool_section_text, tool_images)。

每个工具的定制渲染 / 图片模式 / subagent 逻辑都在各自的 ``BaseTool.render()``
里；这里只负责调 render() + 顺序聚合 + 保证整体的 <image> 与 images 对齐。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Tuple

from .tools.base_tool import IMAGE_TAG, BaseTool, ToolRenderResult, align_images

_EMPTY = "（Planner 未调用任何工具，或工具调用为空。）"
_DEFAULT_TOOL_IMAGE_LIMIT = -1


def _tool_image_limit() -> int:
    raw = os.environ.get("AUDIT_TOOL_IMAGE_LIMIT", str(_DEFAULT_TOOL_IMAGE_LIMIT))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TOOL_IMAGE_LIMIT


def _limit_observation_images(rr: ToolRenderResult, max_images: int) -> ToolRenderResult:
    """Limit images for one tool observation while preserving alignment.

    Keep the first ``max_images`` image placeholders and image paths. Extra
    ``<image>`` placeholders are replaced by an empty string instead of text,
    so downstream multimodal construction sees exactly the kept placeholders.
    """
    aligned = align_images(rr)
    if max_images < 0:
        return aligned
    images = list(aligned.images or [])
    if len(images) <= max_images:
        return aligned

    parts = (aligned.text or "").split(IMAGE_TAG)
    rebuilt = parts[0]
    for i, seg in enumerate(parts[1:]):
        rebuilt += (IMAGE_TAG if i < max_images else "") + seg
    return ToolRenderResult(text=rebuilt, images=images[:max_images])


def render_observations(
    observations: List[Any],
    registry: Mapping[str, BaseTool],
) -> Tuple[str, List[str]]:
    """聚合工具结果。

    Args:
        observations: List[ToolObservation]（或等价 dict/对象，含 tool_name /
            status / result / error）。
        registry: name -> BaseTool 实例。

    Returns:
        (text, images)：text 含 ``<image>`` 占位符；images 与之严格对齐。
    """
    if not observations:
        return _EMPTY, []

    fragments: List[str] = []
    images: List[str] = []
    max_images_per_tool = _tool_image_limit()

    for obs in observations:
        tool_name = _get(obs, "tool_name")
        status = _get(obs, "status", "ok")
        result = _get(obs, "result")
        error = _get(obs, "error")

        tool = registry.get(tool_name)
        if status == "ok" and tool is not None:
            rr = tool.render(result if isinstance(result, dict) else {})
        else:
            reason = error or ("unknown_tool" if tool is None else "failed")
            rr = ToolRenderResult(text=f"- 工具 {tool_name} 调用失败：{reason}", images=[])

        rr = _limit_observation_images(rr, max_images_per_tool)
        fragments.append(rr.text)
        images.extend(rr.images)

    text = "\n\n".join(f for f in fragments if f)
    # 整体再兜底一次：防止某工具 render() 返回的 text/images 未对齐
    aligned = align_images(ToolRenderResult(text=text, images=images))
    return aligned.text, aligned.images


def _get(obs: Any, key: str, default: Any = None) -> Any:
    """兼容 pydantic 对象 / dict 两种 observation 形态。"""
    if isinstance(obs, Mapping):
        return obs.get(key, default)
    return getattr(obs, key, default)
