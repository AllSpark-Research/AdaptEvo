"""评论工具：获取笔记完整评论列表（前10条）。

基础特征里只展示作者评论和置顶评论，本工具返回完整评论列表。

涉及服务:
- getNoteCommentApp: 笔记评论列表
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..services._base import DataBackend, OnlineBackend, OfflineBackend


def _extract_all_comments(data: Optional[dict], author_user_id: str = "", max_count: int = 0) -> Dict[str, Any]:
    """从 getNoteCommentApp 返回中提取完整评论列表。max_count=0 表示不限。"""
    if not data:
        return {"total": 0, "comments": []}

    pageable = data.get("pageable", "{}")
    if isinstance(pageable, str):
        try:
            pageable = json.loads(pageable)
        except (ValueError, TypeError):
            pageable = {}
    total = pageable.get("total", 0) if isinstance(pageable, dict) else 0

    comment_raw = data.get("comment", "[]")
    if isinstance(comment_raw, str):
        try:
            comment_raw = json.loads(comment_raw)
        except (ValueError, TypeError):
            return {"total": total, "comments": []}

    if not isinstance(comment_raw, list):
        return {"total": total, "comments": []}

    results = []
    for c in (comment_raw[:max_count] if max_count else comment_raw):
        if not isinstance(c, dict):
            continue
        user_id = c.get("userId", "")
        is_author = (user_id == author_user_id) if author_user_id else False

        results.append({
            "content": c.get("content", ""),
            "user_name": c.get("userName", ""),
            "user_avatar": c.get("userAvatar", ""),
            "is_author": is_author,
            "is_top": bool(c.get("topStatus", False)),
            "like_count": c.get("likeCount", 0),
            "time": c.get("time"),
            "ip": c.get("ip", ""),
        })

    return {"total": total, "comments": results}


def _format_ts(ts) -> str:
    """时间戳（毫秒）转具体时间。"""
    if not ts:
        return ""
    from datetime import datetime
    try:
        return datetime.fromtimestamp(int(ts) / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, OSError):
        return ""


def render_comments_text(result: Dict[str, Any]) -> str:
    """渲染评论为文本。"""
    total = result.get("total", 0)
    comments = result.get("comments", [])

    if not comments:
        return "## 笔记评论\n\n- 无评论\n"

    lines = ["## 笔记评论", f"（共 {total} 条，展示 {len(comments)} 条）", ""]

    for c in comments:
        marks = []
        if c.get("is_top"):
            marks.append("置顶")
        if c.get("is_author"):
            marks.append("作者")
        mark_str = f" [{','.join(marks)}]" if marks else ""

        name = c.get("user_name", "")
        content = c.get("content", "")
        likes = c.get("like_count", 0)
        time_str = _format_ts(c.get("time"))
        ip = c.get("ip", "")

        meta_parts = []
        if time_str:
            meta_parts.append(time_str)
        if ip:
            meta_parts.append(f"IP:{ip}")
        if likes:
            meta_parts.append(f"赞{likes}")
        meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""

        lines.append(f"- {name}{mark_str}: {content}{meta_str}")

    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# Tool SPEC
# ===========================================================================

TOOL_NAME = "get_comments"
DESCRIPTION = (
    "获取笔记完整评论列表。"
    "基础特征里只有作者评论和置顶评论，调用本工具可查看所有评论。"
)
USE_WHEN = [
    "需要查看笔记完整评论区",
    "判断评论区是否有营销导流",
    "查看用户评论区互动情况",
]
INPUT_SCHEMA = {"note_id": "string"}
OUTPUT_SCHEMA = {"text": "string", "total": "int"}
COST = 1


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    """Tool 入口。"""
    note_id = args.get("note_id")
    if not isinstance(note_id, str) or not note_id:
        return {"text": "", "total": 0, "error": "missing note_id"}

    # blob 来源
    calls_blob = args.get("_calls_blob")
    if not calls_blob:
        from .check_plagiarism_video import _get_blob_index
        calls_blob = _get_blob_index().get(note_id, "")

    if calls_blob:
        backend = OfflineBackend(calls_blob, note_id=note_id)
    else:
        backend = OnlineBackend()

    # 拿 author_user_id
    author_user_id = args.get("user_id", "")
    if not author_user_id:
        note_info = backend.call("queryNoteInfo", noteId=note_id)
        if note_info:
            info = note_info.get("noteInfo", note_info)
            detail = info.get("noteDetailInfo", {}) if isinstance(info, dict) else {}
            author_user_id = detail.get("userId", "") if isinstance(detail, dict) else ""

    data = backend.call("getNoteCommentApp", noteId=note_id, pageable="{}")
    if not data:
        online = OnlineBackend()
        data = online.call("getNoteCommentApp", noteId=note_id, pageable="{}")
    result = _extract_all_comments(data, author_user_id)
    text = render_comments_text(result)

    return {"text": text, "total": result["total"]}


from .base_tool import BaseTool


class GetCommentsTool(BaseTool):
    name = TOOL_NAME
    description = DESCRIPTION
    brief = "完整评论列表：获取笔记所有评论（基础特征里只有作者评论和置顶评论）。"
    when = "需要查看笔记完整评论区；判断评论区是否有营销导流；查看用户评论区互动情况。"
    input_schema = {"note_id": "待审笔记ID"}
    output_schema = {
        "text": "string  完整评论区 Markdown",
        "total": "int  评论总数",
    }
    cost = COST
    image_mode = "inline"
    RESULT_TEMPLATE = "【笔记评论】\n{{ text }}"

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return run(args)


TOOL = GetCommentsTool()
