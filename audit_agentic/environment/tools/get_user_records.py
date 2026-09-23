"""用户资料修改记录工具：查询用户的资料变更历史。

用于判断用户是否频繁修改昵称/简介/头像来规避审核。

涉及服务:
- GetUserInfoRecordsService: 用户资料修改记录（昵称/简介/头像/背景图等变更）
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..services._base import DataBackend, OnlineBackend, OfflineBackend


def _format_ts(ts) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(int(ts) / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, OSError):
        return ""


def _extract_user_records(data: Optional[dict]) -> List[dict]:
    """从 GetUserInfoRecordsService 返回中提取修改记录。"""
    if not data:
        return []

    records = data.get("data", [])
    if isinstance(records, str):
        try:
            records = json.loads(records)
        except (ValueError, TypeError):
            return []

    if not isinstance(records, list):
        return []

    results = []
    for r in records:
        if not isinstance(r, dict):
            continue
        content = r.get("content", "")
        record_type = r.get("type", "")
        # banner_image/avatar 修改的 content 是图片 URL
        is_image = record_type in ("banner_image", "banner", "avatar", "redId_image") and (
            "http" in content or "sns-avatar" in content or "user_banner" in content
        )

        results.append({
            "type": record_type,
            "action": r.get("action", ""),
            "content": content,
            "time": _format_ts(r.get("createTime")),
            "operator": r.get("name", ""),
            "is_image": is_image,
        })

    return results


def render_user_records_text(records: List[dict]) -> str:
    """渲染用户资料修改记录为文本。"""
    if not records:
        return "## 用户资料修改记录\n\n- 无修改记录\n"

    lines = ["## 用户资料修改记录", f"（共 {len(records)} 条）", ""]

    for r in records:
        action = r.get("action", "")
        content = r.get("content", "")[:100]
        time_str = r.get("time", "")
        record_type = r.get("type", "")
        is_image = r.get("is_image", False)

        lines.append(f"- [{record_type}] {action}")
        if is_image:
            lines.append(f"  内容: <image>")
        elif content:
            lines.append(f"  内容: {content}")
        if time_str:
            lines.append(f"  时间: {time_str}")

    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# Tool SPEC
# ===========================================================================

TOOL_NAME = "get_user_records"
DESCRIPTION = (
    "查询用户资料修改记录。"
    "返回昵称/简介/头像/背景图等变更历史，用于判断用户是否频繁修改资料规避审核。"
)
USE_WHEN = [
    "怀疑用户频繁修改资料规避审核",
    "需要查看用户历史昵称/简介变更",
    "判断用户资料是否近期异常修改",
]
INPUT_SCHEMA = {"note_id": "string"}
OUTPUT_SCHEMA = {"text": "string", "count": "int"}
COST = 1


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    """Tool 入口：给 note_id，内部查 userId 后拉取修改记录。"""
    note_id = args.get("note_id")
    if not isinstance(note_id, str) or not note_id:
        return {"text": "", "count": 0, "error": "missing note_id"}

    # blob 来源
    calls_blob = args.get("_calls_blob")
    if not calls_blob:
        from .check_plagiarism_video import _get_blob_index
        calls_blob = _get_blob_index().get(note_id, "")

    if calls_blob:
        backend = OfflineBackend(calls_blob, note_id=note_id)
    else:
        backend = OnlineBackend()

    # 从 queryNoteInfo 拿 userId
    user_id = args.get("user_id", "")
    if not user_id:
        note_info = backend.call("queryNoteInfo", noteId=note_id)
        if note_info:
            info = note_info.get("noteInfo", note_info)
            detail = info.get("noteDetailInfo", {}) if isinstance(info, dict) else {}
            user_id = detail.get("userId", "") if isinstance(detail, dict) else ""

    if not user_id:
        return {"text": "- 无法获取用户ID\n", "count": 0}

    data = backend.call("GetUserInfoRecordsService", userId=user_id)
    records = _extract_user_records(data)
    text = render_user_records_text(records)

    # 收集图片 URL（按 text 中 <image> 顺序）
    display_images = []
    for r in records:
        if r.get("is_image"):
            content = r.get("content", "")
            if content.startswith("http"):
                display_images.append(content)
            # Non-URL media identifiers require a deployment-owned resolver.

    return {"text": text, "count": len(records), "display_images": display_images}


from .base_tool import BaseTool


class GetUserRecordsTool(BaseTool):
    name = TOOL_NAME
    description = DESCRIPTION
    brief = "用户资料修改记录：查询用户昵称/简介/头像/背景图等变更历史。"
    when = "怀疑用户频繁修改资料规避审核；需要查看用户历史昵称/简介变更。"
    input_schema = {"note_id": "待审笔记ID"}
    output_schema = {
        "text": "string  用户资料修改记录 Markdown（昵称/简介/头像变更史，可含<image>）",
        "count": "int  修改记录数",
        "display_images": "list  头像/背景图变更图片路径（与 text 中 <image> 对齐）",
    }
    cost = COST
    image_mode = "inline"
    RESULT_TEMPLATE = "【用户资料修改记录】\n{{ text }}"

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return run(args)


TOOL = GetUserRecordsTool()
