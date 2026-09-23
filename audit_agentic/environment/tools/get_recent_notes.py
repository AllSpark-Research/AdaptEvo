"""用户近期笔记工具：查询笔记作者在当前笔记发布前的历史笔记。

用于判断用户是否批量搬运、内容是否同质化等。

涉及服务:
- PageNoteInfoByUserId: 分页查询用户笔记列表（支持 endTime 筛选）
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..services._base import OnlineBackend


def fetch_recent_notes(user_id: str, end_time: int, page_size: int = 20) -> List[dict]:
    """在线调 PageNoteInfoByUserId 获取用户在 end_time 之前的笔记列表。"""
    online = OnlineBackend()
    result = online.call("PageNoteInfoByUserId",
        userId=user_id, endTime=end_time, pageSize=page_size, pageNo=1)
    if not result:
        return []

    note_list = result.get("datas", "[]")
    if isinstance(note_list, str):
        try:
            note_list = json.loads(note_list)
        except (ValueError, TypeError):
            return []

    if not isinstance(note_list, list):
        return []

    notes = []
    for item in note_list:
        if not isinstance(item, dict):
            continue

        # 图片 URL 列表（字段名是 imagesList）
        images = item.get("imagesList", [])
        if isinstance(images, str):
            try:
                images = json.loads(images)
            except (ValueError, TypeError):
                images = []
        image_urls = []
        if isinstance(images, list):
            for img in images:
                if isinstance(img, dict) and img.get("url"):
                    image_urls.append(img["url"])
                elif isinstance(img, str) and img.startswith("http"):
                    image_urls.append(img)

        notes.append({
            "note_id": item.get("noteId", ""),
            "title": item.get("title", ""),
            "content": (item.get("content", "") or "")[:200],
            "type": item.get("type", ""),
            "create_time": item.get("createTime"),
            "enabled": item.get("enabled", True),
            "security_status": item.get("securityStatusCn", "") or item.get("businessStatusCn", ""),
            "views": item.get("views", 0),
            "likes": item.get("likes", 0),
            "fav_count": item.get("favCount", 0),
            "comments_total": item.get("commentsTotal", 0),
            "share_count": item.get("shareCount", 0),
            "has_item": item.get("hasItem", False),
            "images": image_urls,
        })

    return notes


def _get_int_limit(args: Dict[str, Any], key: str, env_key: str, default: int) -> int:
    value = args.get(key)
    if value is None:
        value = os.environ.get(env_key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _limit_items(items: List[Any], limit: int) -> List[Any]:
    if limit < 0:
        return list(items)
    return list(items[:limit])


def render_recent_notes_text(
    notes: List[dict],
    max_notes: int = 5,
    max_images_per_note: int = 4,
) -> str:
    """渲染近期笔记列表为文本。"""
    if not notes:
        return "## 用户近期笔记\n\n- 无\n"

    shown_notes = _limit_items(notes, max_notes)
    lines = ["## 用户近期笔记", f"（共 {len(notes)} 篇，展示 {len(shown_notes)} 篇）", ""]
    for i, n in enumerate(shown_notes, 1):
        title = n.get("title", "") or "(无标题)"
        content = n.get("content", "")
        enabled = n.get("enabled", True)
        security = n.get("security_status", "")

        status_parts = []
        if not enabled:
            status_parts.append("已下架")
        if security and security not in ("机器通过", "非商业"):
            status_parts.append(security)
        status_str = f" [{', '.join(status_parts)}]" if status_parts else ""

        lines.append(f"- 笔记{i}: {title}{status_str}")
        if content:
            lines.append(f"  正文: {content[:100]}{'...' if len(content) > 100 else ''}")

        images = n.get("images", [])
        shown_images = _limit_items(images, max_images_per_note)
        if shown_images:
            suffix = ""
            if len(images) > len(shown_images):
                suffix = f" （共{len(images)}张，展示{len(shown_images)}张）"
            lines.append(f"  图片: " + " ".join(["<image>"] * len(shown_images)) + suffix)

        fav = n.get("fav_count", 0)
        comments = n.get("comments_total", 0)
        shares = n.get("share_count", 0)
        views = n.get("views", 0)
        likes = n.get("likes", 0)

        meta_parts = []
        if views:
            meta_parts.append(f"阅读{views}")
        if likes:
            meta_parts.append(f"赞{likes}")
        if fav:
            meta_parts.append(f"藏{fav}")
        if comments:
            meta_parts.append(f"评{comments}")
        if shares:
            meta_parts.append(f"转{shares}")
        if security:
            meta_parts.append(security)
        if meta_parts:
            lines.append(f"  ({', '.join(meta_parts)})")
    if len(notes) > len(shown_notes):
        lines.append(f"- 已截断: 还有 {len(notes) - len(shown_notes)} 篇近期笔记未展示")
    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# Tool SPEC
# ===========================================================================

TOOL_NAME = "get_recent_notes"
DESCRIPTION = (
    "查询笔记作者在当前笔记发布前的近期发布列表。"
    "返回用户历史笔记标题和正文摘要，用于判断是否批量搬运或内容同质化。"
)
USE_WHEN = [
    "怀疑用户批量搬运",
    "需要查看作者其他笔记内容",
    "判断用户是否同质化发布",
]
INPUT_SCHEMA = {
    "note_id": "string",
    "max_notes": "int optional, default 5, -1 means all",
    "max_images_per_note": "int optional, default 4, -1 means all",
}
OUTPUT_SCHEMA = {
    "text": "string",
    "count": "int",
    "shown_count": "int",
    "display_images": "list",
    "note_ids": "list",
}
COST = 2


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    """Tool 入口：给 note_id，返回用户在当前笔记发布前的近期笔记列表。"""
    note_id = args.get("note_id")
    user_id = args.get("user_id", "")

    if not isinstance(note_id, str) or not note_id:
        return {"text": "", "count": 0, "error": "missing note_id"}

    # 从 queryNoteInfo 拿 userId 和 createTime
    online = OnlineBackend()
    note_info = online.call("queryNoteInfo", noteId=note_id)
    if not note_info:
        return {"text": "- 无法获取笔记信息\n", "count": 0}

    info = note_info.get("noteInfo", note_info)
    detail = info.get("noteDetailInfo", {}) if isinstance(info, dict) else {}
    if not user_id:
        user_id = detail.get("userId", "") if isinstance(detail, dict) else ""
    create_time = detail.get("createTime") if isinstance(detail, dict) else None

    if not user_id:
        return {"text": "- 无法获取用户ID\n", "count": 0}
    if not create_time:
        return {"text": "- 无法获取笔记发布时间\n", "count": 0}

    max_notes = _get_int_limit(args, "max_notes", "RECENT_NOTES_MAX_NOTES", 5)
    max_images_per_note = _get_int_limit(
        args,
        "max_images_per_note",
        "RECENT_NOTES_MAX_IMAGES_PER_NOTE",
        4,
    )

    notes = fetch_recent_notes(user_id, int(create_time), page_size=20)

    # 排除当前笔记自身
    notes = [n for n in notes if n.get("note_id") != note_id]

    shown_notes = _limit_items(notes, max_notes)
    text = render_recent_notes_text(
        notes,
        max_notes=max_notes,
        max_images_per_note=max_images_per_note,
    )

    display_images = []
    for n in shown_notes:
        for url in _limit_items(n.get("images", []), max_images_per_note):
            display_images.append(url)

    note_ids = [n.get("note_id", "") for n in shown_notes]

    return {
        "text": text,
        "count": len(notes),
        "shown_count": len(shown_notes),
        "display_images": display_images,
        "note_ids": note_ids,
        "max_notes": max_notes,
        "max_images_per_note": max_images_per_note,
    }


from .base_tool import BaseTool


class GetRecentNotesTool(BaseTool):
    name = TOOL_NAME
    description = DESCRIPTION
    brief = "用户近期笔记：查询笔记作者在当前笔记发布前的历史笔记，含审核状态、互动数据、图片。"
    when = "怀疑用户批量搬运；需要查看作者其他笔记内容；判断用户是否同质化发布。"
    input_schema = {
        "note_id": "待审笔记ID",
        "max_notes": "最多展示近期笔记数，默认5，-1表示全量",
        "max_images_per_note": "每篇近期笔记最多展示图片数，默认4，-1表示全量",
    }
    output_schema = {
        "text": "string  用户近期笔记 Markdown（含标题/正文摘要/互动，可含<image>）",
        "count": "int  近期笔记总数",
        "shown_count": "int  实际展示的近期笔记数",
        "display_images": "list  与 text 中 <image> 对齐的图片路径",
        "note_ids": "list  实际展示的近期笔记ID列表",
        "max_notes": "int  本次使用的笔记数上限",
        "max_images_per_note": "int  本次使用的每篇图片数上限",
    }
    cost = COST
    image_mode = "inline"
    RESULT_TEMPLATE = "【用户近期笔记】\n{{ text }}"

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return run(args)


TOOL = GetRecentNotesTool()
