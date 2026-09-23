"""用户评论记录工具：查询用户在其他笔记下发表的评论。

用于判断用户是否为营销号/导流号（在多条笔记下发相似评论引流）。

涉及服务:
- UserCommentPicQueryService: 用户评论列表（含内容/审核状态/图片/OCR）
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..services._base import DataBackend, OfflineBackend


MAX_COMMENTS = 10


def _format_ts(ts) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(int(ts) / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, OSError):
        return ""


def _extract_user_comments(data: Optional[dict]) -> List[dict]:
    """从 UserCommentPicQueryService 返回中提取评论列表。"""
    if not data:
        return []

    comments = data.get("commentList", [])
    if isinstance(comments, str):
        try:
            comments = json.loads(comments)
        except (ValueError, TypeError):
            return []

    if not isinstance(comments, list):
        return []

    results = []
    for c in comments[:MAX_COMMENTS]:
        if not isinstance(c, dict):
            continue

        # 提取图片URL
        images = []
        for res in c.get("resourceList", []):
            if isinstance(res, dict) and res.get("cover"):
                images.append(res["cover"])

        # 提取图片OCR文字
        ocr_texts = []
        for item in c.get("urlWithOcrList", []):
            if isinstance(item, dict):
                for ocr in item.get("ocrList", []):
                    if isinstance(ocr, dict) and ocr.get("text"):
                        ocr_texts.append(ocr["text"])

        results.append({
            "content": c.get("commentContent", ""),
            "time": _format_ts(c.get("commentTime")),
            "target_note_id": c.get("targetNoteId", ""),
            "audit_status": c.get("commentAuditStatus", ""),
            "images": images,
            "ocr_text": " ".join(ocr_texts) if ocr_texts else "",
            "author_liked": c.get("authorLiked", False),
            "author_reply": c.get("authorReply", False),
        })

    return results


def render_user_comments_text(comments: List[dict]) -> str:
    """渲染用户评论为文本。"""
    if not comments:
        return "## 用户评论记录\n\n- 无评论记录\n"

    lines = ["## 用户评论记录", f"（共 {len(comments)} 条）", ""]

    for c in comments:
        content = c.get("content", "")[:200]
        time_str = c.get("time", "")
        status = c.get("audit_status", "")
        target = c.get("target_note_id", "")
        images = c.get("images", [])
        ocr = c.get("ocr_text", "")

        lines.append(f"- 评论: {content}")
        if time_str:
            lines.append(f"  时间: {time_str}")
        if status:
            lines.append(f"  审核状态: {status}")
        if target:
            lines.append(f"  目标笔记: {target}")
        if images:
            lines.append(f"  图片: " + " ".join(["<image>"] * len(images)))
        if ocr:
            lines.append(f"  图片OCR: {ocr[:100]}")
        lines.append("")

    return "\n".join(lines)


# ===========================================================================
# Tool SPEC
# ===========================================================================

TOOL_NAME = "get_user_comments"
DESCRIPTION = (
    "查询用户在其他笔记下发表的评论。"
    "返回评论内容、时间、审核状态、图片及OCR，用于判断是否为营销/导流账号。"
)
USE_WHEN = [
    "怀疑用户在多条笔记下发导流评论",
    "需要查看用户评论行为模式",
    "判断用户是否为营销号",
]
INPUT_SCHEMA = {"note_id": "string"}
OUTPUT_SCHEMA = {"text": "string", "count": "int", "display_images": "list"}
COST = 1


def _fetch_user_comments_online(note_id: str) -> Optional[Dict[str, Any]]:
    """在线调 UserCommentPicQueryService，需先从 queryNoteInfo 拿 userId。"""
    from ..services._base import OnlineBackend
    online = OnlineBackend()

    note_info = online.call("queryNoteInfo", noteId=note_id)
    if not note_info:
        return None
    info = note_info.get("noteInfo", note_info)
    detail = info.get("noteDetailInfo", {}) if isinstance(info, dict) else {}
    user_id = detail.get("userId", "") if isinstance(detail, dict) else ""
    if not user_id:
        return None

    return online.call("UserCommentPicQueryService",
        userId=user_id, auditStatus=0, needGender=False,
        needAgeInfo=True, needStatus=False, needIsAuthor=False,
        needOcrInfo=True, pageNum=1, pageSize=MAX_COMMENTS, onlyNoteComment=True)


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    """Tool 入口：给 note_id，获取用户评论记录（离线优先，fallback 在线）。"""
    note_id = args.get("note_id")
    if not isinstance(note_id, str) or not note_id:
        return {"text": "", "count": 0, "display_images": [], "error": "missing note_id"}

    calls_blob = args.get("_calls_blob")
    if not calls_blob:
        from .check_plagiarism_video import _get_blob_index
        calls_blob = _get_blob_index().get(note_id, "")

    if calls_blob:
        backend = OfflineBackend(calls_blob, note_id=note_id)
        data = backend.call("UserCommentPicQueryService")
        if not data:
            data = _fetch_user_comments_online(note_id)
    else:
        data = _fetch_user_comments_online(note_id)

    if not data:
        return {"text": "- 未查询到用户评论记录\n", "count": 0, "display_images": []}

    comments = _extract_user_comments(data)
    text = render_user_comments_text(comments)

    display_images = []
    for c in comments:
        display_images.extend(c.get("images", []))

    return {"text": text, "count": len(comments), "display_images": display_images}


from .base_tool import BaseTool


class GetUserCommentsTool(BaseTool):
    name = TOOL_NAME
    description = DESCRIPTION
    brief = "用户在其他笔记下的评论：查询用户在站内其他笔记的评论记录，含内容、时间、审核状态、图片OCR。"
    when = "怀疑用户在多条笔记下发导流评论；判断是否为营销/导流账号；查看用户评论行为模式。"
    input_schema = {"note_id": "待审笔记ID"}
    output_schema = {
        "text": "string  用户评论记录 Markdown（内容/时间/审核状态/目标笔记/图片OCR，可含<image>）",
        "count": "int  评论条数",
        "display_images": "list  评论图片路径（与 text 中 <image> 对齐）",
    }
    cost = COST
    image_mode = "inline"
    RESULT_TEMPLATE = "【用户评论记录】\n{{ text }}"

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return run(args)


TOOL = GetUserCommentsTool()
