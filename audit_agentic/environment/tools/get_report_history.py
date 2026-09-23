"""用户举报记录工具：查询用户被举报且处理为"接受"的历史记录。

只返回举报成功（接受）的记录，驳回的不展示。
输出格式参考审核台：举报人、举报时间、举报原因、举报描述、图片证据。

涉及服务:
- batchQueryReportInfoService: 用户维度举报记录（含 totalCount + reportRecordInfoList）
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


def _parse_reason_json(reason_json) -> tuple[str, List[str]]:
    """从 reasonJson 解析举报描述和图片证据。"""
    if not reason_json:
        return "无", []
    if isinstance(reason_json, str):
        try:
            reason_json = json.loads(reason_json)
        except (ValueError, TypeError):
            return "无", []
    if not isinstance(reason_json, list):
        return "无", []

    desc = "无"
    images = []
    for item in reason_json:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and item.get("text"):
            desc = item["text"]
        elif item.get("type") == "image":
            img_list = item.get("image_list", [])
            if isinstance(img_list, list):
                images.extend(img_list)
    return desc, images


def _extract_report_records(data: Optional[dict]) -> tuple[int, int, List[dict]]:
    """从 batchQueryReportInfoService 返回中提取举报记录，只保留"接受"的。

    Returns: (total_all, accepted_count, accepted_records)
    """
    if not data:
        return 0, 0, []

    total = int(data.get("totalCount", 0) or 0)
    records_raw = data.get("reportRecordInfoList", [])
    if isinstance(records_raw, str):
        try:
            records_raw = json.loads(records_raw)
        except (ValueError, TypeError):
            return total, 0, []

    if not isinstance(records_raw, list):
        return total, 0, []

    results = []
    for r in records_raw:
        if not isinstance(r, dict):
            continue
        action = r.get("action", "")
        if action == "驳回":
            continue

        reason_type = r.get("reasonTypeDesc", r.get("reasonType", ""))
        second_reason = r.get("secondReasonType", "")
        reason_str = f"{reason_type}-{second_reason}" if second_reason else reason_type

        reporter = r.get("userName", "") or "未知"
        processor = r.get("processorName", "")
        desc, images = _parse_reason_json(r.get("reasonJson"))

        results.append({
            "reporter": reporter,
            "create_time": _format_ts(r.get("createTime")),
            "reason": reason_str,
            "desc": desc,
            "images": images,
            "processor": processor,
        })

    return total, len(results), results


def render_report_history_text(total: int, accepted_count: int, records: List[dict]) -> str:
    """渲染举报记录为文本（只展示接受的）。"""
    if accepted_count == 0:
        return "## 举报记录\n\n- 无有效举报记录（累计 {} 条举报均被驳回）\n".format(total)

    lines = ["## 举报记录", f"（累计 {total} 条，其中 {accepted_count} 条接受）", ""]

    for r in records:
        lines.append(f"- 举报人: {r['reporter']}")
        lines.append(f"  举报时间: {r['create_time']}")
        lines.append(f"  举报原因: {r['reason']}")
        lines.append(f"  举报描述: {r['desc']}")
        if r["images"]:
            lines.append(f"  图片证据: " + " ".join(["<image>"] * len(r["images"])))
        else:
            lines.append(f"  图片证据: 无")
        lines.append("")

    return "\n".join(lines)


# ===========================================================================
# Tool SPEC
# ===========================================================================

TOOL_NAME = "get_report_history"
DESCRIPTION = (
    "查询用户举报记录（只返回接受/成立的举报）。"
    "返回举报人、举报时间、举报原因、举报描述、图片证据。"
)
USE_WHEN = [
    "需要了解用户是否有举报成立的历史",
    "判断用户历史违规情况",
    "查看被接受的举报原因和描述",
]
INPUT_SCHEMA = {"note_id": "string"}
OUTPUT_SCHEMA = {"text": "string", "total": "int", "accepted_count": "int"}
COST = 1


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    """Tool 入口：离线读 blob；online 数据模式下直接查询服务。"""
    note_id = args.get("note_id")
    if not isinstance(note_id, str) or not note_id:
        return {"text": "", "total": 0, "accepted_count": 0, "error": "missing note_id"}

    calls_blob = args.get("_calls_blob")
    if not calls_blob:
        from .check_plagiarism_video import _get_blob_index
        calls_blob = _get_blob_index().get(note_id, "")

    if calls_blob:
        backend = OfflineBackend(calls_blob, note_id=note_id)
    else:
        backend = OnlineBackend()

    data = backend.call("batchQueryReportInfoService", noteId=note_id)
    if not data:
        data = backend.call("queryReportInfoService", noteId=note_id)

    total, accepted_count, records = _extract_report_records(data)
    text = render_report_history_text(total, accepted_count, records)

    # 收集图片 URL
    display_images = []
    for r in records:
        display_images.extend(r.get("images", []))

    return {"text": text, "total": total, "accepted_count": accepted_count, "display_images": display_images}


from .base_tool import BaseTool


class GetReportHistoryTool(BaseTool):
    name = TOOL_NAME
    description = DESCRIPTION
    brief = "用户举报记录：查询接受/成立的举报，返回举报人、时间、原因、描述、图片证据。"
    when = "需要了解用户是否有举报成立的历史；判断笔记是否被多人举报；查看举报原因和证据。"
    input_schema = {"note_id": "待审笔记ID"}
    output_schema = {
        "text": "string  举报记录 Markdown（举报人/时间/原因/描述/图片证据，可含<image>）",
        "total": "int  累计举报条数",
        "accepted_count": "int  其中接受/成立的条数",
        "display_images": "list  举报图片证据路径（与 text 中 <image> 对齐）",
    }
    cost = COST
    image_mode = "inline"
    RESULT_TEMPLATE = "【用户举报记录】\n{{ text }}"

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return run(args)


TOOL = GetReportHistoryTool()
