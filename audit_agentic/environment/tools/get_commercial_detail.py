"""商业详情工具：查询笔记关联的商品 SKU。

涉及服务:
- GetNoteItemDetail: 笔记关联商品的 SKU 信息（标题/品牌/是否可购买）
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..services._base import DataBackend, OnlineBackend, OfflineBackend
from .base_tool import BaseTool


def _extract_item_detail(data: Optional[dict]) -> Dict[str, Any]:
    """从 GetNoteItemDetail 返回中提取商品信息。"""
    if not data:
        return {"has_item": False, "items": []}

    item_list = data.get("itemSkuDetailList", [])
    if isinstance(item_list, str):
        try:
            item_list = json.loads(item_list)
        except (ValueError, TypeError):
            item_list = []

    if not isinstance(item_list, list) or not item_list:
        return {"has_item": False, "items": []}

    items = []
    for item in item_list:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except (ValueError, TypeError):
                continue
        if not isinstance(item, dict):
            continue

        title = ""
        brand = item.get("brandName", "")
        fields = item.get("itemDataFieldDtos", [])
        if isinstance(fields, list):
            for field in fields:
                if isinstance(field, dict) and field.get("dataField") == "NAME":
                    contents = field.get("contents", [])
                    if contents and isinstance(contents, list):
                        title = contents[0].get("content", "") if isinstance(contents[0], dict) else ""

        # 商品图片
        item_images = []
        for field in fields:
            if isinstance(field, dict) and field.get("dataField") == "IMAGES":
                for c in field.get("contents", []):
                    if isinstance(c, dict) and c.get("content") and c.get("contentType") == "IMAGE":
                        item_images.append(c["content"])

        items.append({
            "title": title,
            "brand": brand,
            "buyable": item.get("buyable", False),
            "images": item_images,
        })

    return {"has_item": True, "items": items}


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


def render_commercial_detail_text(
    result: Dict[str, Any],
    max_images_per_item: int = 2,
) -> str:
    """渲染商业详情为文本。"""
    lines = ["## 商业详情", ""]

    if result.get("has_item"):
        lines.append("### 关联商品")
        for i, item in enumerate(result["items"], 1):
            lines.append(f"- 商品{i}: {item['title']}")
            if item.get("brand"):
                lines.append(f"  品牌: {item['brand']}")
            lines.append(f"  可购买: {'是' if item.get('buyable') else '否'}")
            images = item.get("images", [])
            shown_images = _limit_items(images, max_images_per_item)
            if shown_images:
                suffix = ""
                if len(images) > len(shown_images):
                    suffix = f" （共{len(images)}张，展示{len(shown_images)}张）"
                lines.append(f"  商品图: " + " ".join(["<image>"] * len(shown_images)) + suffix)
        lines.append("")
    else:
        lines.append("### 关联商品")
        lines.append("- 无")
        lines.append("")

    return "\n".join(lines)


class GetCommercialDetailTool(BaseTool):
    name = "get_commercial_detail"
    description = (
        "查询笔记的商业详情：关联商品 SKU（标题/品牌/是否可购买）。"
        "用于判断笔记是否挂商品、商品品牌信息。"
    )
    brief = "商业详情：查询笔记关联的商品SKU信息。"
    when = "需要查看笔记关联的商品详情；候选标签涉及营销/带货/商品相关。"
    input_schema = {
        "note_id": "待审笔记ID",
        "max_images_per_item": "每个商品最多展示图片数，默认2，-1表示全量",
    }
    output_schema = {
        "text": "string  商业详情 Markdown（关联商品）",
        "has_item": "bool  是否关联商品",
        "items": "list  商品SKU列表",
        "display_images": "list  商品图片路径（与 text 中 <image> 对齐）",
        "max_images_per_item": "int  本次使用的每商品图片数上限",
    }
    cost = 1
    image_mode = "inline"
    RESULT_TEMPLATE = "【商业详情】\n{{ text }}"

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        note_id = args.get("note_id")
        if not isinstance(note_id, str) or not note_id:
            return {"text": "", "has_item": False, "items": [], "display_images": [], "error": "missing note_id"}

        calls_blob = args.get("_calls_blob")
        if not calls_blob:
            from .check_plagiarism_video import _get_blob_index
            calls_blob = _get_blob_index().get(note_id, "")

        if calls_blob:
            backend = OfflineBackend(calls_blob, note_id=note_id)
        else:
            backend = OnlineBackend()

        max_images_per_item = _get_int_limit(
            args,
            "max_images_per_item",
            "COMMERCIAL_DETAIL_MAX_IMAGES_PER_ITEM",
            2,
        )

        item_data = backend.call("GetNoteItemDetail", noteId=note_id)
        result = _extract_item_detail(item_data)
        text = render_commercial_detail_text(result, max_images_per_item=max_images_per_item)

        display_images = []
        for item in result.get("items", []):
            item["images"] = _limit_items(item.get("images", []), max_images_per_item)
            display_images.extend(item["images"])

        return {
            "text": text,
            "has_item": result["has_item"],
            "items": result["items"],
            "display_images": display_images,
            "max_images_per_item": max_images_per_item,
        }


TOOL = GetCommercialDetailTool()
