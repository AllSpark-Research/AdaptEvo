"""图文搬运检测工具。

逻辑:
1. BatchQueryStealInfoByNoteId(离线) → 相似笔记列表（noteId + 匹配图 + 作者 + 状态）
2. QuerySimilarInfoBetweenTwoNote(离线) → 当前笔记每张图 → 相似笔记图的映射
3. 统计：当前笔记 N 张图，去重后有几张被匹配到（放最前面）
4. 对每条相似笔记调 get_note_detail(在线) → 完整内容
5. QuerySimilarNotesTextsComparisonService(在线) → 文本重叠 ratio + 片段

返回:
{
    "text": str,
    "display_images": [str],
    "similar_note_ids": [str],
}
"""

from __future__ import annotations

import json
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..services._base import DataBackend, OnlineBackend, OfflineBackend


MAX_DISPLAY_IMAGES = 24


def _truncate_display_images(
    text: str,
    display_images: List[str],
    max_images: int = MAX_DISPLAY_IMAGES,
) -> tuple[str, List[str]]:
    """Keep image placeholders aligned while limiting multimodal payload size."""
    if max_images < 0 or len(display_images) <= max_images:
        return text, display_images

    kept_placeholders = 0
    truncated_lines = []
    for line in text.split("\n"):
        placeholder_count = line.count("<image>")
        if placeholder_count == 0:
            truncated_lines.append(line)
            continue

        remaining = max_images - kept_placeholders
        if remaining <= 0:
            # The whole line refers only to images that are no longer returned.
            continue

        if placeholder_count <= remaining:
            truncated_lines.append(line)
            kept_placeholders += placeholder_count
            continue

        # A line containing multiple placeholders can straddle the limit. Keep
        # the allowed prefix and remove only the surplus placeholders.
        marker = "__AUDIT_IMAGE_PLACEHOLDER__"
        for _ in range(remaining):
            line = line.replace("<image>", marker, 1)
        line = line.replace("<image>", "").replace(marker, "<image>")
        if line.strip():
            truncated_lines.append(line)
        kept_placeholders = max_images

    notice = (
        f"- 图片展示已截断: 原始 {len(display_images)} 张，"
        f"仅展示前 {max_images} 张"
    )
    insert_at = next(
        (
            i + 1
            for i, line in enumerate(truncated_lines)
            if line.startswith("- 相似笔记数:")
        ),
        len(truncated_lines),
    )
    truncated_lines.insert(insert_at, notice)
    return "\n".join(truncated_lines), display_images[:max_images]


def _format_ts(ts) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(int(ts) / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, OSError):
        return ""


def _extract_steal_infos(data: Optional[dict]) -> List[dict]:
    """从 BatchQueryStealInfoByNoteId 提取相似笔记列表。"""
    if not data:
        return []

    infos = data.get("stealInfos", [])
    if isinstance(infos, str):
        try:
            infos = json.loads(infos)
        except (ValueError, TypeError):
            return []

    if not isinstance(infos, list):
        return []

    # 按 noteId 去重（同一条相似笔记可能匹配多张图）
    seen = {}
    for info in infos:
        if not isinstance(info, dict):
            continue
        steal_images = info.get("stealImages", [])
        if isinstance(steal_images, str):
            try:
                steal_images = json.loads(steal_images)
            except (ValueError, TypeError):
                steal_images = []

        for img in (steal_images if isinstance(steal_images, list) else []):
            if not isinstance(img, dict):
                continue
            nid = img.get("noteId", "")
            if not nid:
                continue
            if nid not in seen:
                seen[nid] = {
                    "note_id": nid,
                    "author_name": img.get("authorName", ""),
                    "is_same_author": bool(img.get("isSameAuthor", False)),
                    "note_create_time": _format_ts(img.get("noteCreateTime")),
                    "security_status": img.get("noteSecurityStatusCn", ""),
                    "enabled": img.get("enabled", True),
                    "matched_images": [],
                }
            seen[nid]["matched_images"].append({
                "url": img.get("url", ""),
                "file_id": img.get("fileId", ""),
            })

    return list(seen.values())


def _parse_image_similar_entries(entries: List[dict]) -> Dict[str, List[dict]]:
    """解析 QuerySimilarInfoBetweenTwoNote 结果列表为统一格式。

    返回: {当前笔记 fileId: [{similar_file_id, similarity, similar_url}, ...]}
    """
    matched: Dict[str, List[dict]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        img_map = entry.get('imageSimilarInfoMap', {})
        if isinstance(img_map, str):
            try:
                img_map = json.loads(img_map)
            except (ValueError, TypeError):
                continue
        if not isinstance(img_map, dict):
            continue
        for similar_file, current_info in img_map.items():
            if isinstance(current_info, str):
                try:
                    current_info = json.loads(current_info)
                except (ValueError, TypeError):
                    continue
            if not isinstance(current_info, dict):
                continue
            current_file = current_info.get("fileId", "")
            similarity = current_info.get("similarity", 0)
            if current_file not in matched:
                matched[current_file] = []
            matched[current_file].append({
                "similar_file_id": similar_file,
                "similarity": similarity,
                "similar_url": current_info.get("url", ""),
            })
    return matched


def _call_similar_info_online(note_id: str, similar_note_ids: List[str]) -> List[dict]:
    """在线调 QuerySimilarInfoBetweenTwoNote，对每个相似笔记各调一次。"""
    from ..services._base import online_services_disabled, warn_online_disabled_once

    if online_services_disabled():
        warn_online_disabled_once("QuerySimilarInfoBetweenTwoNote")
        return []
    H = {"Content-Type": "application/json", "User-Agent": "contact@example.invalid"}
    url = "https://service.example.invalid"
    entries = []
    for target_id in similar_note_ids:
        try:
            resp = requests.post(url, headers=H,
                json={"params": {"originNoteId": note_id, "targetNoteId": target_id}},
                timeout=20).json()
            datas = resp.get("data", {}).get("datas", {})
            if datas:
                entries.append(datas)
        except Exception:
            continue
    return entries


def _extract_image_similar_maps(backend: DataBackend, note_id: str, similar_note_ids: Optional[List[str]] = None) -> Dict[str, List[dict]]:
    """收集所有 QuerySimilarInfoBetweenTwoNote 调用结果。

    优先从离线 blob 取，无数据则在线调用。
    返回: {当前笔记 fileId: [{similar_file_id, similarity, similar_url}, ...]}
    """
    # 离线：从 blob cache 取
    entries = []
    if hasattr(backend, '_cache'):
        entries = backend._cache.get('QuerySimilarInfoBetweenTwoNote', [])

    if entries:
        return _parse_image_similar_entries(entries)

    # 在线 fallback
    if similar_note_ids:
        online_entries = _call_similar_info_online(note_id, similar_note_ids)
        if online_entries:
            return _parse_image_similar_entries(online_entries)

    return {}


def _call_text_comparison(note_text: str, similar_text: str) -> Dict[str, Any]:
    """在线调 QuerySimilarNotesTextsComparisonService 对比两段正文。"""
    from ..services._base import online_services_disabled, warn_online_disabled_once

    if not note_text or not similar_text:
        return {}
    if online_services_disabled():
        warn_online_disabled_once("QuerySimilarNotesTextsComparisonService")
        return {}

    try:
        H = {"Content-Type": "application/json", "User-Agent": "contact@example.invalid"}
        url = "https://service.example.invalid"
        resp = requests.post(url, headers=H,
            json={"params": {"noteText": note_text, "similarNoteText": similar_text}},
            timeout=20).json()
        datas = resp.get('data', {}).get('datas', {})
        if not datas:
            return {}

        match_info = datas.get('matchInfo', '[]')
        if isinstance(match_info, str):
            match_info = json.loads(match_info)
        ratio = datas.get('ratio', '0')

        return {
            "ratio": float(ratio),
            "match_info": match_info if isinstance(match_info, list) else [],
        }
    except Exception:
        return {}


def _render_text_comparison(note_text: str, comparison: Dict[str, Any], max_snippets: int = 5) -> List[str]:
    """渲染文本重叠片段。"""
    lines = []
    ratio = comparison.get("ratio", 0)
    lines.append(f"- 文本重叠比例: {ratio:.1%}")

    match_info = comparison.get("match_info", [])
    # 只展示长度 >= 5 的片段
    long_matches = [m for m in match_info if m.get("length", 0) >= 5]

    if long_matches:
        lines.append(f"- 重叠片段:")
        for m in long_matches[:max_snippets]:
            start = int(m.get("noteTextStart", 0))
            end = int(m.get("noteTextEnd", 0))
            snippet = note_text[start:end]
            if len(snippet) > 50:
                snippet = snippet[:50] + "..."
            lines.append(f"  \"{snippet}\"")

    return lines


def check_plagiarism_image(backend: DataBackend, note_id: str) -> Dict[str, Any]:
    """图文搬运检测主入口。"""
    # 1. BatchQueryStealInfoByNoteId → 相似笔记列表
    steal_data = backend.call("BatchQueryStealInfoByNoteId",
        noteIds=[note_id], orderBySimilarity="true",
        filterDisabledUsers="true", needSameManTag="false",
        useSimilarityQueryV2="false")
    steal_infos = _extract_steal_infos(steal_data)

    if not steal_infos:
        return {
            "text": "## 搬运检测\n\n- 未匹配到相似笔记\n",
            "display_images": [],
            "similar_note_ids": [],
        }

    # 2. QuerySimilarInfoBetweenTwoNote → 图片映射
    # key=相似笔记图, value=当前笔记图 → 反转为：当前笔记图 → [相似笔记图]
    similar_note_ids = [s["note_id"] for s in steal_infos]
    image_maps = _extract_image_similar_maps(backend, note_id, similar_note_ids)

    # 3. 当前笔记总图数 + 正文
    note_info = backend.call("queryNoteInfo", noteId=note_id)
    total_images = 0
    current_content = ""
    if note_info:
        info = note_info.get("noteInfo", note_info)
        detail = info.get("noteDetailInfo", {}) if isinstance(info, dict) else {}
        if isinstance(detail, str):
            detail = json.loads(detail)
        images = detail.get("images", [])
        total_images = len(images) if isinstance(images, list) else 0
        current_content = detail.get("content", "") if isinstance(detail, dict) else ""

    # 被匹配的当前笔记图数（去重）
    matched_src_count = len(image_maps)

    # 按相似笔记分组：通过 BatchQuery 的 fileId 关联 QuerySimilarInfo 的 key
    # 每条 steal_info 的 fileId 是相似笔记的一张图，在 image_maps 的 value 里能找到
    # 但实际映射里同一条相似笔记可能有多张图（调用里 key 有多个）
    # 从离线 cache 或在线结果按调用顺序分组
    per_call_maps = []
    entries = []
    if hasattr(backend, '_cache'):
        entries = backend._cache.get('QuerySimilarInfoBetweenTwoNote', [])
    if not entries:
        entries = _call_similar_info_online(note_id, similar_note_ids)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        img_map = entry.get('imageSimilarInfoMap', {})
        if isinstance(img_map, str):
            try:
                img_map = json.loads(img_map)
            except (ValueError, TypeError):
                continue
        if not isinstance(img_map, dict):
            continue
        call_pairs = []
        for similar_file, current_info in img_map.items():
            if isinstance(current_info, str):
                try:
                    current_info = json.loads(current_info)
                except (ValueError, TypeError):
                    continue
            if isinstance(current_info, dict):
                call_pairs.append({
                    "similar_file_id": similar_file,
                    "current_file_id": current_info.get("fileId", ""),
                    "similarity": current_info.get("similarity", 0),
                    "current_url": current_info.get("url", ""),
                })
        per_call_maps.append(call_pairs)

    # 关联：steal_info 的 fileId 出现在哪个 call 的 similar_file_id 里
    steal_to_call = {}
    for i, info in enumerate(steal_infos):
        for img in info["matched_images"]:
            fid = img["file_id"]
            for j, pairs in enumerate(per_call_maps):
                for p in pairs:
                    if fid in p["similar_file_id"] or p["similar_file_id"] in fid:
                        steal_to_call[info["note_id"]] = j
                        break
                if info["note_id"] in steal_to_call:
                    break
            if info["note_id"] in steal_to_call:
                break

    # 4. 对每条相似笔记调 get_note_detail + 文本对比
    from ..services.note_content import get_note_detail
    online = OnlineBackend()

    similar_note_ids = [s["note_id"] for s in steal_infos]
    similar_details = {}
    text_comparisons = {}

    for sim_nid in similar_note_ids:
        # get_note_detail 拿完整内容（不展示图片，图片对比在上面已有）
        try:
            sim_detail = get_note_detail(online, sim_nid, show_images=False)
            similar_details[sim_nid] = sim_detail
        except Exception:
            similar_details[sim_nid] = None

        # 文本对比（用当前笔记正文 vs 相似笔记正文）
        sim_content = ""
        if similar_details[sim_nid]:
            # 从 text 里提取正文（在 ### 正文 后面）
            sim_text = similar_details[sim_nid].get("text", "")
            # 直接用 fetch_note_content 拿到的原始 content
            # 但 get_note_detail 返回的是渲染后的 text，没有原始 content
            # 需要在线调 queryNoteInfo 拿正文
            sim_info = online.call("queryNoteInfo", noteId=sim_nid)
            if sim_info:
                si = sim_info.get("noteInfo", sim_info)
                sd = si.get("noteDetailInfo", {}) if isinstance(si, dict) else {}
                if isinstance(sd, str):
                    sd = json.loads(sd)
                sim_content = sd.get("content", "") if isinstance(sd, dict) else ""

        if current_content and sim_content:
            comparison = _call_text_comparison(current_content, sim_content)
            if comparison:
                text_comparisons[sim_nid] = comparison

    # === 渲染 ===
    lines = ["## 搬运检测", ""]

    # 统计放最前面
    lines.append("### 匹配统计")
    lines.append(f"- 当前笔记图片数: {total_images}")
    lines.append(f"- 被匹配图片数: {matched_src_count}")
    lines.append(f"- 相似笔记数: {len(steal_infos)}")
    lines.append("")

    # 每条相似笔记
    display_images = []

    for i, info in enumerate(steal_infos):
        sim_nid = info["note_id"]
        same_author = " [同作者]" if info["is_same_author"] else ""
        enabled = " [已下架]" if not info["enabled"] else ""

        lines.append(f"### 相似笔记{i+1}{same_author}{enabled}")
        lines.append(f"- 作者: {info['author_name']}")
        lines.append(f"- 发布时间: {info['note_create_time']}")
        lines.append(f"- 审核状态: {info['security_status']}")

        # 匹配的图对（从 QuerySimilarInfoBetweenTwoNote 拿）
        call_idx = steal_to_call.get(sim_nid)
        if call_idx is not None and call_idx < len(per_call_maps):
            pairs = per_call_maps[call_idx]
            lines.append(f"- 匹配图片对 ({len(pairs)} 对):")
            for p in pairs:
                sim_pct = (1 - p["similarity"]) * 100
                sim_url = p.get("similar_image_url", "")
                if sim_url:
                    lines.append(f"  - 相似笔记图 <image> (相似度: {sim_pct:.0f}%)")
                    display_images.append(sim_url)
                else:
                    lines.append(f"  - Comparison image unavailable (similarity: {sim_pct:.0f}%)")
        else:
            # fallback: 用 BatchQuery 里的图
            for img in info["matched_images"]:
                if img["url"]:
                    lines.append(f"- 匹配图: <image>")
                    display_images.append(img["url"])

        # 文本重叠
        comparison = text_comparisons.get(sim_nid)
        if comparison:
            comp_lines = _render_text_comparison(current_content, comparison)
            lines.extend(comp_lines)

        # 相似笔记完整内容
        sim_detail = similar_details.get(sim_nid)
        if sim_detail and sim_detail.get("text"):
            lines.append("")
            lines.append("#### 相似笔记内容")
            for line in sim_detail["text"].split("\n"):
                lines.append(f"> {line}")
            # 相似笔记的图片也加到 display_images
            for url in sim_detail.get("display_images", []):
                display_images.append(url)

        lines.append("")

    text = "\n".join(lines)
    text, display_images = _truncate_display_images(text, display_images)

    return {
        "text": text,
        "display_images": display_images,
        "similar_note_ids": similar_note_ids,
    }


# ===========================================================================
# Tool SPEC
# ===========================================================================

TOOL_NAME = "check_plagiarism_image"
DESCRIPTION = (
    "图文搬运检测：查询当前图文笔记是否搬运了其他笔记。"
    "返回相似笔记列表、图片匹配统计、文本重叠分析、相似笔记完整内容。"
)
USE_WHEN = [
    "判断图文笔记是否搬运/盗用他人内容",
    "需要对比待审图文笔记和相似笔记的图片/文本差异",
    "候选标签包含 搬运/重复/非原创 类（图文队列）",
]
INPUT_SCHEMA = {"note_id": "string"}
OUTPUT_SCHEMA = {"text": "string", "display_images": "list", "similar_note_ids": "list"}
COST = 5


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    """Tool 入口。"""
    import os as _os

    note_id = args.get("note_id")
    if not isinstance(note_id, str) or not note_id:
        return {"text": "", "display_images": [], "similar_note_ids": [], "error": "missing note_id"}

    # blob 来源
    calls_blob = args.get("_calls_blob")
    if not calls_blob:
        from .check_plagiarism_video import _get_blob_index
        calls_blob = _get_blob_index().get(note_id, "")

    if calls_blob:
        backend = OfflineBackend(calls_blob, note_id=note_id)
    else:
        backend = OnlineBackend()

    result = check_plagiarism_image(backend, note_id)
    return result


from .base_tool import BaseTool


class CheckPlagiarismImageTool(BaseTool):
    name = TOOL_NAME
    description = DESCRIPTION
    brief = (
        "图文搬运检测：查询当前图文笔记是否搬运了其他笔记。"
        "返回图片匹配统计、文本重叠分析、相似笔记完整内容。"
    )
    when = "判断图文笔记是否搬运/盗用他人内容；需要对比待审图文笔记和相似笔记的图片/文本差异；候选标签含搬运/重复/非原创等（图文队列）。"
    input_schema = {"note_id": "待审笔记ID"}
    output_schema = {
        "text": "string  图文搬运检测 Markdown（含匹配统计+相似笔记内容，可含<image>）",
        "display_images": "list  与 text 中 <image> 对齐的图片路径",
        "similar_note_ids": "list  命中的相似笔记ID列表",
    }
    cost = COST
    image_mode = "inline"
    RESULT_TEMPLATE = "【图文搬运检测结果】\n{{ text }}"

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return run(args)


TOOL = CheckPlagiarismImageTool()
