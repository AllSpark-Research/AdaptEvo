"""搬运检测工具（视频非原创队列）。

逻辑:
1. NoteEvidenceService → 拿到 similarNoteId + denseFrameSimilarRatio
2. 用 similarNoteId 调 get_note_detail → 拿到相似笔记完整信息（text + images）
3. GetUnoriginalText → 非原创文本标注

返回:
{
    "raw": {...},              # 搬运检测原始数据
    "text": str,              # 搬运检测 Markdown 文本
    "display_images": [str],  # text 中 <image> 对应的 URL 列表
    "all_images": [str],      # 相似笔记全量图片 URL
}
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..services._base import DataBackend, OnlineBackend


def _extract_evidence(data: Optional[dict]) -> Dict[str, Any]:
    """从 NoteEvidenceService 返回中提取相似笔记信息和相似度指标。"""
    if not data:
        return {}

    evidences = data.get("evidences", [])
    if isinstance(evidences, str):
        try:
            evidences = json.loads(evidences)
        except (ValueError, TypeError):
            evidences = []

    similar_info = {}
    for ev in (evidences if isinstance(evidences, list) else []):
        for ei in ev.get("evidenceInfos", []):
            if isinstance(ei, str):
                try:
                    ei = json.loads(ei)
                except (ValueError, TypeError):
                    continue
            for info in ei.get("infos", []):
                if isinstance(info, str):
                    try:
                        info = json.loads(info)
                    except (ValueError, TypeError):
                        continue
                if isinstance(info, dict) and info.get("similarNoteId"):
                    similar_info = {
                        "similar_note_id": info.get("similarNoteId", ""),
                        "dense_frame_similar_ratio": info.get("denseFrameSimilarRatio"),
                        "similar_score": info.get("similarScore"),
                    }
                    break
            if similar_info:
                break
        if similar_info:
            break

    return similar_info


def _extract_unoriginal_text(data: Optional[dict]) -> str:
    """从 GetUnoriginalText 返回中提取非原创文本。"""
    if not data:
        return ""
    result = data.get("similarText", "")
    if isinstance(result, list):
        return " ".join(str(x) for x in result)
    return str(result) if result else ""


def _query_factor_nonoriginal(note_id: str) -> Dict[str, Any]:
    """调因子 hitEcologyNonoriginalNoteVideoForSendAuditFuben 获取相似笔记信息。"""
    import sys
    # 优先用项目内的 query_factor（environment/services/），fallback example_user 原始路径
    _local_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "services")
    _example_user_path = "/path/to/local-assets"
    for _p in [_local_path, _example_user_path]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from query_factor import query_factor, extract_result

    try:
        resp = query_factor(
            "hitEcologyNonoriginalNoteVideoForSendAuditFuben",
            {"noteId": note_id},
        )
        result = extract_result(resp)
        if not result:
            return {}
        if isinstance(result, str):
            result = json.loads(result)
        if not isinstance(result, dict):
            return {}
        similar_note_id = result.get("similarNoteId", "")
        if not similar_note_id:
            return {}
        return {
            "similar_note_id": similar_note_id,
            "dense_frame_similar_ratio": result.get("denseFrameSimilarRatio"),
            "similar_score": result.get("similarScore"),
        }
    except Exception:
        return {}


def _fetch_ocr_combine(backend: DataBackend, note_id: str) -> str:
    """获取笔记的视频 OCR 全文（ocrCombine）。"""
    from ..services._base import OfflineBackend

    if isinstance(backend, OfflineBackend):
        resource = backend.call("QueryNoteResourceHistoryService", noteId=note_id)
    else:
        note_info = backend.call("queryNoteInfo", noteId=note_id)
        history_id = ""
        if note_info:
            detail = note_info.get("noteInfo", note_info)
            if isinstance(detail, dict):
                detail = detail.get("noteDetailInfo", detail)
            if isinstance(detail, dict):
                history_id = detail.get("historyId", "")
        online = OnlineBackend()
        resource = online.call("QueryNoteResourceHistoryService", noteId=note_id, historyId=history_id)

    if not resource:
        return ""
    resource_list = resource.get("resourceList", [])
    if isinstance(resource_list, str):
        try:
            resource_list = json.loads(resource_list)
        except (ValueError, TypeError):
            return ""
    if not isinstance(resource_list, list) or not resource_list:
        return ""
    r0 = resource_list[0]
    if isinstance(r0, str):
        try:
            r0 = json.loads(r0)
        except (ValueError, TypeError):
            return ""
    ocr_combine = r0.get("ocrCombine", "") if isinstance(r0, dict) else ""
    if isinstance(ocr_combine, str) and ocr_combine.startswith("["):
        try:
            parsed = json.loads(ocr_combine)
            if isinstance(parsed, list):
                ocr_combine = " ".join(str(x) for x in parsed)
        except (ValueError, TypeError):
            pass
    return ocr_combine if isinstance(ocr_combine, str) else ""


def _call_unoriginal_text_online(ocr_text: str, similar_ocr_text: str) -> str:
    """在线调 GetUnoriginalText，返回 similarText。"""
    from ..services._base import online_services_disabled, warn_online_disabled_once

    if not ocr_text or not similar_ocr_text or ocr_text == "null" or similar_ocr_text == "null":
        return ""
    if online_services_disabled():
        warn_online_disabled_once("GetUnoriginalText")
        return ""
    try:
        import requests
        H = {"Content-Type": "application/json", "User-Agent": "contact@example.invalid"}
        url = "https://service.example.invalid"
        resp = requests.post(url, headers=H,
            json={"params": {"ocrTextCombine": ocr_text, "similarOcrTextCombine": similar_ocr_text}},
            timeout=20).json()
        datas = resp.get("data", {}).get("datas", {})
        if not datas:
            return ""
        return datas.get("similarText", "")
    except Exception:
        return ""




def fetch_plagiarism(backend: DataBackend, note_id: str) -> Dict[str, Any]:
    """拉取搬运检测原始数据。"""
    from ..services._base import OfflineBackend

    # 离线模式：从 blob 解析 NoteEvidenceService，没有则 fallback 因子接口
    if isinstance(backend, OfflineBackend):
        evidence = backend.call("NoteEvidenceService", noteId=note_id)
        similar_info = _extract_evidence(evidence)
        if not similar_info:
            similar_info = _query_factor_nonoriginal(note_id)
    else:
        similar_info = _query_factor_nonoriginal(note_id)

    # 2. 用 similarNoteId 调 get_note_detail 拿相似笔记完整信息
    similar_note_detail = None
    similar_note_id = similar_info.get("similar_note_id", "")
    if similar_note_id:
        from ..services.note_content import get_note_detail
        online = OnlineBackend()
        similar_note_detail = get_note_detail(
            online,
            similar_note_id,
            include_original_statement=True,
            max_display_frames=8,
        )

    # 3. GetUnoriginalText：在线调（用两段 OCR）
    unoriginal_text = ""
    if similar_note_id:
        current_ocr = _fetch_ocr_combine(backend, note_id)
        similar_ocr = _fetch_ocr_combine(OnlineBackend(), similar_note_id)
        unoriginal_text = _call_unoriginal_text_online(current_ocr, similar_ocr)

    return {
        "similar_info": similar_info,
        "similar_note_detail": similar_note_detail,
        "unoriginal_text": unoriginal_text,
    }


def render_plagiarism_text(raw: Dict[str, Any]) -> str:
    """将 fetch_plagiarism 返回组装成 Markdown 格式字符串。"""
    similar_info = raw.get("similar_info", {})
    similar_detail = raw.get("similar_note_detail")
    unoriginal = raw.get("unoriginal_text", "")

    lines = []
    lines.append("## 搬运检测")
    lines.append("")

    # 相似度指标
    similar_note_id = similar_info.get("similar_note_id", "")
    if similar_note_id:
        ratio = similar_info.get("dense_frame_similar_ratio")
        if ratio is not None:
            lines.append("### 相似度指标")
            if isinstance(ratio, (int, float)):
                ratio_str = f"{float(ratio):.2%}"
            else:
                ratio_str = str(ratio)
            lines.append(f"- 无成本加工画面相似比例: {ratio_str}")
            lines.append("")

        # 相似笔记完整内容
        if similar_detail and similar_detail.get("text"):
            lines.append("### 相似笔记内容")
            lines.append("")
            for line in similar_detail["text"].split("\n"):
                lines.append(f"> {line}")
            lines.append("")
    else:
        lines.append("- 未匹配到相似笔记")
        lines.append("")

    # 非原创文本
    if unoriginal:
        lines.append("### 非原创文本标注")
        lines.append(unoriginal)
        lines.append("")

    return "\n".join(lines)


def check_plagiarism(backend: DataBackend, note_id: str) -> Dict[str, Any]:
    """搬运检测工具主入口，返回统一格式。

    返回:
    {
        "text": str,              # 搬运检测 Markdown 文本
        "display_images": [str],  # text 中相似笔记 <image> 对应 URL
        "all_images": [str],      # 相似笔记全量图片 URL
    }
    """
    raw = fetch_plagiarism(backend, note_id)
    text = render_plagiarism_text(raw)

    # images 来自相似笔记的 get_note_detail 结果
    similar_detail = raw.get("similar_note_detail")
    if similar_detail:
        display_images = similar_detail.get("display_images", [])
        all_images = similar_detail.get("all_images", [])
    else:
        display_images = []
        all_images = []

    # get_note_detail may keep placeholders for all source keyframes in text
    # while display_images is capped to cover + 8 keyframes. Remove only the
    # surplus placeholders so the raw cached result is already one-to-one.
    image_count = len(display_images)
    parts = text.split("<image>")
    if len(parts) - 1 > image_count:
        aligned_text = parts[0]
        for index, part in enumerate(parts[1:]):
            aligned_text += ("<image>" if index < image_count else "") + part
        text = aligned_text

    similar_note_id = raw.get("similar_info", {}).get("similar_note_id", "")

    return {
        "similar_note_id": similar_note_id,
        "text": text,
        "display_images": display_images,
        "all_images": all_images,
    }


# ===========================================================================
# Tool SPEC 定义
# ===========================================================================

TOOL_NAME = "check_plagiarism_video"
DESCRIPTION = (
    "搬运检测：查询当前笔记是否搬运了其他笔记。"
    "返回相似笔记的完整内容对比、无成本加工画面相似比例、非原创文本标注。"
)
USE_WHEN = [
    "判断笔记是否搬运/盗用他人内容",
    "需要对比待审笔记和相似笔记的差异",
    "候选标签包含 搬运/重复/非原创 类",
]
INPUT_SCHEMA = {"note_id": "string"}
OUTPUT_SCHEMA = {"text": "string", "display_images": "list", "all_images": "list"}
COST = 3




import os as _os
import pickle as _pickle
import threading as _threading

# 默认缓存路径：environment/cache/plagiarism_cache.pkl (相对本模块)
_DEFAULT_CACHE_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "cache", "plagiarism_cache.pkl",
)


_blob_index_lock = _threading.Lock()
_blob_index: Optional[Dict[str, str]] = None


def _get_blob_index() -> Dict[str, str]:
    """懒加载 blob 索引（首次调用时从 BLOB_INDEX_PATH 环境变量指定的 pkl 加载）。"""
    from ..data_access_mode import online_data_access_enabled

    if online_data_access_enabled():
        return {}
    global _blob_index
    if _blob_index is not None:
        return _blob_index
    with _blob_index_lock:
        if _blob_index is not None:
            return _blob_index
        path = _os.environ.get("BLOB_INDEX_PATH", "")
        if path and _os.path.exists(path):
            with open(path, "rb") as f:
                _blob_index = _pickle.load(f)
        else:
            _blob_index = {}
        return _blob_index


_result_cache_lock = _threading.Lock()
_result_cache = None


def _get_result_cache():
    from ..data_access_mode import online_data_access_enabled

    if online_data_access_enabled():
        return {}
    global _result_cache
    if _result_cache is not None:
        return _result_cache
    with _result_cache_lock:
        if _result_cache is not None:
            return _result_cache
        path = _os.environ.get("PLAGIARISM_CACHE_PATH") or _DEFAULT_CACHE_PATH
        if path and _os.path.exists(path):
            with open(path, "rb") as f:
                _result_cache = _pickle.load(f)
        else:
            _result_cache = {}
        return _result_cache


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    """Tool 入口：给 note_id，返回搬运检测完整结果。

    环境变量配置（启动时设置，工具内部自动读取）：
        BLOB_INDEX_PATH: blob 索引 pkl 路径（note_id → calls_blob）

    也可通过 args 直接传入（优先级更高）：
        _calls_blob: 直接传 blob

    注：IMAGE_CACHE_DIR 由 ToolExecutor 统一处理，无需工具内部处理。
    """
    from ..services._base import OfflineBackend, OnlineBackend

    note_id = args.get("note_id")
    if not isinstance(note_id, str) or not note_id:
        return {"text": "", "display_images": [], "all_images": [], "error": "missing note_id"}

    # 预计算缓存优先：命中则直接返回，完全跳过 1.6GB blob 加载
    # 显式传 _calls_blob 或 _no_cache=True 时绕过缓存
    if not args.get("_no_cache") and not args.get("_calls_blob"):
        _hit = _get_result_cache().get(note_id)
        if _hit is not None:
            return dict(_hit)

    # blob 来源：args 直传 > 环境变量索引
    calls_blob = args.get("_calls_blob")
    if not calls_blob:
        calls_blob = _get_blob_index().get(note_id, "")

    if calls_blob:
        backend = OfflineBackend(calls_blob, note_id=note_id)
    else:
        backend = OnlineBackend()

    result = check_plagiarism(backend, note_id)
    return result


from .base_tool import BaseTool


class CheckPlagiarismVideoTool(BaseTool):
    name = TOOL_NAME
    description = DESCRIPTION
    brief = (
        "视频搬运检测：查询当前视频笔记是否搬运了其他笔记。"
        "返回相似笔记完整内容对比、无成本加工画面相似比例、非原创文本标注。"
    )
    when = "判断视频笔记是否搬运/盗用他人内容；需要对比待审视频笔记和相似笔记差异；候选标签含搬运/重复/非原创等（视频队列）。"
    input_schema = {"note_id": "待审笔记ID"}
    output_schema = {
        "text": "string  搬运检测 Markdown（含相似度指标+相似笔记内容，可含<image>）",
        "display_images": "list  与 text 中 <image> 对齐的图片路径",
        "all_images": "list  相似笔记全量图片路径",
        "similar_note_id": "string  命中的相似笔记ID（无则空串）",
    }
    cost = COST
    image_mode = "inline"
    RESULT_TEMPLATE = "【视频搬运检测结果】\n{{ text }}"

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return run(args)


TOOL = CheckPlagiarismVideoTool()
