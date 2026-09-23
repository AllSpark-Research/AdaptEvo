"""笔记内容板块：正文/标题/标签/类型/互动数 + 图片视频资源/OCR/ASR。

涉及服务:
第一批（并行，只需 noteId）:
- queryNoteInfo: 笔记详情（标题/正文/作者/互动/标签/historyId/userId）
- QueryNoteAsrInfo: 视频 ASR 语音转文字
- MultiGetNoteTypeByNoteId: 笔记类型
- GetNoteAtStart: @提及用户
- GetNotePkVoteInfoService: PK投票信息

第二批（并行，需 historyId，来自 queryNoteInfo）:
- QueryNoteResourceService: 图片/视频资源+OCR坐标+视频关键帧抽帧
- QueryNoteResourceHistoryService: 历史版本资源
- GetNoteHistoryDetail: 历史版本详情
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ._base import DataBackend, OfflineBackend


def _extract_note_detail(note_info: Optional[dict]) -> dict:
    """从 queryNoteInfo 返回中提取笔记详情字段。"""
    if not note_info:
        return {}

    # noteInfo 可能直接是顶层，也可能在 noteInfo key 下
    info = note_info.get("noteInfo", note_info)

    detail = info.get("noteDetailInfo", {})
    user = info.get("userDetailInfo", {})
    counts = info.get("noteRelatedCountInfo", {})
    tags = info.get("noteTagInfos", [])

    images = detail.get("images", [])
    video = detail.get("video", {})

    metadata = detail.get("metadata", {})
    if isinstance(metadata, str):
        import json as _json
        try:
            metadata = _json.loads(metadata)
        except (ValueError, TypeError):
            metadata = {}

    # 视频时长
    video_duration = None
    if isinstance(video, dict):
        meta = video.get("meta", {})
        if isinstance(meta, dict):
            video_duration = meta.get("duration")

    return {
        "note_id": detail.get("noteId", ""),
        "title": detail.get("title", ""),
        "content": detail.get("content", ""),
        "user_id": detail.get("userId", "") or user.get("userId", ""),
        "history_id": detail.get("historyId", ""),
        "birth_place": metadata.get("birth_place", "") if isinstance(metadata, dict) else "",
        "create_time": detail.get("createTime"),
        "update_time": detail.get("updateTime"),
        "type": detail.get("type", ""),
        "video_duration": video_duration,
        "level": detail.get("level", ""),
        "is_top": detail.get("isTop", False),
        "cover_url": images[0].get("url", "") if images else "",
        "image_count": len(images),
        "images": [
            {
                "url": img.get("url", ""),
                "file_id": img.get("fileId", ""),
                "width": img.get("width"),
                "height": img.get("height"),
            }
            for img in images
        ],
        "video": {
            "video_url": video.get("videoUrl", ""),
            "duration": video.get("meta", {}).get("duration"),
            "width": video.get("meta", {}).get("width"),
            "height": video.get("meta", {}).get("height"),
        } if video else None,
        "author": {
            "user_id": user.get("userId", ""),
            "nickname": user.get("userNickName", ""),
            "avatar": user.get("userAvatar", ""),
            "desc": user.get("userDesc", ""),
            "fans_count": user.get("fansTotal", 0),
            "gender": user.get("gender", ""),
        },
        "counts": {
            "like": counts.get("likeCount", 0),
            "collect": counts.get("collectCount", 0),
            "comment": counts.get("commentCount", 0),
            "read": counts.get("readCount", 0),
            "share": counts.get("shareCount", 0),
            "video_play": counts.get("videoPlayCount", 0),
            "beautiful_read": counts.get("beautifulReadCount", 0),
        },
        "tags": [
            {"name": t.get("name", ""), "type": t.get("type", "")}
            for t in (tags if isinstance(tags, list) else [])
        ],
    }


def _extract_resource_ocr(resource_data: Optional[dict]) -> List[dict]:
    """从 QueryNoteResourceV2Service 返回中提取资源列表和 OCR 文本。"""
    if not resource_data:
        return []

    resource_list = resource_data.get("resourceList", resource_data.get("resources", []))
    if isinstance(resource_list, str):
        import json
        try:
            resource_list = json.loads(resource_list)
        except (ValueError, TypeError):
            return []

    if not isinstance(resource_list, list):
        return []

    results = []
    for res in resource_list:
        if not isinstance(res, dict):
            continue
        ocr_info = res.get("coverImageOcrInfo", {})
        if isinstance(ocr_info, str):
            import json
            try:
                ocr_info = json.loads(ocr_info)
            except (ValueError, TypeError):
                ocr_info = {}

        ocr_text = ""
        if isinstance(ocr_info, dict):
            ocr_items = ocr_info.get("ocrInfos", [])
            if isinstance(ocr_items, list):
                ocr_text = " ".join(
                    item.get("text", "") for item in ocr_items if isinstance(item, dict)
                )

        results.append({
            "file_id": res.get("fileId", ""),
            "url": res.get("url", ""),
            "type": res.get("type", ""),
            "ocr_text": ocr_text,
        })
    return results


def _extract_asr(asr_data: Optional[dict]) -> List[dict]:
    """从 QueryNoteAsrInfo 返回中提取 ASR 分段（含时间戳）。

    返回: [{"start": 0, "end": 7, "text": "..."}, ...]
    """
    if not asr_data:
        return []

    segments = asr_data.get("noteAsrResultWithTimestamp", [])
    if not isinstance(segments, list):
        return []

    results = []
    for seg in segments:
        if isinstance(seg, list) and len(seg) >= 3:
            try:
                results.append({
                    "start": int(seg[0]),
                    "end": int(seg[1]),
                    "text": seg[2],
                })
            except (ValueError, TypeError):
                results.append({"start": 0, "end": 0, "text": seg[2]})
        elif isinstance(seg, dict):
            results.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", ""),
            })
    return results


def _extract_note_type(type_data: Optional[dict], note_id: str) -> List[str]:
    """从 MultiGetNoteTypeByNoteId 返回中提取笔记类型标签。"""
    if not type_data:
        return []
    note_and_type = type_data.get("noteAndType", {})
    if isinstance(note_and_type, dict):
        types = note_and_type.get(note_id, [])
        if isinstance(types, list):
            return types
    return []


def _extract_at_users(at_data: Optional[dict]) -> List[dict]:
    """从 GetNoteAtStart 返回中提取 @提及用户列表。

    实际返回结构: {noteAtStarts: "[{nickname, userId}]", userIsStarMap: "{userId: bool}"}
    """
    if not at_data:
        return []

    # 实际线上格式
    at_starts = at_data.get("noteAtStarts", at_data.get("atUserList", []))
    if isinstance(at_starts, str):
        import json as _json
        try:
            at_starts = _json.loads(at_starts)
        except (ValueError, TypeError):
            at_starts = []

    star_map = at_data.get("userIsStarMap", {})
    if isinstance(star_map, str):
        import json as _json
        try:
            star_map = _json.loads(star_map)
        except (ValueError, TypeError):
            star_map = {}

    if not isinstance(at_starts, list):
        return []

    results = []
    for u in at_starts:
        if isinstance(u, dict):
            uid = u.get("userId", "")
            results.append({
                "nickname": u.get("nickname", ""),
                "user_id": uid,
                "is_star": star_map.get(uid, False) if isinstance(star_map, dict) else False,
            })
        elif isinstance(u, str):
            results.append({"nickname": u, "user_id": "", "is_star": False})
    return results


def _parse_str_list(v) -> list:
    """将可能是 JSON 字符串的值解析为 list。"""
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, TypeError):
            pass
    return []


def _extract_commercial_info(data: Optional[dict], ads_data: Optional[dict] = None) -> dict:
    """从 GetNoteInfoNewService + NoteAdsInfoService 合并提取商业属性。"""
    if not data and not ads_data:
        return {}

    data = data or {}
    ads_data = ads_data or {}

    is_group_banned = data.get("isGroupBanned", "false")
    if isinstance(is_group_banned, str):
        is_group_banned = is_group_banned.lower() == "true"

    # 广告属性：优先用 NoteAdsInfoService（更详细），fallback 到 GetNoteInfoNewService
    ads = _parse_str_list(ads_data.get("adsAttributes")) or _parse_str_list(data.get("adsAttributes"))
    sale = _parse_str_list(ads_data.get("saleAttributes")) or _parse_str_list(data.get("saleAttributes"))
    chip = _parse_str_list(data.get("chipNoteStatus"))
    unit_types = _parse_str_list(ads_data.get("unitTypes"))

    # NoteAdsInfoService 额外字段
    def _to_bool(v):
        if isinstance(v, bool):
            return v
        return str(v).lower() == "true"

    is_ad_note = _to_bool(ads_data.get("status", "false"))
    is_order_bind = _to_bool(ads_data.get("orderBindNote", "false"))
    is_shu_shop = _to_bool(ads_data.get("shuShopNote", "false"))
    is_brand_draw = _to_bool(ads_data.get("brandDrawNote", "false"))
    is_goods_cooperate = _to_bool(ads_data.get("goodsCooperateNote", "false"))
    is_experience = _to_bool(ads_data.get("experienceNote", "false"))

    return {
        "is_group_banned": bool(is_group_banned),
        "ads_attributes": ads,
        "sale_attributes": sale,
        "chip_note_status": chip,
        "unit_types": unit_types,
        "is_ad_note": is_ad_note,
        "is_order_bind": is_order_bind,
        "is_shu_shop": is_shu_shop,
        "is_brand_draw": is_brand_draw,
        "is_goods_cooperate": is_goods_cooperate,
        "is_experience": is_experience,
    }


def _extract_high_risk_ip(data: Optional[dict]) -> dict:
    """从 queryNoteHighRiskIp 返回中提取高风险IP信息。"""
    if not data:
        return {"is_high_risk": False}
    is_high_risk = data.get("isHighRiskIp", False)
    if isinstance(is_high_risk, str):
        is_high_risk = is_high_risk.lower() == "true"
    return {"is_high_risk": bool(is_high_risk)}


def _extract_ad_result(data: Optional[dict]) -> dict:
    """从 GetNoteAdResult 返回中提取广告检测结果。"""
    if not data:
        return {"is_ad": False}
    is_ad = data.get("isAdsAduit", data.get("isAdsAudit", False))
    if isinstance(is_ad, str):
        is_ad = is_ad.lower() == "true"
    return {"is_ad": bool(is_ad)}


def _extract_content_category(data: Optional[dict]) -> str:
    """从 GetLabelClassificationService 返回中提取内容分类（一/二/三级拼接）。"""
    if not data:
        return ""

    taxonomies = data.get("noteTaxonomies", [])
    if isinstance(taxonomies, str):
        try:
            taxonomies = json.loads(taxonomies)
        except (ValueError, TypeError):
            return ""

    if not isinstance(taxonomies, list) or not taxonomies:
        return ""

    t = taxonomies[0]
    if not isinstance(t, dict):
        return ""

    parts = []
    for key in ("taxonomy1Name", "taxonomy2Name", "taxonomy3Name"):
        v = t.get(key, "")
        if v:
            parts.append(v)

    return "/".join(parts)


def _extract_comments(data: Optional[dict], author_user_id: str = "") -> List[dict]:
    """从 getNoteCommentApp 返回中提取作者评论和置顶评论（基础特征用）。"""
    if not data:
        return []

    comment_raw = data.get("comment", "[]")
    if isinstance(comment_raw, str):
        try:
            comment_raw = json.loads(comment_raw)
        except (ValueError, TypeError):
            return []

    if not isinstance(comment_raw, list):
        return []

    results = []
    for c in comment_raw:
        if not isinstance(c, dict):
            continue
        user_id = c.get("userId", "")
        is_author = (user_id == author_user_id) if author_user_id else False
        is_top = bool(c.get("topStatus", False))

        # 基础特征只保留作者评论和置顶评论
        if not is_author and not is_top:
            continue

        results.append({
            "content": c.get("content", ""),
            "user_name": c.get("userName", ""),
            "is_author": is_author,
            "is_top": is_top,
            "like_count": c.get("likeCount", 0),
        })

    return results


def _extract_user_all(data: Optional[dict]) -> dict:
    """从 GetUserInfoAll 返回中提取用户画像（审核相关字段）。"""
    if not data:
        return {}

    def _to_int(v):
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            try:
                return int(v)
            except (ValueError, TypeError):
                return 0
        return 0

    # userLabelFields 解析
    labels = data.get("userLabelFields", [])
    if isinstance(labels, str):
        try:
            labels = json.loads(labels)
        except (ValueError, TypeError):
            labels = []

    # userDisplayLabel 解析
    display_label = data.get("userDisplayLabel", [])
    if isinstance(display_label, str):
        try:
            display_label = json.loads(display_label)
        except (ValueError, TypeError):
            display_label = []

    # userAttributes 解析
    attributes = data.get("userAttributes", [])
    if isinstance(attributes, str):
        try:
            attributes = json.loads(attributes)
        except (ValueError, TypeError):
            attributes = []

    # businessUserIdentity 解析
    business_identity = data.get("businessUserIdentity", {})
    if isinstance(business_identity, str):
        try:
            business_identity = json.loads(business_identity)
        except (ValueError, TypeError):
            business_identity = {}

    # medicalKosQuality 解析
    medical_kos = data.get("medicalKosQuality", [])
    if isinstance(medical_kos, str):
        try:
            medical_kos = json.loads(medical_kos)
        except (ValueError, TypeError):
            medical_kos = []

    return {
        "user_id": data.get("userId", ""),
        "nickname": data.get("userName", ""),
        "avatar": data.get("avatar", ""),
        "banner_image": data.get("lastBannerImageUrl", data.get("bannerImageUrl", "")),
        "desc": data.get("desc", ""),
        "gender": data.get("gender", ""),
        "ip_location": data.get("location", ""),
        "fans_count": _to_int(data.get("fansCount", 0)),
        "following_count": _to_int(data.get("followCount", 0)),
        "note_count": _to_int(data.get("noteCount", 0)),
        "like_count": _to_int(data.get("likedCount", 0)),
        "collect_count": _to_int(data.get("collectedCount", 0)),
        "user_level": data.get("userLevel", ""),
        "level_reason": data.get("levelReason", ""),
        "punish_level": data.get("userPunishLevelCn", ""),
        "group": data.get("group", ""),
        "register_time": data.get("createTime", ""),
        "is_brand_account": str(data.get("isBrandAccount", "")).lower() == "true",
        "is_brand_official": str(data.get("isBrandOfficial", "")).lower() == "true",
        "is_ads_account": str(data.get("isAdsAccount", "")).lower() == "true",
        "is_official_auth": str(data.get("isOfficialAuth", "")).lower() == "true",
        "is_brand_cooperator": str(data.get("isBrandCooperator", "")).lower() == "true",
        "red_official_verify_type": data.get("redOfficialVerifyType", "0"),
        "user_copied_account_flag": data.get("userCopiedAccountFlag", ""),
        "business_user_identity": business_identity if isinstance(business_identity, dict) else {},
        "medical_kos_quality": medical_kos if isinstance(medical_kos, list) else [],
        "user_attributes": attributes if isinstance(attributes, list) else [],
        "display_label": display_label if isinstance(display_label, list) else [],
        "model_age": data.get("modelAge", ""),
        "manual_age": data.get("manualAge", ""),
    }


def _extract_big_v(data: Optional[dict]) -> dict:
    """从 JudgeBigVByNoteId 返回中提取大V判断。"""
    if not data:
        return {"is_big_v": False, "dimensions": []}
    is_big_v = data.get("isBigV", data.get("bigV", False))
    if isinstance(is_big_v, str):
        is_big_v = is_big_v.lower() == "true"
    dimensions = data.get("bigVDimensions", data.get("dimensions", []))
    if isinstance(dimensions, str):
        try:
            dimensions = json.loads(dimensions)
        except (ValueError, TypeError):
            dimensions = []
    return {
        "is_big_v": bool(is_big_v),
        "dimensions": dimensions if isinstance(dimensions, list) else [],
    }


def _extract_note_images(resource_data: Optional[dict]) -> List[dict]:
    """从 QueryNoteResourceHistoryService 返回中提取图文笔记的图片列表。

    图文笔记: resourceList 有 N 条，每条 resourceType=image，有 url + OCR。
    视频笔记: resourceType=video，返回空列表（由 _extract_key_frames 处理）。

    返回: [{url, ocr_text, width, height}, ...]
    """
    if not resource_data:
        return []

    resource_list = resource_data.get("resourceList", [])
    if isinstance(resource_list, str):
        try:
            resource_list = json.loads(resource_list)
        except (ValueError, TypeError):
            return []

    if not isinstance(resource_list, list) or not resource_list:
        return []

    # 检查是否是图文（第一条 resourceType == image）
    r0 = resource_list[0]
    if isinstance(r0, str):
        try:
            r0 = json.loads(r0)
        except (ValueError, TypeError):
            return []
    if not isinstance(r0, dict) or r0.get("resourceType") not in ("image", "livephoto"):
        return []

    results = []
    for r in resource_list:
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except (ValueError, TypeError):
                continue
        if not isinstance(r, dict):
            continue

        # OCR
        ocr_info = r.get("coverImageOcrInfo", {})
        if isinstance(ocr_info, str):
            try:
                ocr_info = json.loads(ocr_info)
            except (ValueError, TypeError):
                ocr_info = {}
        ocr_text = ocr_info.get("textOcrCombine", "") if isinstance(ocr_info, dict) else ""

        url = r.get("url", r.get("originCoverUrl", ""))
        # Live Photo / 视频类型：url 是 mp4，用 cover 字段取静态封面图
        if url and ('.mp4' in url or 'video' in url):
            url = r.get("cover", "")
        if not url or '.mp4' in url or 'video' in url:
            continue
        results.append({
            "url": url,
            "ocr_text": ocr_text,
        })

    return results


def _get_frame_index_from_url(url: str) -> Optional[int]:
    """从关键帧 URL 中提取帧序号（URL 末尾 _数字.jpg）。"""
    import re
    match = re.search(r'_(\d+)\.jpg', url)
    if match:
        return int(match.group(1))
    return None


def _fix_frame_timestamps(frames: List[dict]) -> List[dict]:
    """修正异常时间戳：后面的帧 timestamp 回退到 0 时，直接丢弃这些异常帧。"""
    if not frames or len(frames) < 2:
        return frames

    # 找到最后一个正常递增的时间戳位置
    max_ts = 0.0
    last_normal_idx = 0
    for i, f in enumerate(frames):
        ts = f.get("timestamp", 0)
        if ts >= max_ts:
            max_ts = ts
            last_normal_idx = i
        else:
            break

    # 如果没有异常（全部递增），直接返回
    if last_normal_idx == len(frames) - 1:
        return frames

    # 有异常帧：直接截断，只保留正常递增的部分
    return frames[: last_normal_idx + 1]


def _extract_key_frames(resource_data: Optional[dict], note_id: str = "", duration: int = 0) -> dict:
    """提取视频关键帧信息。

    帧图片从 GetNoteVideoDenseFrameData 因子接口获取（1s1帧，在线）。
    封面和 OCR 全文从 QueryNoteResourceHistoryService 获取。

    返回: {
        "cover": {url, ocr_text, found_right_thumb, have_text2image},
        "frames": [{timestamp, url}, ...],  # 1s1帧
        "total_frames": int,
        "ocr_combine": str,  # 视频OCR全文
    }
    """
    if not resource_data:
        return {}

    resource_list = resource_data.get("resourceList", [])
    if isinstance(resource_list, str):
        try:
            resource_list = json.loads(resource_list)
        except (ValueError, TypeError):
            return {}

    if not isinstance(resource_list, list) or not resource_list:
        return {}

    r0 = resource_list[0]
    if isinstance(r0, str):
        try:
            r0 = json.loads(r0)
        except (ValueError, TypeError):
            return {}

    if not isinstance(r0, dict) or r0.get("resourceType") != "video":
        return {}

    # 封面信息（从 QueryNoteResourceHistoryService）
    cover_ocr_info = r0.get("coverImageOcrInfo", {})
    if isinstance(cover_ocr_info, str):
        try:
            cover_ocr_info = json.loads(cover_ocr_info)
        except (ValueError, TypeError):
            cover_ocr_info = {}
    cover_ocr = cover_ocr_info.get("textOcrCombine", "") if isinstance(cover_ocr_info, dict) else ""

    cover = {
        "url": r0.get("cover", r0.get("originCoverUrl", "")),
        "ocr_text": cover_ocr,
        "found_right_thumb": bool(r0.get("foundRightThumbType", False)),
        "have_text2image": bool(r0.get("haveText2Image", False)),
    }

    # OCR 全文（从 QueryNoteResourceHistoryService）
    ocr_combine = r0.get("ocrCombine", "")
    if isinstance(ocr_combine, str) and ocr_combine.startswith("["):
        try:
            ocr_combine = json.loads(ocr_combine)
            if isinstance(ocr_combine, list):
                ocr_combine = " ".join(str(x) for x in ocr_combine)
        except (ValueError, TypeError):
            pass

    # 帧图片从 HuiShenVideoDenseFrame 因子接口获取（最大300帧）
    frames = []
    if note_id:
        frames = _fetch_dense_frames(note_id, duration=duration)

    return {
        "cover": cover,
        "frames": frames,
        "total_frames": len(frames),
        "ocr_combine": ocr_combine if isinstance(ocr_combine, str) else "",
    }


def _fetch_dense_frames(note_id: str, duration: int = 0) -> List[dict]:
    """从 HuiShenVideoDenseFrame 因子接口获取帧列表。

    最大 300 帧，已采样。timestamp 根据 index 和视频时长计算。
    """
    import sys
    sys.path.insert(0, "/path/to/local-assets")
    try:
        from query_factor import query_factor, extract_result
        resp = query_factor("HuiShenVideoDenseFrame", {"noteId": note_id})
        result = extract_result(resp)
        if isinstance(result, str):
            result = json.loads(result)
        if not isinstance(result, list) or not result:
            return []

        total_frames = len(result)
        # 计算每帧对应的秒数
        if duration and total_frames > 0:
            seconds_per_frame = duration / total_frames
        else:
            seconds_per_frame = 1.0

        frames = []
        for i, url in enumerate(result):
            if isinstance(url, str) and url.startswith("http"):
                ts = round(i * seconds_per_frame)
                frames.append({"timestamp": ts, "url": url})
        return frames
    except Exception:
        return []


def _extract_similar_tip(tip_data: Optional[dict]) -> dict:
    """从 SimilarNoteTipService 返回中提取相似笔记提示。（已废弃，保留兼容）"""
    if not tip_data:
        return {}
    tip_info = tip_data.get("tipInfo", tip_data)
    if not isinstance(tip_info, dict):
        return {}
    return {
        "contains_star": tip_info.get("noteInfoContainStar", False),
        "is_commercial": tip_info.get("commercialNote", False),
        "contains_user": tip_info.get("noteInfoContainUser", False),
        "star_names": tip_info.get("noteInfoContainStarNames", ""),
    }


def fetch_note_content(backend: DataBackend, note_id: str) -> Dict[str, Any]:
    """拉取笔记内容板块全部数据，返回结构化 dict。

    第一批（只需 noteId，并行）:
    - queryNoteInfo, QueryNoteResourceV2Service, QueryNoteAsrInfo,
      MultiGetNoteTypeByNoteId, GetNoteAtStart, GetNotePkVoteInfoService,
      SimilarNoteTipService

    第二批（依赖 queryNoteInfo 返回的 historyId）:
    - QueryNoteResourceService: 图片/视频资源+OCR坐标+关键帧抽帧
    - QueryNoteResourceHistoryService: 历史版本资源
    - GetNoteHistoryDetail: 历史版本详情（修改记录）

    返回的 dict 包含以下顶层 key:
    - note_id, title, content, type, level, ...（笔记基础）
    - author: {user_id, nickname, ...}（作者摘要）
    - counts: {like, collect, comment, ...}（互动数）
    - tags: [{name, type}, ...]（标签列表）
    - images: [{url, file_id, ...}, ...]（图片列表）
    - video: {video_url, duration, ...} | None
    - resources: [{file_id, url, type, ocr_text}, ...]（资源+OCR，来自V2）
    - asr_text: str（ASR 全文）
    - note_types: [str, ...]（笔记类型标签）
    - at_users: [str, ...]（@提及）
    - similar_tip: {contains_star, is_commercial, ...}
    - note_resource: dict | None（第二批：QueryNoteResourceService 全量原始返回）
    - note_resource_history: dict | None（第二批：QueryNoteResourceHistoryService 全量原始返回）
    - note_history_detail: dict | None（第二批：GetNoteHistoryDetail 全量原始返回）
    """
    # === 第一批：只需 noteId ===
    note_info = backend.call("queryNoteInfo", noteId=note_id)
    asr = backend.call("QueryNoteAsrInfo", noteId=note_id)
    note_type = backend.call("MultiGetNoteTypeByNoteId",
        noteIds=json.dumps([note_id]),
        needShowNoteTypes=json.dumps(
            ["购物笔记", "商品笔记", "晒单笔记", "直播预告笔记", "交易笔记",
             "报备笔记", "好物体验笔记", "本地商品笔记", "本地购物笔记", "抽奖笔记",
             "薯条加热版投放中", "薯条竞价版投放中", "薯条竞价版", "效果广告",
             "品牌合作", "素材笔记"],
            ensure_ascii=False,
        ))
    at_start = backend.call("GetNoteAtStart", noteId=note_id)
    pk_vote = backend.call("GetNotePkVoteInfoService", noteId=note_id)
    note_info_new = backend.call("GetNoteInfoNewService", noteId=note_id)
    note_ads_info = backend.call("NoteAdsInfoService", noteId=note_id)
    high_risk_ip = backend.call("queryNoteHighRiskIp", noteId=note_id)
    ad_result = backend.call("GetNoteAdResult", noteId=note_id)
    comments_raw = backend.call("getNoteCommentApp", noteId=note_id, pageable="{}")
    if not comments_raw:
        from ._base import OnlineBackend as _OB
        comments_raw = _OB().call("getNoteCommentApp", noteId=note_id, pageable="{}")
    label_class = backend.call("GetLabelClassificationService", noteIds=[note_id])

    result = _extract_note_detail(note_info)
    result["asr_segments"] = _extract_asr(asr)
    result["note_types"] = _extract_note_type(note_type, note_id)
    result["at_users"] = _extract_at_users(at_start)
    result["pk_vote"] = pk_vote if pk_vote else {}
    result["commercial_info"] = _extract_commercial_info(note_info_new, note_ads_info)
    result["high_risk_ip"] = _extract_high_risk_ip(high_risk_ip)
    result["ad_result"] = _extract_ad_result(ad_result)
    result["comments"] = _extract_comments(comments_raw, result.get("user_id", ""))
    result["content_category"] = _extract_content_category(label_class)

    # === 第二批：依赖 userId / historyId ===
    user_id = result.get("user_id", "")
    history_id = result.get("history_id", "")

    # 用户全量信息
    user_all = backend.call(
        "GetUserInfoAll",
        uid=user_id,
        needLevel=True,
        needAttributes=True,
        needUserTag=True,
        needNegativeInfo=True,
        needBusinessInfo=True,
        needPortrait=True,
        needPanelInfo=True,
        needIdentityInfo=True,
        needBannerImage=True,
    )
    result["user_all"] = _extract_user_all(user_all)

    # 大V判断
    big_v = backend.call("JudgeBigVByNoteId", noteId=note_id, userId=user_id)
    result["is_big_v"] = _extract_big_v(big_v)

    # 资源数据统一走在线 QueryNoteResourceHistoryService（避免离线截断问题）
    from ._base import OnlineBackend
    online = OnlineBackend()
    note_resource = online.call(
        "QueryNoteResourceHistoryService", noteId=note_id, historyId=history_id
    )

    # 第二批暂存全量原始返回
    result["note_resource"] = note_resource

    # 提取关键帧（视频）或图片列表（图文）
    result["key_frames"] = _extract_key_frames(note_resource, note_id=note_id, duration=result.get("video_duration") or 0)
    result["note_images"] = _extract_note_images(note_resource)

    # 提取 labelMap（原创声明等标记）
    label_map = {}
    if isinstance(note_resource, dict):
        lm = note_resource.get("labelMap", {})
        if isinstance(lm, str):
            try:
                label_map = json.loads(lm)
            except (ValueError, TypeError):
                label_map = {}
        elif isinstance(lm, dict):
            label_map = lm
    result["original_statement"] = label_map.get("original_statement", "") == "true"

    return result


# ===========================================================================
# 组装函数：将 raw dict 组装成最终结构化输出
# ===========================================================================


def render_note_content(raw: Dict[str, Any]) -> Dict[str, Any]:
    """将 fetch_note_content 返回的 raw dict 组装成结构化 dict。

    只保留审核需要的字段，去掉冗余。
    修改组装逻辑只改这个函数即可。
    """
    # --- queryNoteInfo 部分 ---
    note_info = {
        "标题": raw.get("title", ""),
        "正文": raw.get("content", ""),
        "笔记类型": raw.get("type", ""),
        "视频时长": raw.get("video_duration"),
        "发布时间": _format_timestamp(raw.get("create_time")),
        "发布地区": raw.get("birth_place", ""),
    }

    author_info = {
        "昵称": raw.get("author", {}).get("nickname", ""),
        "头像": raw.get("author", {}).get("avatar", ""),
        "简介": raw.get("author", {}).get("desc", ""),
        "粉丝数": raw.get("author", {}).get("fans_count", 0),
        "性别": raw.get("author", {}).get("gender", ""),
    }

    counts = raw.get("counts", {})
    互动数据 = {
        "点赞数": counts.get("like", 0),
        "收藏数": counts.get("collect", 0),
        "评论数": counts.get("comment", 0),
        "阅读数": counts.get("read", 0),
        "分享数": counts.get("share", 0),
        "视频播放数": counts.get("video_play", 0),
        "精选阅读数": counts.get("beautiful_read", 0),
    }

    # 标签只保留 location 类型（topic 已在正文 #话题# 里）
    tags = raw.get("tags", [])
    定位 = [t["name"] for t in tags if t.get("type") == "location"]

    # --- 商业属性（GetNoteInfoNewService + NoteAdsInfoService 合并）---
    commercial = raw.get("commercial_info", {})
    商业属性 = {
        "群组封禁": commercial.get("is_group_banned", False),
        "广告属性": commercial.get("ads_attributes", []),
        "售卖属性": commercial.get("sale_attributes", []),
        "薯条投放状态": commercial.get("chip_note_status", []),
        "广告投放": commercial.get("is_ad_note", False),
        "订单绑定": commercial.get("is_order_bind", False),
        "薯店笔记": commercial.get("is_shu_shop", False),
        "品牌合作": commercial.get("is_brand_draw", False),
        "商品合作": commercial.get("is_goods_cooperate", False),
        "体验笔记": commercial.get("is_experience", False),
    }

    # --- MultiGetNoteTypeByNoteId 部分 ---
    note_types = raw.get("note_types", [])

    # --- ASR 部分 ---
    asr_segments = raw.get("asr_segments", [])

    # --- @提及 ---
    at_users = raw.get("at_users", [])

    # --- 内容分类 ---
    content_category = raw.get("content_category", "")

    # --- 高风险IP ---
    high_risk_ip = raw.get("high_risk_ip", {})

    # --- 广告检测 ---
    ad_result = raw.get("ad_result", {})

    return {
        "笔记信息": note_info,
        "作者信息": author_info,
        "互动数据": 互动数据,
        "定位": 定位 if 定位 else None,
        "笔记商业属性": 商业属性,
        "笔记子类型": note_types if note_types else None,
        "ASR语音转文字": asr_segments if asr_segments else None,
        "@提及用户": at_users if at_users else None,
        "内容分类": content_category if content_category else None,
        "高风险IP": high_risk_ip,
        "广告检测": ad_result,
        "PK投票": raw.get("pk_vote") or {},
        "视频关键帧": raw.get("key_frames") or {},
        "用户画像": raw.get("user_all") or {},
        "is_big_v": raw.get("is_big_v") or {},
        "_user_id": raw.get("user_id", ""),
        "_history_id": raw.get("history_id", ""),
    }


def _format_timestamp(ts) -> str:
    """毫秒时间戳转 YYYY-MM-DD HH:MM:SS 格式。"""
    if not ts:
        return ""
    try:
        from datetime import datetime
        if isinstance(ts, str):
            ts = int(ts)
        return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, OSError):
        return str(ts)


def render_note_content_text(raw: Dict[str, Any], show_images: bool = True) -> str:
    """将 fetch_note_content 返回的 raw dict 组装成 Markdown 格式字符串。"""
    d = render_note_content(raw)

    lines = []

    # ============ 笔记信息 ============
    lines.append("## 笔记信息")
    lines.append("")

    # 标题
    title = d["笔记信息"]["标题"]
    if title:
        lines.append(f"### 标题")
        lines.append(title)
        lines.append("")

    # 正文
    content = d["笔记信息"]["正文"]
    if content:
        lines.append(f"### 正文")
        lines.append(content)
        lines.append("")

    # 发布时间 & 地区
    create_time = d["笔记信息"]["发布时间"]
    birth_place = d["笔记信息"]["发布地区"]
    note_type = d["笔记信息"]["笔记类型"]
    video_duration = d["笔记信息"].get("视频时长")
    lines.append("### 发布信息")
    if note_type:
        lines.append(f"- 笔记类型: {note_type}")
    if video_duration:
        lines.append(f"- 视频时长: {video_duration}s")
    if create_time:
        lines.append(f"- 发布时间: {create_time}")
    if birth_place:
        lines.append(f"- 发布地区: {birth_place}")
    lines.append("")

    # 定位
    定位 = d.get("定位")
    if 定位:
        lines.append(f"### 定位")
        lines.append(f"- {', '.join(定位)}")
        lines.append("")

    # 内容分类
    content_category = d.get("内容分类")
    if content_category:
        lines.append(f"### 内容分类")
        lines.append(f"- {content_category}")
        lines.append("")

    # 互动数据
    互动 = d["互动数据"]
    lines.append("### 互动数据")
    lines.append(f"- 点赞: {互动['点赞数']} | 收藏: {互动['收藏数']} | 评论: {互动['评论数']}")
    lines.append(f"- 阅读: {互动['阅读数']} | 分享: {互动['分享数']} | 视频播放: {互动['视频播放数']}")
    lines.append(f"- 精选阅读: {互动['精选阅读数']}")
    lines.append("")

    # 商业属性/投放状态
    商业 = d.get("笔记商业属性", {})
    commercial_flags = []
    if 商业.get("广告投放"):
        commercial_flags.append("广告投放")
    if 商业.get("薯店笔记"):
        commercial_flags.append("薯店笔记")
    if 商业.get("品牌合作"):
        commercial_flags.append("品牌合作")
    if 商业.get("商品合作"):
        commercial_flags.append("商品合作")
    if 商业.get("体验笔记"):
        commercial_flags.append("体验笔记")
    if 商业.get("订单绑定"):
        commercial_flags.append("订单绑定")

    has_commercial = (
        商业.get("群组封禁")
        or 商业.get("广告属性")
        or 商业.get("售卖属性")
        or 商业.get("薯条投放状态")
        or commercial_flags
    )
    if has_commercial:
        lines.append("### 商业属性/投放状态")
        if 商业.get("群组封禁"):
            lines.append("- 群组封禁: 是")
        if 商业.get("广告属性"):
            lines.append(f"- 广告属性: {', '.join(商业['广告属性'])}")
        if 商业.get("售卖属性"):
            lines.append(f"- 售卖属性: {', '.join(商业['售卖属性'])}")
        if 商业.get("薯条投放状态"):
            lines.append(f"- 薯条投放: {', '.join(商业['薯条投放状态'])}")
        if commercial_flags:
            lines.append(f"- 商业标识: {', '.join(commercial_flags)}")
        lines.append("")

    # 笔记子类型
    note_types = d.get("笔记子类型")
    if note_types:
        lines.append("### 子类型标签")
        lines.append(f"- {', '.join(note_types)}")
        lines.append("")

    # ASR 语音转文字
    asr = d.get("ASR语音转文字")
    if asr:
        lines.append("### ASR语音转文字")
        for seg in asr:
            lines.append(f"- [{seg['start']}s-{seg['end']}s] {seg['text']}")
        lines.append("")

    # 视频关键帧
    kf = d.get("视频关键帧", {})
    if kf and kf.get("frames") and show_images:
        lines.append("### 视频封面与关键帧")

        cover = kf.get("cover", {})
        if cover.get("url"):
            lines.append(f"- 封面: <image>")
            if cover.get("ocr_text"):
                lines.append(f"  封面OCR: {cover['ocr_text']}")
            if cover.get("have_text2image"):
                lines.append(f"  文字转图片: 是")
            if not cover.get("found_right_thumb"):
                lines.append(f"  封面与内容不匹配")
        lines.append("")

        # 展示时采样到最多16帧
        all_frames = kf["frames"]
        max_display = 16
        if len(all_frames) > max_display:
            step = len(all_frames) / max_display
            display_frames = [all_frames[int(i * step)] for i in range(max_display)]
        else:
            display_frames = all_frames

        for frame in display_frames:
            ts = frame.get("timestamp", 0)
            lines.append(f"- [{ts}s] | <image>")

        ocr_combine = kf.get("ocr_combine", "")
        if ocr_combine and ocr_combine != "null":
            lines.append("")
            lines.append(f"- 视频OCR全文: {ocr_combine}")
        lines.append("")

    # 图文笔记图片
    note_images = raw.get("note_images", [])
    if note_images and show_images:
        lines.append("### 笔记图片")
        for i, img in enumerate(note_images):
            ocr = img.get("ocr_text", "")
            if ocr:
                lines.append(f"- <image> | OCR: {ocr}")
            else:
                lines.append(f"- <image>")
        lines.append("")

    # 评论
    comments = raw.get("comments", [])
    if comments:
        lines.append("### 评论区")
        for c in comments:
            marks = []
            if c.get("is_top"):
                marks.append("置顶")
            if c.get("is_author"):
                marks.append("作者")
            mark_str = f" [{','.join(marks)}]" if marks else ""
            likes = c.get("like_count", 0)
            like_str = f" (赞{likes})" if likes else ""
            lines.append(f"- {c.get('user_name', '')}{mark_str}: {c.get('content', '')}{like_str}")
        lines.append("")

    # ============ 作者信息 ============
    lines.append("## 作者信息")
    lines.append("")

    user = d.get("用户画像", {})
    if user:
        lines.append(f"- 昵称: {user.get('nickname', '')}")
        # 作者头像/背景图对审核判定帮助有限，不作为 multimodal 输入返回。
        lines.append(f"- 简介: {user.get('desc', '')}")
        lines.append(f"- 性别: {user.get('gender', '')}")
        lines.append(f"- 预测年龄: {user.get('model_age', '')} | 手填年龄: {user.get('manual_age', '')}")
        lines.append(f"- IP属地: {user.get('ip_location', '')}")
        lines.append(f"- 粉丝: {user.get('fans_count', 0)} | 关注: {user.get('following_count', 0)} | 笔记数: {user.get('note_count', 0)}")
        lines.append(f"- 获赞: {user.get('like_count', 0)} | 收藏: {user.get('collect_count', 0)}")
        lines.append(f"- 注册时间: {_format_timestamp(int(user.get('register_time', 0)) * 1000) if user.get('register_time') else ''}")
        user_level = user.get('user_level', '')
        level_reason = user.get('level_reason', '')
        if level_reason:
            lines.append(f"- 用户分级: {user_level} ({level_reason})")
        else:
            lines.append(f"- 用户分级: {user_level}")
        lines.append(f"- 用户组: {user.get('group', '')}")
        lines.append(f"- 惩罚状态: {user.get('punish_level', '无')}")
        # 官方认证类型
        verify_type = user.get('red_official_verify_type', '0')
        if str(verify_type) != '0' and verify_type:
            verify_map = {'1': '个人红V', '2': '企业蓝V'}
            lines.append(f"- 官方认证: {verify_map.get(str(verify_type), f'类型{verify_type}')}")
        # 用户身份标记
        copied_flag = user.get('user_copied_account_flag', '')
        if copied_flag:
            lines.append(f"- 账号身份标记: {copied_flag}")
        # 用户属性
        attrs = user.get('user_attributes', [])
        if attrs:
            lines.append(f"- 用户属性: {', '.join(attrs)}")
        # 生态标签
        display = user.get('display_label', [])
        if display:
            label_strs = [item.get('content', '') for item in display if isinstance(item, dict)]
            if label_strs:
                lines.append(f"- 生态标签: {', '.join(label_strs)}")
        # 品牌/广告/认证标识
        flags = []
        if user.get('is_brand_account'):
            flags.append("品牌账号")
        if user.get('is_brand_official'):
            flags.append("品牌官方")
        if user.get('is_ads_account'):
            flags.append("广告账号")
        if user.get('is_official_auth'):
            flags.append("官方认证")
        if user.get('is_brand_cooperator'):
            flags.append("品牌合作")
        if flags:
            lines.append(f"- 特殊标识: {', '.join(flags)}")
        else:
            lines.append(f"- 特殊标识: 无")
        # 商业用户身份
        biz = user.get('business_user_identity', {})
        if biz:
            score = biz.get('business_user_score_v1', '')
            if score:
                lines.append(f"- 商业用户评分: {score}")
            rg_cnt = biz.get('note_marketing_integrated_level_rg_cnt', '0')
            if str(rg_cnt) != '0':
                lines.append(f"- 营销笔记数(软广): {rg_cnt}")
        # 医疗资质
        medical = user.get('medical_kos_quality', [])
        if medical:
            lines.append(f"- 医疗资质: {', '.join(medical)}")
        # 大V判断
        big_v = d.get("is_big_v", raw.get("is_big_v", {}))
        if isinstance(big_v, dict) and big_v.get("is_big_v"):
            dims = big_v.get("dimensions", [])
            dims_str = f" ({', '.join(dims)})" if dims else ""
            lines.append(f"- 大V: 是{dims_str}")
    else:
        author = d["作者信息"]
        lines.append(f"- 昵称: {author['昵称']}")
        lines.append(f"- 简介: {author['简介']}")
        lines.append(f"- 粉丝数: {author['粉丝数']}")
        lines.append(f"- 性别: {author['性别']}")
    lines.append("")

    # ============ 风险信号 ============
    has_risk = False
    risk_lines = []

    # @提及用户
    at_users = d.get("@提及用户")
    if at_users:
        has_risk = True
        risk_lines.append("### @提及用户")
        for u in at_users:
            star_mark = " ⭐明星" if u.get("is_star") else ""
            risk_lines.append(f"- {u['nickname']}{star_mark}")
        risk_lines.append("")

    # 高风险IP
    ip_info = d.get("高风险IP")
    if ip_info and ip_info.get("is_high_risk"):
        has_risk = True
        risk_lines.append("### 高风险IP")
        risk_lines.append("- 是")
        risk_lines.append("")

    # 广告检测
    ad = d.get("广告检测")
    if ad and ad.get("is_ad"):
        has_risk = True
        risk_lines.append("### 广告检测")
        risk_lines.append("- 判定为广告")
        risk_lines.append("")

    # PK投票
    pk = d.get("PK投票")
    if pk:
        has_risk = True
        risk_lines.append("### PK投票")
        risk_lines.append(f"- {json.dumps(pk, ensure_ascii=False)}")
        risk_lines.append("")

    if has_risk:
        lines.append("## 风险信号")
        lines.append("")
        lines.extend(risk_lines)

    return "\n".join(lines)


def get_note_detail(
    backend: DataBackend,
    note_id: str,
    include_original_statement: bool = False,
    show_images: bool = True,
    max_display_frames: int = 16,
) -> Dict[str, Any]:
    """获取笔记完整信息，返回统一格式的字典。

    Args:
        backend: 数据后端
        note_id: 笔记ID
        include_original_statement: 是否在 text 里展示原创声明状态（搬运检测时用）
        show_images: 是否在 text 里展示图片占位符（False 时不展示笔记图片/关键帧，但保留头像背景图）
        max_display_frames: 视频关键帧最多展示数量，不含封面。默认 16。

    返回:
    {
        "text": str,              # render 后的 Markdown 文本（帧采样到16帧）
        "display_images": [str],  # text 中 <image> 对应的 URL 列表（按出现顺序）
        "all_images": [str],      # 全量关键帧 URL 列表
    }
    """
    raw = fetch_note_content(backend, note_id)
    text = render_note_content_text(raw, show_images=show_images)

    # 搬运检测时额外展示原创声明状态
    if include_original_statement:
        is_original = raw.get("original_statement", False)
        statement_line = f"\n### 原创声明\n- {'有原创声明' if is_original else '无原创声明'}\n"
        text = statement_line + text

    # 收集 images（视频和图文互斥）
    kf = raw.get("key_frames", {})
    note_images = raw.get("note_images", [])
    user_all = raw.get("user_all", {})

    all_images = []
    display_images = []

    if kf and kf.get("frames") and show_images:
        # 视频笔记：封面 + 关键帧
        cover = kf.get("cover", {})
        if cover.get("url"):
            display_images.append(cover["url"])

        all_frames = kf.get("frames", [])
        max_display = max(0, int(max_display_frames))
        if len(all_frames) > max_display:
            step = len(all_frames) / max_display if max_display else 0
            display_frames = [all_frames[int(i * step)] for i in range(max_display)]
        else:
            display_frames = all_frames

        for frame in display_frames:
            if frame.get("url"):
                display_images.append(frame["url"])

        # all_images = 全量关键帧
        for frame in all_frames:
            if frame.get("url"):
                all_images.append(frame["url"])

    elif note_images and show_images:
        # 图文笔记：所有图片
        for img in note_images:
            if img.get("url"):
                display_images.append(img["url"])
                all_images.append(img["url"])

    # 作者头像/背景图不进入 display_images，避免与审核无关的视觉噪声。

    return {
        "text": text,
        "display_images": display_images,
        "all_images": all_images,
    }
