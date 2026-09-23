"""在线规则查询服务：调 Mars Jupiter API 获取队列完整规则信息。

调用 3 个接口：
1. assembleDetailedRuleJson → 队列关联的所有 tag 规则
2. sopNodeDetailRelation → SOP 流程（可能为空）
3. queryAuditNotice → 审核须知（优先级、全局豁免、判断逻辑等）

返回结构化 dict，HTML 已清洗为纯文本，冗余字段已去除。
"""

from __future__ import annotations

import json
import os
import logging
import re
import html as html_lib
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_JUPITER_BASE = os.environ.get("AUDIT_RULE_SERVICE_URL", "").rstrip("/")
_HEADERS = {"Content-Type": "application/json"}
_TIMEOUT = 30


# ─────────────────────────────────────────────────────────────────────────────
# HTML 清洗
# ─────────────────────────────────────────────────────────────────────────────


def _html_to_text(html_str: str) -> str:
    """HTML 富文本 → 纯文本（保留基本换行结构）。"""
    if not html_str:
        return ""
    text = html_str
    text = re.sub(r"<li[^>]*>", "- ", text, flags=re.IGNORECASE)
    text = re.sub(r"</li>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</h[1-6]>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<h[1-6][^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<img[^>]*>", "<image>", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_images_from_html(html_str: str) -> List[str]:
    """从 HTML 中提取所有 <img src="..."> 的 URL。"""
    if not html_str:
        return []
    matches = re.findall(
        r"<img\b[^>]*\bsrc\s*=\s*(['\"])(.*?)\1",
        html_str,
        flags=re.IGNORECASE,
    )
    return [html_lib.unescape(url) for _, url in matches if url]


def _html_to_text_and_images(html_str: str) -> tuple[str, List[str]]:
    """HTML 富文本 → 文本 + 与 <image> 占位符对齐的图片 URL。"""
    return _html_to_text(html_str), _extract_images_from_html(html_str)


def _set_html_section(result: dict, key: str, raw_html: str) -> None:
    text, images = _html_to_text_and_images(raw_html)
    result[key] = text
    if images:
        result[f"{key}图片"] = images


def _with_missing_image_tags(text: str, images: Any) -> str:
    """补足 section 文本中缺失的 <image>，确保图片 URL 不会错位。"""
    text = str(text or "")
    image_count = len(images) if isinstance(images, list) else 0
    missing = image_count - text.count("<image>")
    if missing <= 0:
        return text
    suffix = "\n".join("<image>" for _ in range(missing))
    return f"{text}\n{suffix}".strip() if text else suffix


# ─────────────────────────────────────────────────────────────────────────────
# 接口调用
# ─────────────────────────────────────────────────────────────────────────────


def _post_jupiter(endpoint: str, params: dict) -> dict:
    """调用 Jupiter API，返回 response JSON。"""
    from ._base import online_services_disabled, warn_online_disabled_once

    if online_services_disabled():
        warn_online_disabled_once(f"jupiter/{endpoint}")
        return {"success": False, "data": {}}
    if not _JUPITER_BASE:
        raise RuntimeError("Configure AUDIT_RULE_SERVICE_URL or use local rule caches")
    url = f"{_JUPITER_BASE}/{endpoint}"
    try:
        resp = requests.post(url, headers=_HEADERS, json=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            logger.warning(f"Jupiter {endpoint} returned success=false: {data.get('msg')}")
        return data
    except Exception as e:
        logger.error(f"Jupiter {endpoint} failed: {e}")
        return {"success": False, "data": {}}


def _fetch_rules_raw(source_type: str, category_type: str) -> dict:
    """接口1: 查询队列关联的所有规则内容。"""
    return _post_jupiter("assembleDetailedRuleJson", {
        "params": {
            "request": json.dumps({
                "sourceTypeInfo": {
                    "categoryType": category_type,
                    "sourceType": source_type,
                }
            })
        }
    })


def _fetch_rules_by_tag_ids_raw(
    tag_ids: List[str],
    source_type: Optional[str] = None,
    category_type: str = "note",
) -> dict:
    """接口1补充用法: 按 tagId 查询规则详情。

    部分队列（例如 example_queue_003）用 sourceType 拉完整队列时
    structureInfoList 为空，但 Jupiter 支持在同一个 endpoint 的 request
    中传 tagIds 直接返回规则详情。
    """
    request: Dict[str, Any] = {"tagIds": [str(t) for t in tag_ids if str(t).strip()]}
    if source_type:
        request["sourceTypeInfo"] = {
            "categoryType": category_type,
            "sourceType": source_type,
        }
    return _post_jupiter("assembleDetailedRuleJson", {
        "params": {
            "request": json.dumps(request, ensure_ascii=False)
        }
    })


def _fetch_sop_raw(source_type: str) -> dict:
    """接口3: 查询 SOP。"""
    return _post_jupiter("sopNodeDetailRelation", {
        "params": {
            "request": json.dumps({"sourceType": source_type})
        }
    })


def _fetch_notice_raw(source_type: str) -> dict:
    """接口5: 根据队列 source 查询审核须知。"""
    return _post_jupiter("common/queryAuditNotice", {
        "params": {
            "sourceType": source_type,
        }
    })


# ─────────────────────────────────────────────────────────────────────────────
# 审核须知 key 映射
# ─────────────────────────────────────────────────────────────────────────────

_NOTICE_KEY_MAP = {
    "globalRule": "全局规则",
    "judgmentLogic": "判断逻辑",
    "controlBasis": "管控基础",
    "globalExemption": "全局豁免",
    "frameStructure": "框架结构",
    "governanceBackground": "治理背景",
    "questionDefine": "问题定义",
}


# ─────────────────────────────────────────────────────────────────────────────
# 解析逻辑
# ─────────────────────────────────────────────────────────────────────────────


def _parse_rule_info(raw_info_str: str) -> dict:
    """解析单条规则的 info JSON 字符串，清洗 HTML，去掉冗余字段。"""
    info = json.loads(raw_info_str)
    result = {}

    # ruleName → 规则名称
    rule_name_items = info.get("ruleName", [])
    if rule_name_items:
        content = _parse_content(rule_name_items[0])
        result["规则名称"] = content.get("title", "")

    # riskLevel → 风险等级
    risk_items = info.get("riskLevel", [])
    if risk_items:
        content = _parse_content(risk_items[0])
        result["风险等级"] = content.get("desc", "")

    # tagDescription → 标签定义
    tag_desc_items = info.get("tagDescription", [])
    if tag_desc_items:
        content = _parse_content(tag_desc_items[0])
        _set_html_section(result, "标签定义", content.get("desc", ""))

    # auditPremise → 审核前提
    premise_items = info.get("auditPremise", [])
    if premise_items:
        content = _parse_content(premise_items[0])
        _set_html_section(result, "审核前提", content.get("desc", ""))

    # ruleList → 规则描述
    rule_list_items = info.get("ruleList", [])
    result["规则描述"] = []
    for item in rule_list_items:
        content = _parse_content(item)
        desc, images = _html_to_text_and_images(content.get("desc", ""))
        parsed_item = {
            "title": content.get("title", ""),
            "desc": desc,
        }
        if images:
            parsed_item["images"] = images
        result["规则描述"].append(parsed_item)

    # wholeExemption → 豁免
    exemption_items = info.get("wholeExemption", [])
    if exemption_items:
        content = _parse_content(exemption_items[0])
        _set_html_section(result, "豁免", content.get("desc", ""))

    # governanceExpectations → 管控依据
    gov_items = info.get("governanceExpectations", [])
    if gov_items:
        content = _parse_content(gov_items[0])
        _set_html_section(result, "管控依据", content.get("desc", ""))

    # nounRelated → 名词解释
    noun_items = info.get("nounRelated", [])
    result["名词解释"] = []
    for item in noun_items:
        content = _parse_content(item)
        desc, images = _html_to_text_and_images(content.get("desc", ""))
        parsed_item = {
            "title": content.get("title", ""),
            "desc": desc,
        }
        if images:
            parsed_item["images"] = images
        result["名词解释"].append(parsed_item)

    # knowledgeDevelopment → 知识拓展
    kd_items = info.get("knowledgeDevelopment", [])
    if kd_items:
        content = _parse_content(kd_items[0])
        raw_desc = content.get("desc", "")
        _set_html_section(result, "知识拓展", raw_desc)

    images = format_rule_images(result)
    if images:
        result["_images"] = images

    return result


def _parse_content(item: dict) -> dict:
    """解析 info 里某个 section item 的 content 字段（可能是 JSON 字符串或 dict）。"""
    content = item.get("content", {})
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            content = {"desc": content}
    return content if isinstance(content, dict) else {}


def _parse_notice(raw_response: dict) -> Optional[dict]:
    """解析接口5审核须知返回，清洗 HTML，key 转中文。"""
    data = raw_response.get("data", {})
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not data:
        return None

    notice_info_list_str = data.get("noticeInfoList", "[]")
    if isinstance(notice_info_list_str, str):
        try:
            notice_info_list = json.loads(notice_info_list_str)
        except (json.JSONDecodeError, ValueError):
            return None
    else:
        notice_info_list = notice_info_list_str

    if not notice_info_list:
        return None

    first_notice = notice_info_list[0]
    notice_info_raw = first_notice.get("noticeInfo", "[]")
    if isinstance(notice_info_raw, str):
        try:
            notice_info_raw = json.loads(notice_info_raw)
        except (json.JSONDecodeError, ValueError):
            return None

    if not notice_info_raw or not isinstance(notice_info_raw, list):
        return None

    ni = notice_info_raw[0]
    info_str = ni.get("info", "{}")
    if isinstance(info_str, str):
        try:
            info = json.loads(info_str)
        except (json.JSONDecodeError, ValueError):
            return None
    else:
        info = info_str

    result = {}
    all_images = []
    section_images = {}
    for key, items in info.items():
        cn_key = _NOTICE_KEY_MAP.get(key, key)
        if not isinstance(items, list) or not items:
            result[cn_key] = ""
            continue
        content = items[0].get("content", {})
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                content = {"desc": content}
        if isinstance(content, dict):
            raw_desc = content.get("desc", "")
            imgs = _extract_images_from_html(raw_desc)
            all_images.extend(imgs)
            if imgs:
                section_images[cn_key] = imgs
            result[cn_key] = _html_to_text(raw_desc)
        else:
            result[cn_key] = ""

    if all_images:
        result["_images"] = all_images
        result["_section_images"] = section_images

    return result


def _parse_sop(raw_response: dict) -> Optional[dict]:
    """解析接口3 SOP返回。"""
    data = raw_response.get("data", {})
    sop_data = data.get("data")
    sop_detail = data.get("sopDetail")

    if not sop_data and not sop_detail:
        return None

    nodes = []
    if isinstance(sop_data, list):
        for node in sop_data:
            nodes.append({
                "节点名称": node.get("nodeName", ""),
                "节点描述": node.get("nodeDesc", ""),
                "节点类型": node.get("nodeType", ""),
                "详情": [
                    {
                        "类型": d.get("type", ""),
                        "内容": _html_to_text(
                            json.loads(d.get("detailInfo", "{}")).get("content", "")
                            if d.get("detailInfo") else ""
                        ),
                    }
                    for d in node.get("detailDto", [])
                    if d.get("detailInfo")
                ],
            })

    flowchart = ""
    if sop_detail:
        if isinstance(sop_detail, str):
            try:
                sop_detail = json.loads(sop_detail)
            except (json.JSONDecodeError, ValueError):
                pass
        if isinstance(sop_detail, dict):
            flowchart = sop_detail.get("display", sop_detail.get("original", ""))

    return {"节点": nodes, "流程图": flowchart}




# ─────────────────────────────────────────────────────────────────────────────
# 公开 API
# ─────────────────────────────────────────────────────────────────────────────


def _parse_structure_rules(structure_list: List[dict]) -> tuple[Dict[str, dict], str, str, dict]:
    """Parse Jupiter structureInfoList into rules plus queue metadata."""
    biz_type = ""
    content_type = ""
    domain = {}
    if structure_list:
        first = structure_list[0]
        biz_type = first.get("bizType", "")
        content_type = first.get("contentType", "")
        domain = {
            "一级风险域": first.get("h1DomainId", {}).get("domainName", ""),
            "二级风险域": first.get("h2DomainId", {}).get("domainName", ""),
        }

    rules: Dict[str, dict] = {}
    for item in structure_list:
        parsed = _parse_rule_info(item["info"])
        tag_name = parsed.get("规则名称", "")
        if not tag_name:
            continue
        parsed["structureId"] = item.get("structureId", "")
        parsed["tagId"] = item.get("tagId", "")
        rules[tag_name] = parsed
    return rules, biz_type, content_type, domain


def fetch_queue_rules(source_type: str, category_type: str = "note") -> Dict[str, Any]:
    """查询队列的完整规则信息。

    Args:
        source_type: 队列 source，如 "example_queue_004"
        category_type: 业务类型，"note" 或 "comment"

    Returns:
        结构化 dict，包含：
        - source_type, 业务类型, 内容类型, 风险域
        - 审核须知: 各 section（中文 key）
        - SOP: SOP 信息（可能为 None）
        - 规则: {标签名: {...}} 所有规则
        - 标签优先级: 有序列表
    """
    # 1. 拉取所有规则
    rules_resp = _fetch_rules_raw(source_type, category_type)
    structure_list = (
        rules_resp.get("data", {}).get("structureInfoList", [])
    )

    rules, biz_type, content_type, domain = _parse_structure_rules(structure_list)

    # 2. 拉取 SOP
    sop_resp = _fetch_sop_raw(source_type)
    sop = _parse_sop(sop_resp)

    # 3. 拉取审核须知
    notice_resp = _fetch_notice_raw(source_type)
    audit_notice = _parse_notice(notice_resp)

    return {
        "source_type": source_type,
        "业务类型": biz_type,
        "内容类型": content_type,
        "风险域": domain,
        "审核须知": audit_notice,
        "SOP": sop,
        "规则": rules,
    }


def fetch_rules_by_tag_ids(
    tag_ids: List[str],
    source_type: Optional[str] = None,
    category_type: str = "note",
) -> Dict[str, dict]:
    """按 tagId 查询规则详情，返回 ``{标签名: 规则dict}``。

    这是 source + 中文 label miss 后的兜底路径。返回结构与
    ``fetch_queue_rules(...)[\"规则\"]`` 一致，可直接交给
    ``format_rule_markdown`` / ``format_rule_images`` 使用。
    """
    tag_ids = [str(t).strip() for t in tag_ids if str(t).strip()]
    if not tag_ids:
        return {}
    rules_resp = _fetch_rules_by_tag_ids_raw(tag_ids, source_type, category_type)
    structure_list = (
        rules_resp.get("data", {}).get("structureInfoList", [])
    )
    rules, _, _, _ = _parse_structure_rules(structure_list)
    return rules


def get_rule_by_tag(queue_info: Dict[str, Any], tag_name: str) -> Optional[dict]:
    """从 fetch_queue_rules 的返回中按标签名取单条规则。"""
    return queue_info.get("规则", {}).get(tag_name)


def format_rule_markdown(rule: dict) -> str:
    """将单条规则格式化为可注入 prompt 的 markdown 文本。"""
    lines = []

    tag_name = rule.get("规则名称", "")
    risk = rule.get("风险等级", "")
    lines.append(f"## {tag_name}")
    if risk:
        lines.append(f"**风险等级**: {risk}")
    lines.append("")

    tag_desc = rule.get("标签定义", "")
    if tag_desc or rule.get("标签定义图片"):
        tag_desc = _with_missing_image_tags(tag_desc, rule.get("标签定义图片"))
        lines.append(f"**标签定义**: {tag_desc}")
        lines.append("")

    premise = rule.get("审核前提", "")
    if premise or rule.get("审核前提图片"):
        premise = _with_missing_image_tags(premise, rule.get("审核前提图片"))
        lines.append(f"**审核前提**:\n{premise}")
        lines.append("")

    rule_list = rule.get("规则描述", [])
    if rule_list:
        lines.append("**规则描述**:")
        for item in rule_list:
            title = item.get("title", "")
            desc = item.get("desc", "")
            if title:
                lines.append(f"\n### {title}")
            desc = _with_missing_image_tags(desc, item.get("images"))
            if desc:
                lines.append(desc)
        lines.append("")

    exemption = rule.get("豁免", "")
    if exemption or rule.get("豁免图片"):
        exemption = _with_missing_image_tags(exemption, rule.get("豁免图片"))
        lines.append(f"**豁免**: {exemption}")
        lines.append("")

    nouns = rule.get("名词解释", [])
    if nouns:
        lines.append("**名词解释**:")
        for n in nouns:
            title = n.get("title", "")
            desc = n.get("desc", "")
            if title:
                lines.append(f"\n**{title}**")
            desc = _with_missing_image_tags(desc, n.get("images"))
            if desc:
                lines.append(desc)
        lines.append("")

    kd = rule.get("知识拓展", "")
    if kd or rule.get("知识拓展图片"):
        kd = _with_missing_image_tags(kd, rule.get("知识拓展图片"))
        lines.append(f"**知识拓展**: {kd}")
        lines.append("")

    gov = rule.get("管控依据", "")
    if gov or rule.get("管控依据图片"):
        gov = _with_missing_image_tags(gov, rule.get("管控依据图片"))
        lines.append(f"**管控依据**: {gov}")

    return "\n".join(lines)



def format_rule_images(rule: dict) -> List[str]:
    """返回与 ``format_rule_markdown`` 中 <image> 顺序严格一致的图片 URL。"""
    images: List[str] = []

    def _extend(values: Any) -> None:
        if isinstance(values, list):
            images.extend(str(v) for v in values if v)

    _extend(rule.get("标签定义图片"))
    _extend(rule.get("审核前提图片"))

    for item in rule.get("规则描述", []) or []:
        if isinstance(item, dict):
            _extend(item.get("images"))

    _extend(rule.get("豁免图片"))

    for item in rule.get("名词解释", []) or []:
        if isinstance(item, dict):
            _extend(item.get("images"))

    _extend(rule.get("知识拓展图片"))
    _extend(rule.get("管控依据图片"))
    return images


def _notice_keys_in_format_order(notice: dict) -> List[str]:
    preferred = list(_NOTICE_KEY_MAP.values())
    keys: List[str] = []
    for key in preferred:
        if key in notice:
            keys.append(key)
    for key in notice.keys():
        if key.startswith("_") or key in keys:
            continue
        keys.append(key)
    return keys


def format_queue_notice_markdown(queue_info: Dict[str, Any]) -> str:
    """格式化 source 级队列总规则 / 审核须知，供 label 规则前置使用。"""
    notice = queue_info.get("审核须知")
    if not isinstance(notice, dict) or not notice:
        return ""

    lines = ["## 队列总规则 / 审核须知"]
    source_type = queue_info.get("source_type", "")
    if source_type:
        lines.append(f"**source**: {source_type}")
    biz_type = queue_info.get("业务类型", "")
    content_type = queue_info.get("内容类型", "")
    if biz_type or content_type:
        lines.append(f"**业务/内容类型**: {biz_type or '未知'} / {content_type or '未知'}")
    domain = queue_info.get("风险域") or {}
    if isinstance(domain, dict) and any(domain.values()):
        sep = " > " if domain.get("一级风险域") and domain.get("二级风险域") else ""
        lines.append(f"**风险域**: {domain.get('一级风险域', '')}{sep}{domain.get('二级风险域', '')}")
    lines.append("")

    section_images = notice.get("_section_images") or {}
    for key in _notice_keys_in_format_order(notice):
        value = _with_missing_image_tags(notice.get(key, ""), section_images.get(key))
        if not value:
            continue
        lines.append(f"**{key}**:")
        lines.append(value)
        lines.append("")

    return "\n".join(lines).strip()


def format_queue_notice_images(queue_info: Dict[str, Any]) -> List[str]:
    """返回与 ``format_queue_notice_markdown`` 中 <image> 顺序一致的图片 URL。"""
    notice = queue_info.get("审核须知")
    if not isinstance(notice, dict) or not notice:
        return []

    section_images = notice.get("_section_images") or {}
    images: List[str] = []
    for key in _notice_keys_in_format_order(notice):
        values = section_images.get(key, [])
        if isinstance(values, list):
            images.extend(str(v) for v in values if v)
    return images
