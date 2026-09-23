"""用户资质工具：查询用户的交易类目资质 + 医疗资质 + 行业投放资质 + 蒲公英订单。

涉及服务:
- QueryApplyTradeTypeService: 用户报备通过的交易类目（品牌资质）
- QueryMedicalQualificationService: 用户医疗资质类目
- noteCommentUserQualInfoService: 行业投放资质 + 蒲公英订单 + KOS + 广告特批
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..services._base import DataBackend, OnlineBackend, OfflineBackend
from .base_tool import BaseTool


def _extract_trade_type(data: Optional[dict]) -> Dict[str, List[str]]:
    """从 QueryApplyTradeTypeService 返回中提取报备交易类目。"""
    if not data:
        return {}

    trade_map = data.get("tradeTypeMap", {})
    if isinstance(trade_map, str):
        try:
            trade_map = json.loads(trade_map)
        except (ValueError, TypeError):
            return {}

    if not isinstance(trade_map, dict):
        return {}

    return trade_map


def _extract_medical_qualification(data: Optional[dict]) -> List[str]:
    """从 QueryMedicalQualificationService 返回中提取医疗资质类目。"""
    if not data:
        return []

    categories = data.get("categoryNames", "[]")
    if isinstance(categories, str):
        try:
            categories = json.loads(categories)
        except (ValueError, TypeError):
            return []

    if not isinstance(categories, list):
        return []

    return [c for c in categories if isinstance(c, str) and c]


def _extract_qual_info(data: Optional[dict]) -> Dict[str, Any]:
    """从 noteCommentUserQualInfoService 返回中提取资质信息。"""
    if not data:
        return {}

    result = {
        "is_kos": str(data.get("isKosUser", "")).lower() == "true",
        "is_pgy_note": str(data.get("isPgyNote", "")).lower() == "true",
    }

    # 行业投放资质
    qual_list = data.get("industryAndQualInfoList", "[]")
    if isinstance(qual_list, str):
        try:
            qual_list = json.loads(qual_list)
        except (ValueError, TypeError):
            qual_list = []

    qualifications = []
    if isinstance(qual_list, list):
        for q in qual_list:
            if not isinstance(q, dict):
                continue
            trade_first = q.get("tradeTypeFirstName", "")
            trade_second = q.get("tradeTypeSecondName", "")
            trade_name = f"{trade_first}-{trade_second}" if trade_second else trade_first

            cert_list = q.get("qualificationList", [])
            if isinstance(cert_list, str):
                try:
                    cert_list = json.loads(cert_list)
                except (ValueError, TypeError):
                    cert_list = []

            certs = []
            if isinstance(cert_list, list):
                for cert in cert_list:
                    if isinstance(cert, dict) and cert.get("qualificationName"):
                        certs.append(cert["qualificationName"])

            if trade_name or certs:
                qualifications.append({
                    "trade": trade_name,
                    "certs": certs,
                    "status": q.get("auditStatus", ""),
                })

    result["qualifications"] = qualifications

    # 蒲公英订单
    pgy_info = data.get("pgyNoteInfo", "")
    if isinstance(pgy_info, str) and pgy_info:
        try:
            pgy_info = json.loads(pgy_info)
        except (ValueError, TypeError):
            pgy_info = {}

    if isinstance(pgy_info, dict) and pgy_info:
        result["pgy_order"] = {
            "brand_user_id": pgy_info.get("brandUserId", ""),
            "kol_user_id": pgy_info.get("kolUserId", ""),
            "order_id": str(pgy_info.get("orderId", "")),
            "order_status": pgy_info.get("orderStatus", ""),
        }

    # 广告特批
    ad_special = data.get("adSpecialApprovalList", "[]")
    if isinstance(ad_special, str):
        try:
            ad_special = json.loads(ad_special)
        except (ValueError, TypeError):
            ad_special = []

    if isinstance(ad_special, list) and ad_special:
        result["ad_special_approvals"] = ad_special

    return result


def _render_text(trade_map: dict, medical: list, qual_info: dict) -> str:
    """渲染用户资质为文本。"""
    lines = ["## 用户资质", ""]

    # 报备交易类目
    if trade_map:
        lines.append("### 报备交易类目")
        for category, sub_types in trade_map.items():
            if isinstance(sub_types, list):
                lines.append(f"- {category}: {', '.join(sub_types)}")
            else:
                lines.append(f"- {category}: {sub_types}")
        lines.append("")
    else:
        lines.append("### 报备交易类目")
        lines.append("- 无")
        lines.append("")

    # 医疗资质
    if medical:
        lines.append("### 医疗资质")
        lines.append(f"- {', '.join(medical)}")
        lines.append("")

    # 行业投放资质
    qualifications = qual_info.get("qualifications", [])
    if qualifications:
        lines.append("### 行业投放资质")
        for q in qualifications:
            trade = q.get("trade", "")
            certs = q.get("certs", [])
            status = q.get("status", "")
            status_str = f" ({status})" if status else ""
            if certs:
                lines.append(f"- {trade}: {', '.join(certs)}{status_str}")
            else:
                lines.append(f"- {trade}{status_str}")
        lines.append("")

    # KOS / 蒲公英
    flags = []
    if qual_info.get("is_kos"):
        flags.append("KOS用户")
    if qual_info.get("is_pgy_note"):
        flags.append("蒲公英笔记")
    if flags:
        lines.append(f"### 身份标识")
        lines.append(f"- {', '.join(flags)}")
        lines.append("")

    # 蒲公英订单详情
    pgy_order = qual_info.get("pgy_order")
    if pgy_order:
        lines.append("### 蒲公英订单")
        lines.append(f"- 品牌方: {pgy_order['brand_user_id']}")
        lines.append(f"- KOL: {pgy_order['kol_user_id']}")
        lines.append(f"- 订单号: {pgy_order['order_id']}")
        lines.append(f"- 状态: {pgy_order['order_status']}")
        lines.append("")

    # 广告特批
    ad_special = qual_info.get("ad_special_approvals", [])
    if ad_special:
        lines.append("### 广告特批")
        for item in ad_special:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines)


class GetUserQualificationTool(BaseTool):
    name = "get_user_qualification"
    description = (
        "查询用户资质：报备交易类目 + 医疗资质 + 行业投放资质 + 蒲公英订单。"
        "用于判断用户是否有合法经营/医疗/投放资质。"
    )
    brief = "用户资质：查询交易类目、医疗资质、行业投放资质、蒲公英订单、KOS身份。"
    when = "判断用户是否有报备交易类目资质；候选标签涉及营销/带货/商品/资质/医疗相关；需要查看蒲公英合作详情；判断是否KOS用户。"
    input_schema = {"note_id": "待审笔记ID"}
    output_schema = {
        "text": "string  用户资质 Markdown",
        "trade_types": "dict  报备交易类目",
        "medical_qualification": "list  医疗资质类目",
        "is_kos": "bool  是否KOS用户",
        "is_pgy_note": "bool  是否蒲公英笔记",
    }
    cost = 1
    image_mode = "inline"
    RESULT_TEMPLATE = "【用户资质】\n{{ text }}"

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        note_id = args.get("note_id")
        user_id = args.get("user_id", "")
        if not note_id:
            return {"text": "", "error": "missing note_id"}

        calls_blob = args.get("_calls_blob")
        if not calls_blob:
            from .check_plagiarism_video import _get_blob_index
            calls_blob = _get_blob_index().get(note_id, "")

        if calls_blob:
            backend = OfflineBackend(calls_blob, note_id=note_id)
        else:
            backend = OnlineBackend()

        if not user_id:
            note_info = backend.call("queryNoteInfo", noteId=note_id)
            if note_info:
                info = note_info.get("noteInfo", note_info)
                detail = info.get("noteDetailInfo", {}) if isinstance(info, dict) else {}
                user_id = detail.get("userId", "") if isinstance(detail, dict) else ""

        # 交易类目
        trade_data = backend.call("QueryApplyTradeTypeService", noteId=note_id, userId=user_id)
        trade_map = _extract_trade_type(trade_data)

        # 医疗资质
        medical_data = backend.call("QueryMedicalQualificationService", noteId=note_id, userId=user_id)
        medical = _extract_medical_qualification(medical_data)

        # 行业投放资质 + 蒲公英 + KOS
        qual_data = backend.call("noteCommentUserQualInfoService", noteId=note_id)
        qual_info = _extract_qual_info(qual_data)

        text = _render_text(trade_map, medical, qual_info)

        return {
            "text": text,
            "trade_types": trade_map,
            "medical_qualification": medical,
            "is_kos": qual_info.get("is_kos", False),
            "is_pgy_note": qual_info.get("is_pgy_note", False),
            "display_images": [],
        }


TOOL = GetUserQualificationTool()
