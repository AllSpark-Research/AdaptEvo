"""审核工具层：agent 可调用的 tool。

每个工具是一个自包含的 py，实现一个 ``BaseTool`` 子类并在模块底部导出实例
``TOOL``。framework 只依赖 ``BaseTool`` 接口（run / render / brief_entry）。

Active tools:
- check_plagiarism_video: 视频搬运检测
- check_plagiarism_image: 图文搬运检测
- get_commercial_detail: 商业详情（商品SKU）
- get_user_qualification: 用户资质（交易类目 + 医疗资质 + 行业投放资质 + 蒲公英）
- get_recent_notes: 用户近期笔记
- get_comments: 完整评论列表
- get_user_records: 用户资料修改记录
- get_report_history: 用户举报记录（接受/成立）
- get_user_comments: 用户在其他笔记下的评论

加一个新工具：
1. 新建 ``tools/<name>.py``，继承 ``BaseTool`` 填元数据 + 实现 run()（简单工具设
   RESULT_TEMPLATE，复杂工具 override render()），底部 ``TOOL = XxxTool()``。
2. 在下面 ``_TOOL_INSTANCES`` 加一行 import + 注册。
"""

from __future__ import annotations

from typing import Dict, List

from .base_tool import BaseTool
from .check_plagiarism_video import TOOL as _cpv
from .check_plagiarism_image import TOOL as _cpi
from .get_commercial_detail import TOOL as _gcd
from .get_user_qualification import TOOL as _guq
from .get_recent_notes import TOOL as _grn
from .get_comments import TOOL as _gc
from .get_user_records import TOOL as _gur
from .get_report_history import TOOL as _grh
from .get_user_comments import TOOL as _guc

_TOOL_INSTANCES: List[BaseTool] = [_cpv, _cpi, _gcd, _guq, _grn, _gc, _gur, _grh, _guc]

# name -> BaseTool 实例
ACTIVE_TOOLS: Dict[str, BaseTool] = {t.name: t for t in _TOOL_INSTANCES}


def load_tool_briefs(only_active: bool = True) -> List[Dict]:
    """Planner 看到的 brief 列表，从每个工具的类属性自动派生。"""
    return [t.brief_entry() for t in _TOOL_INSTANCES]


__all__ = [
    "ACTIVE_TOOLS",
    "load_tool_briefs",
    "BaseTool",
]
