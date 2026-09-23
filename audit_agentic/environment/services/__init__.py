"""审核服务调用层：基础服务 backend + 笔记内容获取。

services/ 只提供底层能力：
- _base.py: DataBackend / OnlineBackend / OfflineBackend
- note_content.py: fetch_note_content / render_note_content / render_note_content_text
"""

from ._base import DataBackend, OfflineBackend, OnlineBackend
from .note_content import (
    fetch_note_content,
    get_note_detail,
    render_note_content,
    render_note_content_text,
)
from .rule_query import (
    fetch_queue_rules,
    format_rule_markdown,
    get_rule_by_tag,
)

__all__ = [
    "DataBackend",
    "OnlineBackend",
    "OfflineBackend",
    "fetch_note_content",
    "get_note_detail",
    "render_note_content",
    "render_note_content_text",
    "fetch_queue_rules",
    "format_rule_markdown",
    "get_rule_by_tag",
]
