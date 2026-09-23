"""Train-only historical precedent comparison research component.

The maintained AgentScope workflow does not register this tool yet. It is
retained for a future optional history-tool experiment, not as a workflow gate.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import threading
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ...agents.multimodal import make_multimodal_message
from .base_tool import BaseTool, ToolRenderResult


PASS_LABEL = "通过"
HISTORY_TOOL_NAME = "compare_historical_audit_cases"
HISTORY_CONTEXT_LEGACY_COMPACT = "legacy_compact"
HISTORY_CONTEXT_FULL_MULTIMODAL = "full_multimodal"
AUDIT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DATA = AUDIT_ROOT / "data" / "train_local.jsonl"
DEFAULT_COLLAGE_CACHE = AUDIT_ROOT / "environment" / "cache" / "history_collages"
_COLLAGE_LOCK = threading.RLock()


def _label_name(value: Any) -> str:
    return str(value or "").split("|", 1)[-1].strip()


def _labels(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_label_name(item) for item in value if _label_name(item)]
    if isinstance(value, str):
        return [_label_name(item) for item in value.split(",") if _label_name(item)]
    return []


def _metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("metadata") or {}


def _source(row: Dict[str, Any]) -> str:
    return str(_metadata(row).get("source") or row.get("source") or "")


def _note_id(row: Dict[str, Any]) -> str:
    md = _metadata(row)
    audit_input = md.get("audit_input") or {}
    return str(audit_input.get("note_id") or md.get("note_id") or row.get("note_id") or "")


def _note(row: Dict[str, Any]) -> str:
    md = _metadata(row)
    audit_input = md.get("audit_input") or {}
    return str(audit_input.get("note") or row.get("note") or "")


def _images(row: Dict[str, Any]) -> List[str]:
    md = _metadata(row)
    audit_input = md.get("audit_input") or {}
    values = audit_input.get("images") or row.get("images") or []
    return [str(item) for item in values if str(item)]


def _gt_labels(row: Dict[str, Any]) -> List[str]:
    md = _metadata(row)
    audit_input = md.get("audit_input") or {}
    return _labels(
        audit_input.get("raw_gt_labels")
        or audit_input.get("gt_labels")
        or md.get("labels")
        or row.get("label")
    )


def _is_pass(labels: Iterable[str]) -> bool:
    values = list(labels)
    return not values or set(values) == {PASS_LABEL}


def _clean(text: Any) -> str:
    value = str(text or "")
    value = value.replace("<image>", "[图片]")
    value = re.sub(r"/mnt/\S+", "[本地图片]", value)
    value = re.sub(r"https?://\S+", "[链接]", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _full_note(text: Any) -> str:
    # Original notes contain one <image> marker per raw image. History Analyst
    # receives collages instead, so preserve marker position as text without
    # confusing multimodal alignment.
    return str(text or "").replace("<image>", "[原始图片位置]").strip()


def _compact(text: Any, limit: int) -> str:
    value = _clean(text)
    if len(value) <= limit:
        return value
    head = max(1, int(limit * 0.8))
    tail = max(1, limit - head - 16)
    return value[:head] + "\n...[压缩]...\n" + value[-tail:]


def _normalized(text: str) -> str:
    return re.sub(r"\W+", "", _clean(text)).lower()


def _sample_images(values: List[str], limit: int) -> List[str]:
    images = [item for item in values if item and Path(item).exists()]
    if limit <= 0 or len(images) <= limit:
        return images
    if limit == 1:
        return [images[0]]
    indices = [round(i * (len(images) - 1) / (limit - 1)) for i in range(limit)]
    return [images[index] for index in dict.fromkeys(indices)]


def _collage_cache_dir() -> Path:
    return Path(os.environ.get("AUDIT_HISTORY_COLLAGE_CACHE_DIR") or DEFAULT_COLLAGE_CACHE)


def _collage_key(paths: List[str], cell_size: int, columns: int) -> str:
    items: List[str] = [f"v2-upscale-contain|cell={cell_size}|columns={columns}"]
    for path in paths:
        try:
            stat = os.stat(path)
            items.append(f"{path}|{stat.st_size}|{stat.st_mtime_ns}")
        except OSError:
            items.append(f"{path}|missing")
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def _open_for_collage(path: str, image_module: Any) -> Any:
    with _COLLAGE_LOCK:
        old_limit = image_module.MAX_IMAGE_PIXELS
        image_module.MAX_IMAGE_PIXELS = None
        try:
            return image_module.open(path)
        finally:
            image_module.MAX_IMAGE_PIXELS = old_limit


def _build_collage(paths: List[str], image_max_tokens: int) -> str | None:
    paths = [path for path in paths if path and Path(path).exists()]
    if not paths:
        return None
    try:
        from PIL import Image, ImageDraw, ImageOps
    except Exception:
        return None

    columns = min(3, len(paths))
    rows = max(1, math.ceil(len(paths) / columns))
    max_pixels = max(1, int(image_max_tokens)) * 32 * 32
    cell_size = max(192, int(math.sqrt(max_pixels / max(1, columns * rows))))
    cell_size = max(32, (cell_size // 32) * 32)
    digest = _collage_key(paths, cell_size, columns)
    target = _collage_cache_dir() / digest[:2] / f"{digest}.jpg"
    if target.exists() and target.stat().st_size > 0:
        return str(target)

    canvas = Image.new("RGB", (columns * cell_size, rows * cell_size), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(paths):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Corrupt EXIF data.*", category=UserWarning)
                image_ctx = _open_for_collage(path, Image)
            with image_ctx as source:
                source.draft("RGB", (cell_size, cell_size))
                source.load()
                try:
                    source = ImageOps.exif_transpose(source)
                except Exception:
                    pass
                if source.mode == "RGBA":
                    background = Image.new("RGB", source.size, (255, 255, 255))
                    background.paste(source, mask=source.split()[3])
                    source = background
                else:
                    source = source.convert("RGB")
                resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
                max_side = cell_size - 8
                scale = min(max_side / source.width, max_side / source.height)
                resized_size = (
                    max(1, round(source.width * scale)),
                    max(1, round(source.height * scale)),
                )
                if resized_size != source.size:
                    source = source.resize(resized_size, resampling)
                col = index % columns
                row = index // columns
                left = col * cell_size + (cell_size - source.width) // 2
                top = row * cell_size + (cell_size - source.height) // 2
                canvas.paste(source, (left, top))
                draw.rectangle(
                    (col * cell_size + 4, row * cell_size + 4, col * cell_size + 32, row * cell_size + 28),
                    fill=(0, 0, 0),
                )
                draw.text((col * cell_size + 11, row * cell_size + 7), str(index + 1), fill=(255, 255, 255))
        except Exception:
            continue

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        canvas.save(tmp, format="JPEG", quality=90, optimize=True)
        os.replace(tmp, target)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return str(target)


def _collages(values: List[str], image_max_tokens: int) -> List[str]:
    per_collage = max(1, min(9, int(os.environ.get("AUDIT_HISTORY_IMAGES_PER_COLLAGE", "6"))))
    max_images = max(0, int(os.environ.get("AUDIT_HISTORY_MAX_IMAGES_PER_CASE", "24")))
    sampled = _sample_images(values, max_images)
    output: List[str] = []
    for start in range(0, len(sampled), per_collage):
        collage = _build_collage(sampled[start : start + per_collage], image_max_tokens)
        if collage:
            output.append(collage)
    return output


def _observation_summary(observations: Any, limit: int = 1500) -> str:
    if not isinstance(observations, list) or not observations:
        return "（当前尚无普通工具证据）"
    rendered: List[Dict[str, Any]] = []
    for item in observations:
        if not isinstance(item, dict):
            continue
        result = item.get("result")
        if isinstance(result, dict):
            result = {
                key: value
                for key, value in result.items()
                if key not in {"display_images", "images", "image_urls", "urls"}
            }
        rendered.append(
            {
                "tool_name": item.get("tool_name"),
                "status": item.get("status"),
                "result": _compact(json.dumps(result, ensure_ascii=False), 420),
                "error": _compact(item.get("error"), 120),
            }
        )
    return _compact(json.dumps(rendered, ensure_ascii=False), limit)


class _SourceIndex:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        binary_by_text: Dict[str, set[bool]] = defaultdict(set)
        for row in rows:
            binary_by_text[_normalized(_note(row))].add(_is_pass(_gt_labels(row)))
        conflicts = {text for text, values in binary_by_text.items() if len(values) > 1}
        self.rows = [row for row in rows if _normalized(_note(row)) not in conflicts and _note(row)]
        self.texts = [_note(row) for row in self.rows]
        self.norms = [_normalized(text) for text in self.texts]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            max_features=60000,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(
        self,
        query_note: str,
        query_summary: str,
        note_id: str,
        focus_labels: List[str],
        per_class: int,
    ) -> List[Dict[str, Any]]:
        query_text = "\n".join(value for value in [query_note, query_summary] if value)
        scores = cosine_similarity(self.vectorizer.transform([query_text]), self.matrix).ravel()
        order = scores.argsort()[::-1]
        focus = set(focus_labels) - {PASS_LABEL}
        selected: List[Dict[str, Any]] = []
        counts = {True: 0, False: 0}
        query_norm = _normalized(query_note)
        seen: set[str] = set()

        def collect(require_label_overlap: bool) -> None:
            for raw_idx in order:
                idx = int(raw_idx)
                row = self.rows[idx]
                norm = self.norms[idx]
                if norm == query_norm or norm in seen or _note_id(row) == note_id:
                    continue
                labels = _gt_labels(row)
                pass_case = _is_pass(labels)
                if counts[pass_case] >= per_class:
                    continue
                if require_label_overlap and not pass_case and focus and not (focus & set(labels)):
                    continue
                selected.append(
                    {
                        "case_id": f"train:{_source(row)}:{_note_id(row)}",
                        "binary_gt": PASS_LABEL if pass_case else "违规",
                        "gt_labels": labels,
                        "similarity": round(float(scores[idx]), 4),
                        "note": _note(row),
                        "images": _images(row),
                    }
                )
                counts[pass_case] += 1
                seen.add(norm)
                if all(value >= per_class for value in counts.values()):
                    return

        collect(require_label_overlap=True)
        if not all(value >= per_class for value in counts.values()):
            collect(require_label_overlap=False)
        return selected


class _HistoryCorpus:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    self.rows_by_source[_source(row)].append(row)
        self._indexes: Dict[str, _SourceIndex] = {}
        self._lock = threading.RLock()

    def index(self, source: str) -> _SourceIndex | None:
        rows = self.rows_by_source.get(source) or []
        if not rows:
            return None
        with self._lock:
            if source not in self._indexes:
                self._indexes[source] = _SourceIndex(rows)
            return self._indexes[source]


_CORPORA: Dict[str, _HistoryCorpus] = {}
_CORPORA_LOCK = threading.RLock()


def _corpus(path: Path) -> _HistoryCorpus:
    key = f"{path.resolve()}:{path.stat().st_mtime_ns}"
    with _CORPORA_LOCK:
        if key not in _CORPORA:
            _CORPORA.clear()
            _CORPORA[key] = _HistoryCorpus(path)
        return _CORPORA[key]


def _history_schema(candidate_labels: List[str]) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "historical_audit_comparison",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "applicability": {
                        "type": "string",
                        "enum": ["usable", "weak", "insufficient"],
                    },
                    "pass_analogies": {"type": "array", "items": {"type": "string"}},
                    "violation_analogies": {"type": "array", "items": {"type": "string"}},
                    "decisive_boundary": {"type": "array", "items": {"type": "string"}},
                    "current_alignment": {
                        "type": "string",
                        "enum": [PASS_LABEL, "违规", "不确定"],
                    },
                    "recommended_labels": {
                        "type": "array",
                        "items": {"type": "string", "enum": candidate_labels},
                        "uniqueItems": True,
                    },
                    "supporting_case_ids": {"type": "array", "items": {"type": "string"}},
                    "conflict_warnings": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "applicability",
                    "pass_analogies",
                    "violation_analogies",
                    "decisive_boundary",
                    "current_alignment",
                    "recommended_labels",
                    "supporting_case_ids",
                    "conflict_warnings",
                ],
                "additionalProperties": False,
            },
        },
    }


def _history_prompt_full_multimodal(
    context: Dict[str, Any],
    args: Dict[str, Any],
    cases: List[Dict[str, Any]],
    image_max_tokens: int,
) -> tuple[str, List[str]]:
    rules = context.get("rules") or []
    rule_text = "\n\n".join(
        f"【{item.get('label')}】\n{str(item.get('rule_text') or '')}"
        for item in rules
        if isinstance(item, dict)
    )
    current_collages = _collages(list(context.get("images") or []), image_max_tokens)
    visual_images: List[str] = list(current_collages)
    current_visual = "\n".join("<image>" for _ in current_collages) or "（无可读取的当前帖子图片）"

    cards_list: List[str] = []
    for case in cases:
        case_collages = _collages(list(case.get("images") or []), image_max_tokens)
        visual_images.extend(case_collages)
        history_visual = "\n".join("<image>" for _ in case_collages) or "（无可读取的历史图片）"
        cards_list.append(
            f"### {case['case_id']}\n"
            f"- 历史二分类: {case['binary_gt']}\n"
            f"- 历史细标签: {', '.join(case['gt_labels'])}\n"
            f"- 文本/OCR相似度: {case['similarity']:.4f}\n"
            f"- 历史帖子完整文本、ASR与OCR:\n{_full_note(case.get('note'))}\n"
            f"- 历史帖子视觉证据（每张联系图最多包含 6 张原图，保持原顺序）:\n{history_visual}"
        )
    cards = "\n\n".join(cards_list)
    prompt = f"""你是历史审核案例对比 Subagent。只负责比较当前 case 与 Train 历史先例的边界，不拥有最终裁决权。

证据优先级：当前帖子与当前工具事实 > 当前规则和明确豁免 > 可靠历史先例。历史 GT 可能含噪，不得按数量投票，不得从账号身份或常见默认字段创造规则中不存在的豁免。

# 当前 source
{context.get('source_id')}

# 当前帖子
{_full_note(context.get('note'))}

# 当前帖子视觉证据
以下每张联系图最多包含 6 张当前帖子原图，按原始图片顺序排列；每个格子左上角数字表示其在该联系图中的顺序。
{current_visual}

# 主 Agent 对当前视觉/语境的摘要
{_compact(args.get('current_case_summary'), 700) or '（未提供）'}

# 对比问题
{_compact(args.get('evidence_question'), 500) or '判断当前内容是否达到违规门槛'}

# 当前候选标签
{', '.join(context.get('candidate_labels') or [])}

# 当前规则
{rule_text or '（未加载）'}

# 当前普通工具证据
{_observation_summary(context.get('observations'))}

# Train 历史双向案例
{cards or '（无可靠历史案例）'}

# 分析要求
1. 分别说明当前 case 与历史通过、历史违规案例的同构点和关键差异。
2. 先解决通过/违规二分类，再讨论细标签；不要只因关键词相似而改判。
3. 文本相似度低于 0.20 的案例只能作为弱参考。
4. 如果规则版本、可见证据或案例同构性不足，applicability 必须选择 weak 或 insufficient。
5. recommended_labels 只能来自当前候选标签；不确定时可返回空数组。
6. 输出紧凑、可审计的边界结论，不输出长篇自由推理。
7. 历史通过案例只有在当前 case 同样缺少相同的必要违规条件，或同样满足规则明确写出的豁免条件时，才能支持判通过；主题、账号、关键词或画面风格相似本身不能构成豁免。
8. 对视觉标签必须比较画面主体、面积、清晰度、冲击程度和动态/静态条件；不得仅根据标题、ASR或OCR推定图片和视频中未直接展示的内容。
"""
    return prompt, visual_images


def _history_prompt_legacy_compact(
    context: Dict[str, Any],
    args: Dict[str, Any],
    cases: List[Dict[str, Any]],
    image_max_tokens: int,
) -> tuple[str, List[str]]:
    del image_max_tokens

    rule_sections: List[str] = []
    remaining_rule_chars = 3200
    for item in context.get("rules") or []:
        if not isinstance(item, dict) or remaining_rule_chars <= 0:
            continue
        label = str(item.get("label") or "")
        body = _compact(item.get("rule_text"), min(700, remaining_rule_chars))
        section = f"【{label}】\n{body}"[:remaining_rule_chars]
        rule_sections.append(section)
        remaining_rule_chars -= len(section)

    cards: List[str] = []
    for case in cases:
        cards.append(
            f"### {case['case_id']}\n"
            f"- 历史二分类: {case['binary_gt']}\n"
            f"- 历史细标签: {', '.join(case['gt_labels'])}\n"
            f"- 文本/OCR相似度: {case['similarity']:.4f}\n"
            f"- 历史帖子摘要: {_compact(case.get('note'), 420)}"
        )

    prompt = f"""你是历史审核案例对比 Subagent。只负责比较当前 case 与 Train 历史先例的边界，不拥有最终裁决权。

证据优先级：当前帖子与当前工具事实 > 当前规则和明确豁免 > 可靠历史先例。历史 GT 可能含噪，不得按数量投票，不得从账号身份或常见默认字段创造规则中不存在的豁免。

# 当前 source
{context.get('source_id')}

# 当前帖子（压缩文本、ASR 与 OCR）
{_compact(context.get('note'), 1800)}

# 主 Agent 对当前 case 的摘要
{_compact(args.get('current_case_summary'), 700) or '（未提供）'}

# 对比问题
{_compact(args.get('evidence_question'), 500) or '判断当前内容是否达到违规门槛'}

# 当前候选标签
{', '.join(context.get('candidate_labels') or [])}

# 当前规则（每条最多约 700 字，整体最多约 3200 字）
{chr(10).join(rule_sections) or '（未加载）'}

# 当前普通工具证据
{_observation_summary(context.get('observations'))}

# Train 历史双向案例（每条帖子摘要最多约 420 字）
{chr(10).join(cards) or '（无可靠历史案例）'}

# 分析要求
1. 分别说明当前 case 与历史通过、历史违规案例的同构点和关键差异。
2. 先解决通过/违规二分类，再讨论细标签；不要只因关键词相似而改判。
3. 文本相似度低于 0.20 的案例只能作为弱参考。
4. 如果规则版本、可见证据或案例同构性不足，applicability 必须选择 weak 或 insufficient。
5. recommended_labels 只能来自当前候选标签；不确定时可返回空数组。
6. 输出紧凑、可审计的边界结论，不输出长篇自由推理。
7. 历史通过案例只有在当前 case 同样缺少相同的必要违规条件，或同样满足规则明确写出的豁免条件时，才能支持判通过。
8. 本模式不向 History Analyst 提供当前或历史图片；视觉证据不足时必须降低 applicability，并禁止仅凭标题、ASR 或 OCR 推断未直接展示的画面。
"""
    return prompt, []


def _history_context_mode() -> str:
    value = str(
        os.environ.get("AUDIT_HISTORY_CONTEXT_MODE") or HISTORY_CONTEXT_FULL_MULTIMODAL
    ).strip().lower()
    if value not in {HISTORY_CONTEXT_LEGACY_COMPACT, HISTORY_CONTEXT_FULL_MULTIMODAL}:
        return HISTORY_CONTEXT_FULL_MULTIMODAL
    return value


def _history_prompt(
    context: Dict[str, Any],
    args: Dict[str, Any],
    cases: List[Dict[str, Any]],
    image_max_tokens: int,
    context_mode: str,
) -> tuple[str, List[str]]:
    if context_mode == HISTORY_CONTEXT_LEGACY_COMPACT:
        return _history_prompt_legacy_compact(context, args, cases, image_max_tokens)
    return _history_prompt_full_multimodal(context, args, cases, image_max_tokens)


class HistoricalAuditCasesTool(BaseTool):
    name = HISTORY_TOOL_NAME
    description = (
        "从严格隔离的 Train 历史审核库中，同时检索相似的通过与违规案例，并由历史对比 Subagent "
        "总结决定性边界。历史先例只用于主观尺度校准，不能覆盖当前明确规则和客观工具事实。"
    )
    brief = "对比 Train 历史通过/违规先例，校准主观二分类边界"
    when = (
        "仅当当前 source/标签具有主观尺度、通过与违规证据并存、普通工具无法直接定性时调用；"
        "搬运比例、资质硬条件、交易路径等客观证据已经明确时不要调用。每条 case 最多调用一次。"
    )
    public_input_schema = {
        "type": "object",
        "properties": {
            "comparison_axis": {
                "type": "string",
                "enum": ["binary_boundary", "label_boundary"],
                "description": "优先选择 binary_boundary；只有二分类基本明确后才比较细标签。",
            },
            "focus_labels": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
                "description": "当前最需要比较尺度的候选标签名称。",
            },
            "current_case_summary": {
                "type": "string",
                "description": "基于当前帖子和图片，对争议行为、表达强度或画面尺度的简洁事实摘要。",
            },
            "evidence_question": {
                "type": "string",
                "description": "希望历史先例帮助回答的具体审核边界问题。",
            },
        },
        "required": ["comparison_axis", "focus_labels", "current_case_summary", "evidence_question"],
        "additionalProperties": False,
    }
    input_schema = {"source_id": "运行时注入", "note_id": "运行时注入"}
    output_schema = {"text": "string", "history_cases": "array", "analysis": "object"}
    cost = 2
    cacheable = False
    RESULT_TEMPLATE = "{{ text }}"

    def __init__(self, llm_client: Any):
        self.llm = llm_client

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        context = args.get("_audit_context") or {}
        context_mode = _history_context_mode()
        source = str(context.get("source_id") or args.get("source_id") or "")
        note_id = str(context.get("note_id") or args.get("note_id") or "")
        candidate_labels = [str(item) for item in context.get("candidate_labels") or []]
        focus_labels = [
            str(item)
            for item in args.get("focus_labels") or []
            if str(item) in set(candidate_labels)
        ]
        history_path = Path(os.environ.get("AUDIT_HISTORY_DATA_PATH") or DEFAULT_HISTORY_DATA)
        if not history_path.exists():
            return {
                "text": f"历史案例工具不可用：Train 历史库不存在（{history_path}）。",
                "history_cases": [],
                "analysis": {"applicability": "insufficient"},
                "history_snapshot": str(history_path),
                "history_context_mode": context_mode,
            }
        source_index = _corpus(history_path).index(source)
        if source_index is None:
            return {
                "text": f"历史案例工具无可用先例：Train 历史库中没有 source={source}。",
                "history_cases": [],
                "analysis": {"applicability": "insufficient"},
                "history_snapshot": str(history_path),
                "history_context_mode": context_mode,
            }
        per_class = max(1, min(4, int(os.environ.get("AUDIT_HISTORY_CASES_PER_CLASS", "3"))))
        cases = source_index.retrieve(
            str(context.get("note") or ""),
            str(args.get("current_case_summary") or ""),
            note_id,
            focus_labels,
            per_class,
        )
        if not cases:
            return {
                "text": "历史案例工具未找到可靠的 Train 双向先例，本次不要依据历史改判。",
                "history_cases": [],
                "analysis": {"applicability": "insufficient"},
                "history_snapshot": str(history_path),
                "history_context_mode": context_mode,
            }
        image_max_tokens = max(
            64,
            int(os.environ.get("AUDIT_HISTORY_IMAGE_MAX_TOKENS", "1024")),
        )
        prompt, analyst_images = _history_prompt(
            context,
            args,
            cases,
            image_max_tokens,
            context_mode,
        )
        response = self.llm.chat_with_meta(
            [
                {
                    "role": "system",
                    "content": "你是严谨的历史审核案例对比助手，只输出符合 JSON Schema 的结果。",
                },
                make_multimodal_message(
                    "user",
                    prompt,
                    analyst_images,
                    image_max_tokens=image_max_tokens,
                ),
            ],
            temperature=0.0,
            max_tokens=int(os.environ.get("AUDIT_HISTORY_ANALYST_MAX_TOKENS", "1000")),
            response_format=_history_schema(candidate_labels),
            extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
        )
        try:
            analysis = json.loads(response.content)
        except Exception as exc:
            analysis = {
                "applicability": "weak",
                "current_alignment": "不确定",
                "recommended_labels": [],
                "conflict_warnings": [f"History Analyst 输出解析失败：{type(exc).__name__}"],
            }
        analyst_raw_analysis = dict(analysis)
        if analysis.get("applicability") == "insufficient":
            analysis["current_alignment"] = "不确定"
            analysis["recommended_labels"] = []
            warnings_list = list(analysis.get("conflict_warnings") or [])
            warnings_list.append("历史案例同构性不足，本次历史结果不得支持主 Agent 改判。")
            analysis["conflict_warnings"] = list(dict.fromkeys(warnings_list))
        compact_cases = [
            {
                "case_id": case["case_id"],
                "binary_gt": case["binary_gt"],
                "gt_labels": case["gt_labels"],
                "similarity": case["similarity"],
                "note_summary": _compact(case.get("note"), 420),
                "original_image_count": len(case.get("images") or []),
            }
            for case in cases
        ]
        text = (
            "### Train 历史审核边界对比\n"
            f"历史库：{history_path.name}；source={source}；通过/违规各最多 {per_class} 条。\n"
            "历史先例只用于尺度校准，当前规则和当前工具事实优先。\n"
            f"对比分析：{json.dumps(analysis, ensure_ascii=False)}\n"
            f"历史证据卡：{json.dumps(compact_cases, ensure_ascii=False)}"
        )
        return {
            "text": text,
            "history_cases": compact_cases,
            "analysis": analysis,
            "analyst_raw_analysis": analyst_raw_analysis,
            "history_snapshot": str(history_path.resolve()),
            "history_snapshot_mtime_ns": history_path.stat().st_mtime_ns,
            "analyst_usage": getattr(response, "usage", {}) or {},
            "analyst_image_count": len(analyst_images),
            "analyst_image_max_tokens": image_max_tokens,
            "history_context_mode": context_mode,
            "display_images": [],
        }

    def render(self, result: Dict[str, Any]) -> ToolRenderResult:
        return ToolRenderResult(text=str((result or {}).get("text") or ""), images=[])
