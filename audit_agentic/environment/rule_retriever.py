"""Rule retriever interface + implementations.

V1 / mock retrievers stay here for tests / examples that don't have access
to the production rule store. The Zeus-backed real retriever
(`ZeusRuleRetriever`) loads from `agentic/rules/dtm=YYYYMMDD/` and is what
the demo / training pipeline uses by default.

Design note (V1 simplification, per user spec):
Rule retrieval is an ENVIRONMENT QUERY, not an LLM tool. After Main Agent's
router stage gives shortlist_labels, the workflow code calls
`rule_retriever.retrieve(source, labels)` directly and feeds the full
rule_text straight into the Planner Agent's prompt. The LLM never invokes
this through function-calling.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas import AuditInput, RuleInfo


class BaseRuleRetriever:
    def retrieve(self, source_id: str, labels: List[str]) -> List[RuleInfo]:
        raise NotImplementedError


# Sentinel label for the queue-level 审核须知 / 总规则 RuleInfo. Retrievers that
# have a queue-level notice (e.g. OnlineRuleRetriever) inject it as a synthetic
# rule at position 0 with this label. Downstream filters (e.g. remaining_rules
# in main_final) must special-case this label to always keep the notice.
QUEUE_NOTICE_LABEL = "__queue_notice__"


def _strip_label_id_for_lookup(label: str) -> str:
    label = str(label or "")
    return label.split("|", 1)[1] if "|" in label else label


def _is_rule_miss(rule: RuleInfo) -> bool:
    raw = getattr(rule, "raw", None)
    return isinstance(raw, dict) and bool(raw.get("miss"))


def _raw_label_by_name(audit_input: AuditInput) -> Dict[str, str]:
    """Map model-facing Chinese labels back to raw id-prefixed labels."""
    mapping: Dict[str, str] = {}
    for raw in audit_input.raw_candidate_labels or []:
        raw_s = str(raw or "").strip()
        if not raw_s:
            continue
        mapping.setdefault(_strip_label_id_for_lookup(raw_s), raw_s)
    return mapping


def retrieve_rules_with_id_fallback(
    retriever: BaseRuleRetriever,
    audit_input: AuditInput,
    labels: List[str],
) -> List[RuleInfo]:
    """Retrieve by Chinese labels first, then retry misses with raw tag ids.

    Model-facing prompts should only contain stripped Chinese labels. Rule
    lookup still needs to handle queues whose Jupiter data is easiest to match
    by tagId, so this retries missed Chinese labels with the raw
    ``id|label`` value from the input. Returned RuleInfo.label values are kept
    as the Chinese labels to prevent ids leaking back into prompts.
    """
    labels = list(dict.fromkeys(str(label).strip() for label in labels if str(label).strip()))
    rules = retriever.retrieve(audit_input.source_id, labels)
    raw_by_name = _raw_label_by_name(audit_input)
    if not raw_by_name:
        return rules

    fixed: List[RuleInfo] = []
    for rule in rules:
        label = rule.label
        if label == QUEUE_NOTICE_LABEL or not _is_rule_miss(rule):
            fixed.append(rule)
            continue

        raw_label = raw_by_name.get(label)
        if not raw_label or raw_label == label:
            fixed.append(rule)
            continue

        fallback_rules = retriever.retrieve(audit_input.source_id, [raw_label])
        replacement = None
        for candidate in fallback_rules:
            if candidate.label == QUEUE_NOTICE_LABEL:
                continue
            if not _is_rule_miss(candidate):
                candidate.label = label
                replacement = candidate
                break
        fixed.append(replacement or rule)
    return fixed


def _is_pass_label(label: str) -> bool:
    return label.strip().lower() in {"pass", "通过"} or "通过" in label


def _append_source_specific_queue_notice(source_id: str, queue_rule_text: str) -> str:
    """Public snapshot: all business rules must come from the supplied cache."""
    return queue_rule_text


def _rule_images_disabled() -> bool:
    """Whether to strip rule images before sending prompts to LLM services.

    Rule images currently make several new-source planner requests fail with
    SGLang http_500. Keep this enabled by default for eval/training stability.
    Set AUDIT_ENABLE_RULE_IMAGES=1 only after the rule-image path is verified
    end to end.
    """
    return os.environ.get("AUDIT_ENABLE_RULE_IMAGES", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _strip_rule_image_tags(rule_text: str) -> str:
    """Remove rule-image placeholders and image-only lines from cached rules."""
    text = str(rule_text or "")
    if "<image>" not in text:
        return text
    cleaned_lines = []
    for line in text.splitlines():
        if "<image>" in line:
            line = line.replace("<image>", "").rstrip()
            if not line.strip():
                continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _maybe_strip_rule_images(rule_text: str, rule_images: Optional[List[str]] = None) -> tuple[str, List[str]]:
    if not _rule_images_disabled():
        return str(rule_text or ""), list(rule_images or [])
    return _strip_rule_image_tags(rule_text), []


# ─────────────────────────────────────────────────────────────────────────────
# Mock (offline tests / examples)
# ─────────────────────────────────────────────────────────────────────────────


class MockRuleRetriever(BaseRuleRetriever):
    """File-backed mock retriever.

    Mock JSON shape::

        {"<source_id>": {"<label_name>": {"positive_conditions": [...],
                                          "exemption_conditions": [...], ...}}}

    Falls back to a generic empty RuleInfo so the workflow keeps running.
    """

    def __init__(self, mock_data_path: Optional[str | Path] = None):
        self._data: Dict[str, Any] = {}
        if mock_data_path:
            p = Path(mock_data_path)
            if p.exists():
                self._data = json.loads(p.read_text(encoding="utf-8"))

    def retrieve(self, source_id: str, labels: List[str]) -> List[RuleInfo]:
        source_rules = self._data.get(source_id, {})
        out: List[RuleInfo] = []
        for label in labels:
            if _is_pass_label(label):
                out.append(
                    RuleInfo(
                        label=label,
                        rule_text='（"通过" 标签无对应违规规则。当所有违规标签都未命中时输出该标签。）',
                        raw={"note": "pass label has no rule"},
                    )
                )
                continue
            r = source_rules.get(label, {})
            txt = r.get("rule_text") or ""
            if not txt:
                # synthesize a compact text from structured fields if any
                parts = [f"## {label}"]
                pos = r.get("positive_conditions") or []
                exe = r.get("exemption_conditions") or []
                if pos:
                    parts.append("\n**违规判定条件**")
                    parts.extend(f"- {p}" for p in pos)
                if exe:
                    parts.append("\n**豁免条件**")
                    parts.extend(f"- {e}" for e in exe)
                txt = "\n".join(parts) if pos or exe else f"## {label}\n（mock：无规则文本）"
            out.append(
                RuleInfo(
                    label=label,
                    rule_text=txt,
                    positive_conditions=list(r.get("positive_conditions", [])),
                    exemption_conditions=list(r.get("exemption_conditions", [])),
                    examples=list(r.get("examples", [])),
                    hard_negatives=list(r.get("hard_negatives", [])),
                    required_tools=list(r.get("required_tools", [])),
                    raw=r,
                )
            )
        return out


def default_mock_rules() -> Dict[str, Any]:
    """Synthetic rules only, unrelated to any production moderation policy."""
    return {
        "synthetic": {
            "DEMO_UNVERIFIED": {
                "positive_conditions": ["Synthetic evidence has verified=false."],
                "exemption_conditions": ["Synthetic evidence has verified=true."],
                "required_tools": ["lookup_synthetic_evidence"],
            },
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Zeus-backed real retriever (production)
# ─────────────────────────────────────────────────────────────────────────────


_DEFAULT_RULES_BASE = os.environ.get("AUDIT_RULES_BASE", "local_artifacts/rules")
_DEFAULT_ZEUS_SERVER_DIR = os.environ.get("AUDIT_RULE_ADAPTER_PATH")


class ZeusRuleRetriever(BaseRuleRetriever):
    """Real retriever backed by zeus-agentic-rl/server/rule_tools.py.

    * Lazy-loads ``tag_details_cleaned.jsonl`` + ``mars2risk.json`` per dtm,
      caches in-memory.
    * ``retrieve(source, labels)`` returns RuleInfo per label with the full
      rule_text (definition + sections + violations + exemptions + global
      exemption), markdown formatted.

    Args:
        rules_base: dir holding ``dtm=YYYYMMDD/`` subdirs.
        default_dtm: dtm to use when caller doesn't pass one. Resolves to the
            latest available subdir if "latest" or missing.
        zeus_server_dir: path to zeus's ``server/`` so we can import
            rule_tools. Set to None if rule_tools is already on PYTHONPATH.
    """

    _lock = threading.Lock()

    def __init__(
        self,
        rules_base: str = _DEFAULT_RULES_BASE,
        default_dtm: str = "latest",
        zeus_server_dir: Optional[str] = _DEFAULT_ZEUS_SERVER_DIR,
    ):
        self.rules_base = rules_base
        self.default_dtm = default_dtm
        if zeus_server_dir and zeus_server_dir not in sys.path:
            sys.path.insert(0, zeus_server_dir)
        # cache: dtm -> (index, mars2risk_dict)
        self._cache: Dict[str, tuple] = {}

    # ── public API ────────────────────────────────────────────────────────

    def retrieve(
        self,
        source_id: str,
        labels: List[str],
        dtm: Optional[str] = None,
    ) -> List[RuleInfo]:
        index, mars2risk = self._get_for_dtm(dtm or self.default_dtm)
        risk_sources = mars2risk.get(source_id, []) or [source_id]
        out: List[RuleInfo] = []
        for label in labels:
            if _is_pass_label(label):
                out.append(
                    RuleInfo(
                        label=label,
                        rule_text=(
                            "（\"通过\" 标签无对应违规规则。当所有违规标签都未命中时输出该标签。）"
                        ),
                        raw={"note": "pass label has no rule"},
                    )
                )
                continue
            tree = self._find_tree(index, risk_sources, source_id, label)
            if tree is None:
                out.append(
                    RuleInfo(
                        label=label,
                        rule_text=(
                            f"## {label}\n（未在 source={source_id} 下找到对应规则，"
                            f"已尝试 mars2risk 映射: {risk_sources}）"
                        ),
                        raw={"miss": True, "source": source_id, "risk_sources": risk_sources},
                    )
                )
                continue
            out.append(self._tree_to_rule_info(tree))
        return out

    # ── internals ─────────────────────────────────────────────────────────

    def _resolve_dtm(self, dtm: str) -> str:
        if dtm and dtm != "latest":
            return dtm
        try:
            dirs = [d for d in os.listdir(self.rules_base) if d.startswith("dtm=")]
            if not dirs:
                return dtm or ""
            return sorted(dirs)[-1].replace("dtm=", "")
        except FileNotFoundError:
            return dtm or ""

    def _get_for_dtm(self, dtm: str):
        dtm = self._resolve_dtm(dtm)
        if dtm in self._cache:
            return self._cache[dtm]
        with self._lock:
            if dtm in self._cache:  # double-check
                return self._cache[dtm]

            # ── disk pickle cache: each subprocess pays ~1s instead of ~30s ──
            # Cache lives on NFS so multiple nodes share; atomic rename avoids
            # corruption when several pods race on first startup.
            import pickle
            from pathlib import Path as _Path

            _cache_dir = _Path(self.rules_base).parent / ".rule_pickle_cache"
            _cache_file = _cache_dir / f"rule_index_{dtm}.pkl"
            try:
                if _cache_file.exists():
                    payload = pickle.loads(_cache_file.read_bytes())
                    self._cache[dtm] = payload
                    return payload
            except Exception:
                pass  # corrupted or incompatible — fall through to rebuild

            from rule_tools import (  # type: ignore  (lazy import after sys.path push)
                load_trees_from_tag_details,
                build_tree_index,
            )

            dtm_dir = os.path.join(self.rules_base, f"dtm={dtm}")
            rules_path = os.path.join(dtm_dir, "tag_details_cleaned.jsonl")
            mars_path = os.path.join(dtm_dir, "mars2risk.json")
            trees = load_trees_from_tag_details(rules_path)
            index = build_tree_index(trees)
            mars2risk: Dict[str, List[str]] = {}
            if os.path.exists(mars_path):
                raw = json.loads(open(mars_path, encoding="utf-8").read())
                mars2risk = raw.get("mars2risk", raw) if isinstance(raw, dict) else {}

            payload = (index, mars2risk)
            self._cache[dtm] = payload

            # Write pickle atomically (tmp → rename) to avoid torn reads.
            try:
                _cache_dir.mkdir(parents=True, exist_ok=True)
                _tmp = _cache_file.with_suffix(".pkl.tmp")
                _tmp.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
                _tmp.rename(_cache_file)
            except Exception:
                pass  # non-fatal — next invocation will rebuild and retry

            return index, mars2risk

    def _find_tree(self, index, risk_sources, fallback_source, label):
        for rs in list(risk_sources) + [fallback_source]:
            if not rs:
                continue
            for tid in index.get_source_tags(rs):
                t = index.get_tree(rs, tid)
                if t and t.tag_name == label:
                    return t
        return None

    def _tree_to_rule_info(self, tree) -> RuleInfo:
        text = self._format_full_rule(tree)
        return RuleInfo(
            label=tree.tag_name,
            rule_text=text,
            raw={
                "source": tree.source,
                "tag_id": tree.tag_id,
                "section_count": tree.section_count,
                "audit_premise": tree.audit_premise,
                "whole_exemption": tree.whole_exemption,
            },
        )

    @staticmethod
    def _format_full_rule(tree) -> str:
        out = [f"## {tree.tag_name}"]
        if tree.audit_premise and tree.audit_premise.strip() != "暂无":
            out.append(f"\n**定义**: {tree.audit_premise.strip()}\n")
        for sec in tree.sections:
            path_disp = sec.path.replace("/", " > ") if sec.path else "(通用)"
            out.append(f"\n### {path_disp}")
            if sec.desc:
                out.append(sec.desc)
            for v in sec.violations:
                out.append(v)
            for e in sec.exemptions:
                out.append(f"豁免: {e}")
        if tree.whole_exemption and tree.whole_exemption.strip():
            out.append(f"\n**全局豁免**: {tree.whole_exemption.strip()}")
        return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# RulePoolRetriever — 临时方案，用于快速测试
# 正式生产应使用 ZeusRuleRetriever（四级漏斗规则库）
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_RULE_POOL_PATH = (
    Path(__file__).resolve().parent / "rule_pool.json"
)


class RulePoolRetriever(BaseRuleRetriever):
    """Temporary retriever backed by ``environment/rule_pool.json``.

    Covers all 12 sources in the current training/eval data.
    Pre-extracted from ``zeroshot_requests_v4.rules.jsonl`` — one entry per
    (source, label) pair with the full rule text in markdown.

    Usage::

        retriever = RulePoolRetriever()
        rules = retriever.retrieve(source_id, shortlist_labels)

    Switch back to production retriever::

        retriever = ZeusRuleRetriever()

    The ``rule_pool.json`` shape::

        {
          "<source>": {
            "<label>": "<rule_text markdown>",
            ...
          }
        }

    .. note::
        This retriever is intentionally simple and does NOT use the four-level
        funnel (L0/L1/L2/L3) of ZeusRuleRetriever. It exists only to unblock
        testing while the zeus rule index is unavailable for some sources.
    """

    def __init__(self, pool_path: Optional[str | Path] = None):
        self.pool_path = Path(pool_path) if pool_path else _DEFAULT_RULE_POOL_PATH
        self._pool: Optional[Dict[str, Any]] = None

    def _ensure_loaded(self) -> Dict[str, Any]:
        if self._pool is None:
            if not self.pool_path.exists():
                raise FileNotFoundError(
                    f"rule_pool.json not found at {self.pool_path}. "
                    "Run audit_agentic/data/_build_rule_pool.py to regenerate."
                )
            self._pool = json.loads(self.pool_path.read_text(encoding="utf-8"))
        return self._pool

    def retrieve(self, source_id: str, labels: List[str]) -> List[RuleInfo]:
        pool = self._ensure_loaded()
        source_rules = pool.get(source_id, {})
        out: List[RuleInfo] = []
        for label in labels:
            if _is_pass_label(label):
                out.append(
                    RuleInfo(
                        label=label,
                        rule_text='（"通过" 标签无对应违规规则。当所有违规标签都未命中时输出该标签。）',
                        raw={"note": "pass label has no rule"},
                    )
                )
                continue
            rule_text = source_rules.get(label, "")
            if not rule_text:
                rule_text = (
                    f"（规则池中未找到 source={source_id} label={label} 的规则。"
                    f"可用标签：{list(source_rules.keys())[:5]}）"
                )
            out.append(
                RuleInfo(
                    label=label,
                    rule_text=f"## {label}\n{rule_text}",
                    raw={"source": source_id, "from_pool": True},
                )
            )
        return out

# ─────────────────────────────────────────────────────────────────────────────
# OnlineRuleRetriever — 实时调 Jupiter API 获取规则
# ─────────────────────────────────────────────────────────────────────────────


class OnlineRuleRetriever(BaseRuleRetriever):
    """在线规则检索：直接调 Mars Jupiter API 获取队列规则。

    适用于无法使用本地规则文件（ZeusRuleRetriever）或静态规则池
    （RulePoolRetriever）的场景。

    Usage::

        retriever = OnlineRuleRetriever()
        rules = retriever.retrieve("example_queue_004", ["原创-整体搬运"])

    首次调用时会拉取并缓存该队列的所有规则，后续查询直接从缓存取。
    默认同时写入共享磁盘 cache，避免 Relax agentic rollout 中每个
    per-session app 进程都重复请求 Jupiter。
    """

    def __init__(self, category_type: str = "note"):
        from .data_access_mode import online_rule_access_enabled

        self._live_online_mode = online_rule_access_enabled()
        self._category_type = category_type
        self._cache: Dict[str, Dict[str, Any]] = {}
        local_rule_path = ""
        if not self._live_online_mode:
            local_rule_path = (
                os.environ.get("AUDIT_RULE_CACHE_PATH")
                or os.environ.get("AUDIT_LOCAL_RULE_CACHE_PATH")
                or ""
            )
        self._local_rule_cache_path = self._resolve_local_rule_cache_path(local_rule_path)
        self._local_rule_cache = self._load_local_rule_cache(self._local_rule_cache_path)
        default_cache = Path(__file__).resolve().parent / "cache" / "online_rule_cache.json"
        cache_path = os.environ.get("AUDIT_ONLINE_RULE_CACHE_PATH") or str(default_cache)
        self._disk_cache_path = Path(cache_path)
        self._disk_cache_disabled = self._live_online_mode or os.environ.get(
            "AUDIT_ONLINE_RULE_CACHE_DISABLE", ""
        ).lower() in {
            "1",
            "true",
            "yes",
        }
        self._disk_cache_lock = threading.Lock()

    @staticmethod
    def _resolve_local_rule_cache_path(raw_path: str) -> Optional[Path]:
        """Resolve a generated local rule cache path.

        ``AUDIT_RULE_CACHE_PATH`` may point either to ``rules_cache.jsonl`` or
        to the cache directory containing that file.
        """
        raw_path = str(raw_path or "").strip()
        if not raw_path:
            return None
        path = Path(raw_path)
        if path.is_dir():
            path = path / "rules_cache.jsonl"
        return path if path.exists() else None

    @staticmethod
    def _load_local_rule_cache(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
        if path is None:
            return {}

        cache: Dict[str, Dict[str, Any]] = {}
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    source = str(rec.get("source") or "").strip()
                    label = str(rec.get("label") or "").strip()
                    if not source or not label:
                        continue
                    bucket = cache.setdefault(source, {"queue": None, "rules": {}})
                    if label == QUEUE_NOTICE_LABEL:
                        bucket["queue"] = rec
                        continue
                    keys = {label}
                    raw_label = str(rec.get("raw_label") or "").strip()
                    if raw_label:
                        keys.add(raw_label)
                        keys.add(_strip_label_id_for_lookup(raw_label))
                        if "|" in raw_label:
                            raw_id = raw_label.split("|", 1)[0].strip()
                            if raw_id:
                                keys.add(raw_id)
                    raw = rec.get("raw") or {}
                    if isinstance(raw, dict):
                        tag_id = str(raw.get("tagId") or "").strip()
                        if tag_id:
                            keys.add(tag_id)
                    for key in keys:
                        if key:
                            bucket["rules"].setdefault(key, rec)
        except Exception:
            return {}
        return cache

    @staticmethod
    def _local_record_to_rule(rec: Dict[str, Any], requested_label: Optional[str] = None) -> RuleInfo:
        raw = dict(rec.get("raw") or {})
        raw["from_local_rule_cache"] = True
        if rec.get("raw_label"):
            raw.setdefault("raw_label", rec.get("raw_label"))
        rule_text, rule_images = _maybe_strip_rule_images(
            str(rec.get("rule_text") or ""),
            list(rec.get("rule_images") or []),
        )
        return RuleInfo(
            label=requested_label or str(rec.get("label") or ""),
            rule_text=rule_text,
            rule_images=rule_images,
            raw=raw,
        )

    def _retrieve_from_local_cache(self, source_id: str, labels: List[str]) -> tuple[List[RuleInfo], List[str]]:
        bucket = self._local_rule_cache.get(source_id)
        if not bucket:
            return [], list(labels)

        out: List[RuleInfo] = []
        queue_rec = bucket.get("queue")
        if isinstance(queue_rec, dict) and queue_rec.get("rule_text"):
            out.append(self._local_record_to_rule(queue_rec, QUEUE_NOTICE_LABEL))

        missing: List[str] = []
        rules_map = bucket.get("rules") or {}
        for label in labels:
            label_s = str(label or "").strip()
            rec = rules_map.get(label_s) or rules_map.get(_strip_label_id_for_lookup(label_s))
            if rec is None and "|" in label_s:
                raw_id = label_s.split("|", 1)[0].strip()
                rec = rules_map.get(raw_id)
            if rec is None:
                missing.append(label)
                continue
            out.append(self._local_record_to_rule(rec, _strip_label_id_for_lookup(label_s)))
        return out, missing

    def _queue_cache_key(self, source_id: str) -> str:
        return f"queue::{self._category_type}::{source_id}"

    def _tag_cache_key(self, source_id: str, tag_id: str) -> str:
        return f"tag::{self._category_type}::{source_id}::{tag_id}"

    def _load_disk_cache_unlocked(self) -> Dict[str, Any]:
        if self._disk_cache_disabled or not self._disk_cache_path.exists():
            return {}
        try:
            data = json.loads(self._disk_cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _get_disk_cache(self, key: str) -> Any:
        if self._disk_cache_disabled:
            return None
        with self._disk_cache_lock:
            return self._load_disk_cache_unlocked().get(key)

    def _acquire_disk_write_lock(self) -> tuple[int, Path] | None:
        lock_path = self._disk_cache_path.with_suffix(self._disk_cache_path.suffix + ".lock")
        deadline = time.time() + float(os.environ.get("AUDIT_ONLINE_RULE_CACHE_LOCK_TIMEOUT", "30"))
        while time.time() < deadline:
            try:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("utf-8"))
                return fd, lock_path
            except FileExistsError:
                try:
                    if time.time() - lock_path.stat().st_mtime > 120:
                        lock_path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                except Exception:
                    pass
                time.sleep(0.05)
        return None

    def _set_disk_cache(self, key: str, value: Any) -> None:
        if self._disk_cache_disabled:
            return
        with self._disk_cache_lock:
            acquired = self._acquire_disk_write_lock()
            if acquired is None:
                return
            fd, lock_path = acquired
            try:
                cache = self._load_disk_cache_unlocked()
                cache[key] = value
                self._disk_cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = self._disk_cache_path.with_suffix(
                    self._disk_cache_path.suffix + f".{os.getpid()}.tmp"
                )
                tmp_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                tmp_path.replace(self._disk_cache_path)
            except Exception:
                pass
            finally:
                try:
                    os.close(fd)
                except Exception:
                    pass
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

    def _fetch_queue_info(self, source_id: str) -> Dict[str, Any]:
        from .services.rule_query import fetch_queue_rules

        cache_key = self._queue_cache_key(source_id)
        cached = self._get_disk_cache(cache_key)
        if isinstance(cached, dict):
            return cached

        queue_info = fetch_queue_rules(source_id, self._category_type)
        self._set_disk_cache(cache_key, queue_info)
        return queue_info

    def _fetch_tag_rules(self, source_id: str, tag_id: str) -> Dict[str, Any]:
        from .services.rule_query import fetch_rules_by_tag_ids

        cache_key = self._tag_cache_key(source_id, tag_id)
        cached = self._get_disk_cache(cache_key)
        if isinstance(cached, dict):
            return cached

        tag_rules = fetch_rules_by_tag_ids(
            [tag_id],
            source_type=source_id,
            category_type=self._category_type,
        )
        if tag_rules:
            self._set_disk_cache(cache_key, tag_rules)
        return tag_rules

    def retrieve(self, source_id: str, labels: List[str]) -> List[RuleInfo]:
        from .services.rule_query import (
            format_queue_notice_images,
            format_queue_notice_markdown,
            format_rule_images,
            format_rule_markdown,
        )

        labels = list(dict.fromkeys(str(label).strip() for label in labels if str(label).strip()))
        if self._local_rule_cache:
            local_rules, missing_labels = self._retrieve_from_local_cache(source_id, labels)
            if local_rules and not missing_labels:
                return local_rules
            if not self._live_online_mode:
                return local_rules + [
                    self._local_cache_miss_rule(source_id, label)
                    for label in missing_labels
                ]
            if local_rules and missing_labels:
                saved_local_rule_cache = self._local_rule_cache
                self._local_rule_cache = {}
                try:
                    online_rules = self.retrieve(source_id, missing_labels)
                    online_rules = [r for r in online_rules if r.label != QUEUE_NOTICE_LABEL]
                finally:
                    self._local_rule_cache = saved_local_rule_cache
                return local_rules + online_rules

        if not self._live_online_mode:
            return [
                self._local_cache_miss_rule(source_id, label)
                for label in labels
            ]

        # 缓存队列信息
        if source_id not in self._cache:
            self._cache[source_id] = self._fetch_queue_info(source_id)

        queue_info = self._cache[source_id]
        rules_map = queue_info.get("规则", {})
        queue_rule_text = format_queue_notice_markdown(queue_info)
        queue_rule_text = _append_source_specific_queue_notice(source_id, queue_rule_text)
        queue_rule_images = format_queue_notice_images(queue_info)
        queue_rule_text, queue_rule_images = _maybe_strip_rule_images(queue_rule_text, queue_rule_images)

        out: List[RuleInfo] = []
        # 队列总规则 / 审核须知作为特殊的第 0 条注入一次，避免每条规则都 prepend 造成
        # prompt 长度爆炸（历史上出现过 sglang 400 Bad Request）。下游 filter 时通过
        # label == QUEUE_NOTICE_LABEL 特殊保留。
        if queue_rule_text:
            out.append(
                RuleInfo(
                    label=QUEUE_NOTICE_LABEL,
                    rule_text=queue_rule_text,
                    rule_images=list(queue_rule_images),
                    raw={"queue_notice": True, "source": source_id},
                )
            )
        for label in labels:
            if _is_pass_label(label):
                out.append(
                    RuleInfo(
                        label=label,
                        rule_text='（"通过" 标签无对应违规规则。当所有违规标签都未命中时输出该标签。）',
                        rule_images=[],
                        raw={"note": "pass label has no rule"},
                    )
                )
                continue

            rule, matched_key, tried_keys = self._lookup_rule(rules_map, label)
            if rule is None:
                tag_id = self._extract_label_id(label)
                if tag_id:
                    tag_rules = self._fetch_tag_rules(source_id, tag_id)
                    if tag_rules:
                        rules_map.update(tag_rules)
                        rule, matched_key, tried_keys = self._lookup_rule(rules_map, label)
            if rule is None:
                available = list(rules_map.keys())[:8]
                out.append(
                    RuleInfo(
                        label=label,
                        rule_text=(
                            f"## {label}\n（在线规则库中未找到 source={source_id} "
                            f"label={label} 的规则。已尝试：{tried_keys}。"
                            f"可用标签：{available}）"
                        ),
                        rule_images=[],
                        raw={
                            "miss": True,
                            "source": source_id,
                            "tried_keys": tried_keys,
                        },
                    )
                )
                continue

            rule_text = format_rule_markdown(rule)
            rule_images = format_rule_images(rule)
            rule_text, rule_images = _maybe_strip_rule_images(rule_text, rule_images)
            out.append(
                RuleInfo(
                    label=label,
                    rule_text=rule_text,
                    rule_images=rule_images,
                    raw={
                        "source": source_id,
                        "from_online": True,
                        "matched_key": matched_key,
                        "structureId": rule.get("structureId", ""),
                        "tagId": rule.get("tagId", ""),
                    },
                )
            )
        return out

    @staticmethod
    def _local_cache_miss_rule(source_id: str, label: str) -> RuleInfo:
        clean_label = _strip_label_id_for_lookup(label)
        if _is_pass_label(clean_label):
            return RuleInfo(
                label=clean_label,
                rule_text=(
                    "（\"通过\" 标签无对应违规规则。"
                    "当所有违规标签都未命中时输出该标签。）"
                ),
                raw={"note": "pass label has no rule"},
            )
        return RuleInfo(
            label=clean_label,
            rule_text=(
                f"（本地规则 cache 中未找到 source={source_id} "
                f"label={clean_label} 的规则；本次未回退在线 Jupiter。）"
            ),
            raw={
                "source": source_id,
                "miss": True,
                "from_local_rule_cache": True,
            },
        )

    @staticmethod
    def _lookup_rule(rules_map: Dict[str, Any], label: str):
        """Find a rule by full label, Chinese label, or tagId.

        Candidate labels in eval data are often shaped like ``50007792|AI作品``,
        while Jupiter returns a map keyed by Chinese tag name and stores the
        numeric id in each rule's ``tagId`` field. Support all three shapes so
        callers can pass labels without pre-normalising them.
        """
        label = str(label or "").strip()
        tried_keys: List[str] = []

        def _try(key: str):
            key = str(key or "").strip()
            if not key:
                return None, ""
            if key not in tried_keys:
                tried_keys.append(key)
            rule = rules_map.get(key)
            return (rule, key) if rule is not None else (None, "")

        rule, key = _try(label)
        if rule is not None:
            return rule, key, tried_keys

        label_id = ""
        label_name = ""
        if "|" in label:
            label_id, label_name = label.split("|", 1)
            label_id = label_id.strip()
            label_name = label_name.strip()
            rule, key = _try(label_name)
            if rule is not None:
                return rule, key, tried_keys
        elif label.isdigit():
            label_id = label

        if label_id:
            if label_id not in tried_keys:
                tried_keys.append(label_id)
            for name, candidate in rules_map.items():
                if str(candidate.get("tagId", "")).strip() == label_id:
                    return candidate, name, tried_keys

        return None, "", tried_keys

    @staticmethod
    def _extract_label_id(label: str) -> str:
        """Extract numeric tag id from ``id|label`` or raw id strings."""
        label = str(label or "").strip()
        if "|" in label:
            label = label.split("|", 1)[0].strip()
        return label if label.isdigit() else ""

    def get_queue_info(self, source_id: str) -> Dict[str, Any]:
        """获取完整的队列信息（含审核须知、SOP、所有规则）。"""
        if source_id not in self._cache:
            self._cache[source_id] = self._fetch_queue_info(source_id)
        return self._cache[source_id]
