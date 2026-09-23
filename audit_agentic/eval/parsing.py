"""Label parsing utilities.

Ported from sft/data/data_clean/eval_top5_label_select_api.py so the
agentic workflow uses the exact same normalisation + fuzzy-match policy as
the reference SFT eval pipeline:

  1. Extract last <answer>...</answer> if present, else use the whole response.
  2. Split by ``,，、/|;；\\n``.
  3. Per token:
     * exact match against candidate set → keep
     * NFKC + strip whitespace + collapse hyphens → exact match
     * highest-overlap candidate (token-set intersection) if positive
     * else keep normalised form, mark as non-candidate
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, List, Tuple

PASS_LABEL = "通过"
ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|<\\answer>)", re.DOTALL)
SPLIT_RE = re.compile(r"[,，、/|;；\n]+")


def parse_label_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            return parse_label_list(json.loads(text))
        except Exception:
            return [x.strip() for x in SPLIT_RE.split(text) if x.strip()]
    text = str(value).strip()
    return [text] if text else []


def normalize_label_key(label: str) -> str:
    text = unicodedata.normalize("NFKC", str(label)).strip()
    text = re.sub(r"\s*[-‐‑‒–—―－]\s*", "-", text)
    text = re.sub(r"\s+", "", text)
    return text


def normalize_label_display(label: str) -> str:
    return normalize_label_key(label)


def overlap_tokens(label: str) -> set:
    text = normalize_label_key(label).lower()
    tokens: set = set()
    ascii_buf: list = []
    for ch in text:
        if ch == "-":
            if ascii_buf:
                tokens.add("".join(ascii_buf))
                ascii_buf = []
            continue
        if ch.isascii() and ch.isalnum():
            ascii_buf.append(ch)
        else:
            if ascii_buf:
                tokens.add("".join(ascii_buf))
                ascii_buf = []
            tokens.add(ch)
    if ascii_buf:
        tokens.add("".join(ascii_buf))
    return tokens


def overlap_score(raw_label: str, candidate_label: str) -> int:
    return len(overlap_tokens(raw_label) & overlap_tokens(candidate_label))


def best_overlap_candidate(raw_label: str, candidate_labels: List[str]) -> Tuple[str | None, int]:
    best_label: str | None = None
    best_score = 0
    for candidate_label in candidate_labels:
        score = overlap_score(raw_label, candidate_label)
        if score > best_score:
            best_label = candidate_label
            best_score = score
    if best_score <= 0:
        return None, 0
    return best_label, best_score


def parse_answer_labels(
    response: str, candidate_labels: List[str]
) -> Tuple[List[str], str, str, List[str], List[dict]]:
    """Returns (canonical_labels, answer_text, source, non_candidate, fuzzy_mappings).

    Step 0 (added): scan for literal candidate-label occurrences first (longest
    first), so candidates containing separator characters such as ``/`` (e.g.
    ``搬运-站外ui/录屏截图``) are NOT shattered by SPLIT_RE. Remaining text
    fragments still go through the original split + normalize + fuzzy path.
    """
    matches = ANSWER_RE.findall(response or "")
    answer_text = matches[-1].strip() if matches else ""
    source = "answer_tag" if matches else "fallback_text"
    text = answer_text or (response or "").strip()
    candidate_set = set(candidate_labels)
    normalized_candidate_map = {normalize_label_key(l): l for l in candidate_labels}
    labels: List[str] = []
    non_candidate_labels: List[str] = []
    fuzzy_mappings: List[dict] = []
    seen: set = set()

    def _add(canonical: str) -> None:
        if canonical and canonical not in seen:
            labels.append(canonical)
            seen.add(canonical)

    # Step 0: greedy literal-candidate protection (longest-first).
    # Build a split pattern with each candidate captured; even indices are the
    # between-text fragments, odd indices are exact candidate matches.
    if candidate_labels:
        ordered = sorted(set(candidate_labels), key=len, reverse=True)
        protect_re = re.compile("(" + "|".join(re.escape(c) for c in ordered) + ")")
        parts = protect_re.split(text)
    else:
        parts = [text]

    for i, part in enumerate(parts):
        if i % 2 == 1:
            # an exact candidate match (already canonical)
            _add(part)
            continue
        # leftover free text -> original split + normalize + fuzzy path
        for raw in SPLIT_RE.split(part):
            raw = raw.strip()
            if not raw or raw in seen:
                continue
            if raw in candidate_set:
                canonical = raw
            else:
                canonical = normalized_candidate_map.get(normalize_label_key(raw))
                if canonical is None and candidate_labels:
                    fuzzy_label, score = best_overlap_candidate(raw, candidate_labels)
                    if fuzzy_label is not None:
                        canonical = fuzzy_label
                        fuzzy_mappings.append(
                            {"raw": raw, "canonical": fuzzy_label, "overlap": score}
                        )
            if canonical is None:
                canonical = normalize_label_display(raw)
                if candidate_set:
                    non_candidate_labels.append(canonical)
            _add(canonical)

    return labels, answer_text, source, non_candidate_labels, fuzzy_mappings
