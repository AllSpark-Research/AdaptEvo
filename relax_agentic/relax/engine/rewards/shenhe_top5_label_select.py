# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward for shenhe top-5 candidate label selection.

Policy:
  score = base_reward - 0.1 * non_candidate_count - 0.2 * format_error

base_reward is computed after normalizing and fuzzy-mapping predictions to the
provided candidate labels. A sample is correct when any mapped prediction
overlaps any ground-truth active label.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any


PASS_LABEL = "通过"

_ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|<\\answer>)", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SPLIT_RE = re.compile(r"[,，、;；|]+")
_DASH_RE = re.compile(r"[‐‑‒–—―－﹣]")
_SPACE_AROUND_DASH_RE = re.compile(r"\s*-\s*")
_PUNCT_RE = re.compile(r"[\s,，、;；|/\\:：.。!！?？()（）\\[\\]【】{}<>《》\"'“”‘’`~·_-]+")
_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _parse_label_list(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = text
        raw_items = parsed if isinstance(parsed, list) else [parsed]
    else:
        return []

    labels: list[str] = []
    for item in raw_items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            labels.append(text)
    return labels


def _ground_truth_labels(sample) -> list[str]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    for key in ("active_labels", "target_labels", "grpo_target_labels"):
        labels = _parse_label_list(metadata.get(key))
        if labels:
            return labels
    labels = _parse_label_list(getattr(sample, "label", None))
    if labels:
        return labels
    ground_truth = metadata.get("ground_truth")
    return [str(ground_truth).strip()] if ground_truth else []


def _candidate_labels(sample) -> list[str]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    labels = _parse_label_list(metadata.get("candidate_labels"))
    if labels:
        return labels

    reward_spec = metadata.get("reward_spec")
    if isinstance(reward_spec, dict):
        extra_info = reward_spec.get("extra_info")
        if isinstance(extra_info, dict):
            labels = _parse_label_list(extra_info.get("candidate_labels"))
            if labels:
                return labels

    prompt = str(getattr(sample, "prompt", "") or "")
    match = re.search(r"## 候选标签\s*\n(.+?)(?:\n\n|$)", prompt, re.DOTALL)
    if match:
        return [x.strip() for x in re.split(r"[,，]", match.group(1)) if x.strip()]
    return []


def _normalize_for_exact(label: str) -> str:
    text = unicodedata.normalize("NFKC", str(label or "")).strip()
    text = _DASH_RE.sub("-", text)
    text = _SPACE_AROUND_DASH_RE.sub("-", text)
    return re.sub(r"\s+", "", text)


def _overlap_units(label: str) -> set[str]:
    text = _normalize_for_exact(label).lower()
    text = _PUNCT_RE.sub("", text)
    units: set[str] = set()
    occupied = [False] * len(text)
    for match in _ALNUM_RE.finditer(text):
        units.add(match.group(0))
        for idx in range(match.start(), match.end()):
            occupied[idx] = True
    for idx, ch in enumerate(text):
        if not occupied[idx] and ch:
            units.add(ch)
    return units


def _extract_answer(response: str) -> tuple[str, bool]:
    if not isinstance(response, str):
        return "", False
    text = response.strip()
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1).strip(), True

    # Still try to grade malformed outputs, but apply the format penalty.
    text = _THINK_RE.sub("", text).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text, False


def _split_predicted_labels(answer_text: str) -> list[str]:
    if not answer_text:
        return []
    text = answer_text.strip()
    if text.startswith("["):
        parsed = _parse_label_list(text)
        if parsed:
            return parsed
    return [part.strip() for part in _SPLIT_RE.split(text) if part.strip()]


def _map_predictions(raw_labels: list[str], candidates: list[str]) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    candidate_by_norm: dict[str, str] = {}
    for candidate in candidates:
        candidate_by_norm.setdefault(_normalize_for_exact(candidate), candidate)

    candidate_units = [(candidate, _overlap_units(candidate)) for candidate in candidates]
    mapped: list[str] = []
    fuzzy_mappings: list[dict[str, Any]] = []
    non_candidate_labels: list[str] = []

    for raw in raw_labels:
        norm = _normalize_for_exact(raw)
        exact = candidate_by_norm.get(norm)
        if exact is not None:
            mapped.append(exact)
            continue

        non_candidate_labels.append(raw)
        raw_units = _overlap_units(raw)
        best_label = None
        best_score = 0
        for candidate, units in candidate_units:
            score = len(raw_units & units)
            if score > best_score:
                best_score = score
                best_label = candidate
        if best_label is not None and best_score > 0:
            mapped.append(best_label)
            fuzzy_mappings.append({"raw": raw, "mapped": best_label, "overlap": best_score})
        else:
            mapped.append(raw)

    return mapped, fuzzy_mappings, non_candidate_labels


def _score_one(sample) -> dict[str, Any]:
    candidates = _candidate_labels(sample)
    gt = _ground_truth_labels(sample)
    answer_text, format_ok = _extract_answer(getattr(sample, "response", ""))
    raw_pred = _split_predicted_labels(answer_text)
    mapped_pred, fuzzy_mappings, non_candidate_labels = _map_predictions(raw_pred, candidates)

    gt_norm = {_normalize_for_exact(x) for x in gt}
    mapped_norm = {_normalize_for_exact(x) for x in mapped_pred}
    overlap_correct = bool(gt_norm and mapped_norm and (gt_norm & mapped_norm))
    base_reward = 1.0 if overlap_correct else 0.0

    non_candidate_penalty_value = abs(_env_float("SHENHE_TOP5_NON_CANDIDATE_PENALTY", 0.1))
    format_penalty_value = abs(_env_float("SHENHE_TOP5_FORMAT_PENALTY", 0.2))
    non_candidate_count = len(non_candidate_labels)
    format_error = 0 if format_ok and raw_pred else 1
    non_candidate_penalty = non_candidate_penalty_value * non_candidate_count
    format_penalty = format_penalty_value * format_error
    score = base_reward - non_candidate_penalty - format_penalty

    gt_is_pass = set(gt_norm) == {_normalize_for_exact(PASS_LABEL)}
    pred_has_pass = _normalize_for_exact(PASS_LABEL) in mapped_norm

    return {
        "score": score,
        "base_reward": base_reward,
        "overlap_correct": int(overlap_correct),
        "format_ok": int(format_error == 0),
        "format_error": format_error,
        "format_penalty": -format_penalty,
        "non_candidate_count": non_candidate_count,
        "non_candidate_penalty": -non_candidate_penalty,
        "raw_predicted_labels": raw_pred,
        "mapped_predicted_labels": mapped_pred,
        "ground_truth_labels": gt,
        "non_candidate_labels": non_candidate_labels,
        "fuzzy_label_mappings": fuzzy_mappings,
        "pred_label_count": len(raw_pred),
        "candidate_label_count": len(candidates),
        "gt_is_pass": int(gt_is_pass),
        "gt_is_violation": int(bool(gt_norm) and not gt_is_pass),
        "pred_has_pass": int(pred_has_pass),
    }


async def top5_label_select_reward(args, samples, **kwargs):
    if isinstance(samples, list):
        return [_score_one(sample) for sample in samples]
    return _score_one(samples)
