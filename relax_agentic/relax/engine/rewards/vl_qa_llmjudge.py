# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""LLM-judge reward for VL QA data used by multimodel_mixrl.

This follows the evaluation logic in multimodel_mixrl/eval_pass8.py:
the rollout is expected to put the final answer in the last ``\boxed{...}``,
then a judge model grades the boxed answer against the gold answer.

The judge returns one letter:
    A = correct
    B = incorrect

The reward maps A -> 1.0 and B -> 0.0 by default.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import aiohttp


DEFAULT_JUDGE_URL = "https://service.example.invalid"
DEFAULT_JUDGE_MODEL = "Qwen3.5-397B-A17B-FP8"

BOXED_INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."

GRADER_TEMPLATE = """You are an expert grader. Judge whether the candidate's answer matches the gold answer.

Rules:
- Ignore format differences: letter case, quotes, markdown, \\boxed{{}}, "answer:" prefix, LaTeX wrappers, trailing units, etc.
- Multiple-choice: the candidate is correct if it ultimately picks the same letter as the gold, regardless of surrounding reasoning.
- Open-ended: numerical answers match if numerically equal within reasonable rounding; textual answers match if semantically equivalent.
- Output ONLY one letter: A for CORRECT, B for INCORRECT. No other text, no explanation.

<Question>
{question}
</Question>

<Gold Answer>
{answer}
</Gold Answer>

<Candidate Answer>
{prediction}
</Candidate Answer>

Grade (A=CORRECT, B=INCORRECT):"""

_IMAGE_PLACEHOLDER_RE = re.compile(
    r"(<image>|<\|vision_start\|>\s*<\|image_pad\|>\s*<\|vision_end\|>)",
    re.IGNORECASE,
)
_CHAT_MESSAGE_RE = re.compile(
    r"<\|im_start\|>\s*([A-Za-z_]+)\s*\n(.*?)(?:<\|im_end\|>|$)",
    re.DOTALL,
)
_CHAT_TOKEN_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_session: aiohttp.ClientSession | None = None
_semaphore: asyncio.Semaphore | None = None


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return default if value in (None, "") else value


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return float(value)


def _judge_raw_max_chars() -> int:
    return _env_int(
        "MIXRL_JUDGE_RAW_MAX_CHARS",
        _env_int("VL_QA_JUDGE_RAW_MAX_CHARS", 4000),
    )


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif item_type in ("image", "image_url"):
            parts.append("<image>")
    return "".join(parts)


def _extract_user_content(prompt: Any) -> str:
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user":
                return _message_content_to_text(message.get("content"))
        chunks = [
            _message_content_to_text(message.get("content"))
            for message in prompt
            if isinstance(message, dict)
        ]
        return "\n".join(chunk for chunk in chunks if chunk)
    if isinstance(prompt, dict):
        return _message_content_to_text(prompt.get("content"))
    if isinstance(prompt, str):
        messages = [
            (role.strip().lower(), content.strip())
            for role, content in _CHAT_MESSAGE_RE.findall(prompt)
        ]
        for role, content in reversed(messages):
            if role == "user":
                return content
        if messages:
            return "\n".join(content for role, content in messages if role != "assistant")
        return prompt
    return "" if prompt is None else str(prompt)


def _clean_question(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace(BOXED_INSTR, "")
    text = _CHAT_TOKEN_RE.sub(" ", text)
    text = _IMAGE_PLACEHOLDER_RE.sub(" ", text)
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _label_to_answer(label: Any) -> str:
    if isinstance(label, list):
        return " / ".join(str(x).strip() for x in label if str(x).strip())
    if isinstance(label, str):
        stripped = label.strip()
        try:
            parsed = json.loads(stripped)
        except Exception:
            return stripped
        if isinstance(parsed, list):
            return " / ".join(str(x).strip() for x in parsed if str(x).strip())
        return stripped
    return "" if label is None else str(label).strip()


def _strip_thinking(text: Any) -> str:
    if text is None:
        return ""
    text = text if isinstance(text, str) else str(text)
    text = text.replace("<|im_end|>", "").strip()
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return _THINK_BLOCK_RE.sub("", text).strip()


def extract_boxed(text: Any) -> str:
    """Return the content of the last \\boxed{...}; fallback to tail text."""
    text = _strip_thinking(text)
    if not text:
        return ""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return text.strip()[-200:]

    i = idx + len("\\boxed{")
    depth = 1
    out: list[str] = []
    while i < len(text) and depth > 0:
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth > 0:
            out.append(char)
        i += 1
    return "".join(out).strip()


def _build_judge_prompt(sample) -> tuple[str, str, str, str]:
    question = _clean_question(_extract_user_content(getattr(sample, "prompt", "")))
    answer = _label_to_answer(getattr(sample, "label", None))
    prediction = extract_boxed(getattr(sample, "response", ""))
    prompt = GRADER_TEMPLATE.format(question=question, answer=answer, prediction=prediction)
    return prompt, question, answer, prediction


def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(
            total=_env_float(
                "VL_QA_JUDGE_TIMEOUT",
                _env_float("LLM_JUDGE_TIMEOUT", 120.0),
            )
        )
        connector = aiohttp.TCPConnector(
            limit=_env_int(
                "VL_QA_JUDGE_HTTP_LIMIT",
                _env_int("LLM_JUDGE_HTTP_LIMIT", 64),
            ),
            enable_cleanup_closed=True,
        )
        _session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    return _session


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(
            _env_int(
                "VL_QA_JUDGE_CONCURRENCY",
                _env_int("LLM_JUDGE_CONCURRENCY", 32),
            )
        )
    return _semaphore


def _parse_judge_letter(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    stripped = _strip_thinking(text).strip().upper()
    if not stripped:
        return None
    if stripped[-1:] in {"A", "B"}:
        return stripped[-1]
    matches = re.findall(r"\b[AB]\b", stripped)
    if matches:
        return matches[-1]
    compact = re.sub(r"[^AB]", "", stripped)
    if compact:
        return compact[-1]
    return None


async def _call_llm_judge(prompt: str) -> tuple[str | None, str, str]:
    api_key = (
        _env_str("VL_QA_JUDGE_API_KEY")
        or _env_str("LLM_JUDGE_API_KEY")
        or _env_str("QS_TOKEN")
    )
    if not api_key:
        return None, "", "missing_api_key"

    url = _env_str("VL_QA_JUDGE_URL") or _env_str("LLM_JUDGE_URL") or DEFAULT_JUDGE_URL
    model = _env_str("VL_QA_JUDGE_MODEL") or _env_str("LLM_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert grader. Output only A or B."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_tokens": _env_int(
            "VL_QA_JUDGE_MAX_TOKENS",
            _env_int("LLM_JUDGE_MAX_TOKENS", 256),
        ),
        "temperature": _env_float(
            "VL_QA_JUDGE_TEMPERATURE",
            _env_float("LLM_JUDGE_TEMPERATURE", 0.0),
        ),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json", "api-key": api_key}

    session = _get_session()
    max_retries = _env_int(
        "VL_QA_JUDGE_MAX_RETRIES",
        _env_int("LLM_JUDGE_MAX_RETRIES", 3),
    )
    last_error = ""
    async with _get_semaphore():
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        last_error = f"http_{resp.status}:{text[:200]}"
                        await asyncio.sleep(min(2**attempt, 8))
                        continue
                    data = json.loads(text)
                    message = data["choices"][0]["message"]
                    content = message.get("content")
                    if content is None:
                        content = message.get("reasoning_content") or ""
                    elif not isinstance(content, str):
                        content = json.dumps(content, ensure_ascii=False)
                    return _parse_judge_letter(content), content, ""
            except Exception as exc:
                last_error = f"{type(exc).__name__}:{exc}"
                await asyncio.sleep(min(2**attempt, 8))
    return None, "", last_error or "judge_failed"


async def _score_one(args, sample) -> dict[str, Any]:
    prompt, question, answer, prediction = _build_judge_prompt(sample)
    letter, raw, error = await _call_llm_judge(prompt)

    correct_reward = _env_float("VL_QA_CORRECT_REWARD", 1.0)
    wrong_reward = _env_float("VL_QA_WRONG_REWARD", 0.0)
    judge_fail_reward = _env_float("VL_QA_JUDGE_FAIL_REWARD", wrong_reward)

    if letter not in {"A", "B"}:
        return {
            "score": judge_fail_reward,
            "correct": 0,
            "llmjudge_ok": 0,
            "llmjudge_error": error or "judge_parse_failed",
            "question": question[:1000],
            "answer": answer[:1000],
            "prediction": prediction[:1000],
            "judge_raw": (raw or "")[:_judge_raw_max_chars()],
        }

    correct = int(letter == "A")
    return {
        "score": correct_reward if correct else wrong_reward,
        "correct": correct,
        "llmjudge_ok": 1,
        "judge_letter": letter,
        "question": question[:1000],
        "answer": answer[:1000],
        "prediction": prediction[:1000],
        "judge_raw": (raw or "")[:_judge_raw_max_chars()],
    }


async def vl_qa_llmjudge_reward(args, samples, **kwargs):
    if isinstance(samples, list):
        return await asyncio.gather(*[_score_one(args, sample) for sample in samples])
    return await _score_one(args, samples)
