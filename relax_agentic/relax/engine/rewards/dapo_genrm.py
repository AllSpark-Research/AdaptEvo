# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""DAPO GenRM Reward Model Implementation.

This module implements a reward model using GenRM (Generative Reward Model) for
DAPO-style math question answering tasks.
"""

import re
from typing import List

import httpx

from relax.utils.genrm_client import get_genrm_client
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# In-context examples steering the judge to output only "1" or "0" and to
# tolerate common math notation differences (LaTeX vs plain, wrapped in prose).
DAPO_GENRM_ICE_EXAMPLES = """
[Question]: Find $FG^2$.
[Standard Answer]: 145
[Model_answer] : 145
Judgement: 1

[Question]: Simplify the fraction.
[Standard Answer]: 2/3
[Model_answer] : \\frac{2}{3}
Judgement: 1

[Question]: Compute the sum.
[Standard Answer]: 12
[Model_answer] : The sum is 12.
Judgement: 1

[Question]: Find $x$.
[Standard Answer]: 7
[Model_answer] : 8
Judgement: 0

[Question]: Compute the area.
[Standard Answer]: 145
[Model_answer] : \\sqrt{145}
Judgement: 0
"""

# GenRM prompt template for DAPO. `{ice_examples}` is inlined between the
# instruction and the actual query so few-shot demonstrations steer format.
DAPO_GENRM_PROMPT_TEMPLATE = """Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. Different notations of the same value are consistent, e.g. '\\frac{{1}}{{2}}' and '0.5', or '145' and 'the answer is 145'.
If they are consistent, Judgement is 1; if they are different, Judgement is 0. Just output Judgement and don't output anything else.
{ice_examples}
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""


# Cap extracted answer length: prevents the actor from stuffing the answer
# slot with paragraphs to hack the judge into always agreeing.
MAX_ANSWER_LEN = 500

_ANSWER_LINE_RE = re.compile(r"Answer\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def _extract_boxed(text: str) -> str | None:
    # Bracket-balanced walk from the last `\boxed{` — handles nested braces.
    idx = text.rfind(r"\boxed{")
    if idx < 0:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def _extract_answer(text: str) -> str | None:
    """Extract math answer: prefer last `\\boxed{...}`, else last `Answer:`
    line."""
    boxed = _extract_boxed(text)
    if boxed:
        return boxed
    matches = list(_ANSWER_LINE_RE.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    return None


def _format_messages(question: str, ground_truth: str, predict_str: str) -> List[dict]:
    prompt = DAPO_GENRM_PROMPT_TEMPLATE.format(
        ice_examples=DAPO_GENRM_ICE_EXAMPLES,
        question=question,
        ground_truth=ground_truth,
        predict_str=predict_str,
    )
    return [{"role": "user", "content": prompt}]


async def async_compute_score_genrm(args, sample) -> dict:
    """Compute reward score using GenRM service (async).

    Format check short-circuits before calling GenRM: if the actor did not emit
    a parseable answer (`\\boxed{...}` or `Answer: ...`), or the answer is
    absurdly long, reward is 0 and no judge call is issued.

    Returns dict with:
        - score: float (0.0 or 1.0)
        - acc:   int (0 or 1)
        - pred:  str — parsed judgement token (empty on format error)
        - judge_response: str — raw judge output (empty on format error)
        - format_error: str — reason tag ("" when no error)
    """
    try:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        question = metadata.get("question", sample.prompt if hasattr(sample, "prompt") else "")
        ground_truth = metadata.get("label", sample.label if hasattr(sample, "label") else "")
        predict_str = sample.response

        # --- Format check: short-circuit before wasting a judge call ---
        answer_text = _extract_answer(predict_str)
        if answer_text is None:
            return {"score": 0.0, "acc": 0, "pred": "", "judge_response": "", "format_error": "answer_missing"}
        if len(answer_text) > MAX_ANSWER_LEN:
            return {"score": 0.0, "acc": 0, "pred": "", "judge_response": "", "format_error": "answer_too_long"}

        # --- Judge call ---
        # Client already retried transient failures (see genrm_client.py).
        # If it still failed, degrade to score=0 rather than aborting the
        # whole rollout batch — one flaky judge call must not kill training.
        genrm_client = get_genrm_client()
        messages = _format_messages(question, ground_truth, answer_text)
        try:
            judge_response = await genrm_client.generate(messages)
        except httpx.HTTPError as e:
            logger.error(f"GenRM judge call failed after client retries, degrading to score=0: {e}")
            return {
                "score": 0.0,
                "acc": 0,
                "pred": "",
                "judge_response": "",
                "format_error": "judge_transient_error",
            }

        # --- Loose parse: peel off any "Judgement:" prefix, then look for
        # 1 or 0 in the head. Judge sampling is capped short (max_new_tokens
        # ~32), so head=first 16 chars is enough context.
        prediction = judge_response.strip()
        if "Judgement:" in prediction:
            prediction = prediction.split("Judgement:")[-1].strip()
        head = prediction[:16]
        if "1" in head:
            score, acc = 1.0, 1
        elif "0" in head:
            score, acc = 0.0, 0
        else:
            logger.warning(f"GenRM response format unrecognized: {prediction!r}")
            score, acc = 0.0, 0

        return {
            "score": score,
            "acc": acc,
            "pred": prediction,
            "judge_response": judge_response,
            "format_error": "",
        }

    except Exception as e:
        logger.error(f"GenRM async_compute_score_genrm failed: {e}")
        raise
