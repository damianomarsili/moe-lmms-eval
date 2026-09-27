import os
import re

from lmms_eval.tasks._task_utils.hash_answer import extract_answer_for_target
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


_ANSWER_PROCESSOR = EvalAIAnswerProcessor()
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{", flags=re.DOTALL)
_THINK_TAG_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.IGNORECASE | re.DOTALL)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?%?")
_NEGATION_WORDS = {"no", "not", "non", "none", "without", "absent", "absence", "negative"}


def score_medical_vqa_answer(raw_prediction, target):
    extracted = extract_answer_for_target(raw_prediction, target)
    prediction = _ANSWER_PROCESSOR(extracted)
    gold = _ANSWER_PROCESSOR(target)
    if _kimi_vl_yesno_relaxed() and gold in {"yes", "no"}:
        binary_prediction = _normalize_binary_answer(extracted)
        if binary_prediction is not None:
            prediction = binary_prediction
    exact = float(prediction == gold)
    token_f1 = _token_f1(prediction, gold)
    relaxed = float(_relaxed_match(prediction, gold, token_f1))
    format_metrics = _format_metrics(raw_prediction)

    return {
        "exact_match": exact,
        "medical_relaxed_match": relaxed,
        "medical_token_f1": token_f1,
        **format_metrics,
    }


def _format_metrics(raw_prediction):
    tail = str(raw_prediction)[-4000:]
    has_think_tag = bool(_THINK_TAG_RE.search(tail))
    has_answer_tag = bool(_ANSWER_TAG_RE.search(tail))
    has_boxed = bool(_BOXED_RE.search(tail))
    return {
        "medical_format_valid": float(has_think_tag and has_answer_tag and has_boxed),
        "medical_has_answer_tag": float(has_answer_tag),
        "medical_has_boxed": float(has_boxed),
    }


def _relaxed_match(prediction, gold, token_f1):
    if prediction == gold:
        return True
    if not prediction or not gold:
        return False
    if _is_binary(gold) or _is_number(gold):
        return False

    pred_tokens = _canonical_tokens(prediction)
    gold_tokens = _canonical_tokens(gold)
    if not pred_tokens or not gold_tokens:
        return False
    if _has_negation(pred_tokens) != _has_negation(gold_tokens):
        return False
    if pred_tokens == gold_tokens:
        return True

    pred_text = " ".join(pred_tokens)
    gold_text = " ".join(gold_tokens)
    if _contains_phrase(pred_text, gold_text) or _contains_phrase(gold_text, pred_text):
        return True
    return token_f1 >= 0.85


def _kimi_vl_yesno_relaxed():
    return os.environ.get("KIMI_VL_YESNO_RELAXED", "").lower() in {"1", "true", "yes"}


def _normalize_binary_answer(answer):
    answer = _ANSWER_PROCESSOR(answer)
    if re.search(r"\bno\b", answer) or re.search(r"\b(?:not|n't)\b", answer):
        return "no"
    if re.search(r"\byes\b", answer):
        return "yes"
    return None


def _token_f1(prediction, gold):
    pred_tokens = _canonical_tokens(prediction)
    gold_tokens = _canonical_tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = 0
    remaining = list(gold_tokens)
    for token in pred_tokens:
        if token in remaining:
            common += 1
            remaining.remove(token)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _canonical_tokens(text):
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return [_singularize(token) for token in tokens]


def _singularize(token):
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]
    if len(token) > 4 and token.endswith("ses"):
        return token[:-1]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is", "as")):
        return token[:-1]
    return token


def _contains_phrase(text, phrase):
    if len(phrase) < 6:
        return False
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def _is_binary(text):
    return str(text).strip().lower() in {"yes", "no"}


def _is_number(text):
    return bool(_NUMBER_RE.fullmatch(str(text).strip().replace(",", "")))


def _has_negation(tokens):
    return any(token in _NEGATION_WORDS for token in tokens)
