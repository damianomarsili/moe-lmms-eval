import os
import re

from lmms_eval.tasks._task_utils.hash_answer import extract_answer_for_target
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


_ANSWER_PROCESSOR = EvalAIAnswerProcessor()
_AREA_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:m\s*(?:\^?\s*2|sq))?\s*$", re.IGNORECASE)
_HAS_AREA_UNIT_RE = re.compile(r"m\s*(?:\^?\s*2|sq)\s*$", re.IGNORECASE)


def score_remote_sensing_vqa_answer(doc, results):
    prediction = extract_answer_for_target(results[0], doc["answer"])
    target = str(doc["answer"])
    return {"exact_match": float(_answers_match(prediction, target))}


def _answers_match(prediction, target):
    raw_prediction = str(prediction)
    prediction = _ANSWER_PROCESSOR(raw_prediction)
    target = _ANSWER_PROCESSOR(target)
    if prediction == target:
        return True
    if _kimi_vl_yesno_relaxed() and target in {"yes", "no"}:
        binary_prediction = _normalize_binary_answer(raw_prediction)
        if binary_prediction is not None:
            return binary_prediction == target
    return _area_answers_match(prediction, target)


def _kimi_vl_yesno_relaxed():
    return (
        os.environ.get("KIMI_VL_YESNO_RELAXED", "").lower() in {"1", "true", "yes"}
        or os.environ.get("KIMI_VL_REMOTE_YESNO_RELAXED", "").lower() in {"1", "true", "yes"}
    )


def _normalize_binary_answer(answer):
    answer = _ANSWER_PROCESSOR(answer)
    if re.search(r"\bno\b", answer) or re.search(r"\b(?:not|n't)\b", answer):
        return "no"
    if re.search(r"\byes\b", answer):
        return "yes"
    return None


def _area_answers_match(prediction, target):
    pred_value = _parse_area_or_plain_number(prediction)
    target_value = _parse_area_or_plain_number(target)
    if pred_value is None or target_value is None:
        return False
    if not (_has_area_unit(prediction) or _has_area_unit(target)):
        return False
    return pred_value == target_value


def _parse_area_or_plain_number(answer):
    match = _AREA_RE.fullmatch(str(answer).strip())
    if match is None:
        return None
    return float(match.group(1))


def _has_area_unit(answer):
    return bool(_HAS_AREA_UNIT_RE.search(str(answer).strip()))
