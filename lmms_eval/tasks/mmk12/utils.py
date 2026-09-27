import os
import re

from lmms_eval.tasks._task_utils.hash_answer import append_hash_answer_instruction, extract_choice_answer, extract_hash_answer
from lmms_eval.tasks._task_utils.math_verify_utils import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    StringExtractionConfig,
    parse,
    verify,
)

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
_OPTION_RE = re.compile(r"(?m)^\s*([A-E])\.\s*(.+?)\s*$")
_CHOICE_CONFIG = [StringExtractionConfig(strings=("A", "B", "C", "D", "E"), try_extract_without_anchor=True, lowercase=False)]
_MATH_CONFIG = [LatexExtractionConfig(), ExprExtractionConfig()]


def _extract_answer_tag_content(text):
    match = _ANSWER_TAG_RE.search(str(text))
    if match is None:
        return None
    return match.group(1).strip()


def _parse_choice(text):
    tagged = _extract_answer_tag_content(text)
    text_to_parse = tagged if tagged is not None else str(text)
    return parse(
        text_to_parse,
        extraction_config=_CHOICE_CONFIG,
        extraction_mode="first_match",
        fallback_mode="no_fallback",
    )


def _kimi_vl_choice_value_relaxed():
    return os.environ.get("KIMI_VL_CHOICE_VALUE_RELAXED", "").lower() in {"1", "true", "yes"}


def _parse_options(question):
    options = {}
    for match in _OPTION_RE.finditer(str(question)):
        options[match.group(1)] = match.group(2).strip()
    return options


def _normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"^\$+(.*?)\$+$", r"\1", text)
    text = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = text.replace("{", "").replace("}", "").replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text.strip(".。")


def _value_matches_option(prediction_text, option_text):
    if _normalize_text(prediction_text) == _normalize_text(option_text):
        return True

    pred_parsed = parse(str(prediction_text), extraction_config=_MATH_CONFIG, extraction_mode="any_match", fallback_mode="first_match")
    option_parsed = parse(str(option_text), extraction_config=_MATH_CONFIG, extraction_mode="any_match", fallback_mode="first_match")
    return len(pred_parsed) > 0 and len(option_parsed) > 0 and verify(option_parsed, pred_parsed, strict=True)


def _choice_value_matches_target(raw_prediction, doc):
    target = str(doc["answer"]).strip().upper()
    if target not in {"A", "B", "C", "D", "E"}:
        return False
    options = _parse_options(doc.get("question", ""))
    if target not in options:
        return False
    prediction_text = extract_hash_answer(raw_prediction)
    return _value_matches_option(prediction_text, options[target])


def mmk12_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mmk12_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["question"].strip()
    return append_hash_answer_instruction(f"{pre_prompt}{question}{post_prompt}")


def mmk12_process_results(doc, results):
    prediction = _parse_choice(extract_choice_answer(results[0], valid_choices="ABCDE"))
    target = _parse_choice(doc["answer"])
    if len(prediction) == 0 or len(target) == 0:
        if _kimi_vl_choice_value_relaxed() and _choice_value_matches_target(results[0], doc):
            return {"exact_match": 1.0}
        return {"exact_match": 0.0}
    correct = verify(target, prediction, strict=True)
    if not correct and _kimi_vl_choice_value_relaxed() and _choice_value_matches_target(results[0], doc):
        correct = True
    return {"exact_match": float(correct)}
