import re
from collections import Counter

from lmms_eval.tasks._task_utils.hash_answer import append_hash_answer_instruction, extract_answer_for_target
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

_ANSWER_PROCESSOR = EvalAIAnswerProcessor()
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_AREA_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*m\s*(?:\^?\s*2|sq)", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "there",
    "this",
    "to",
    "with",
}


def rsvlm_qa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def rsvlm_qa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return append_hash_answer_instruction(f"{pre_prompt}{str(doc['question']).strip()}{post_prompt}")


def rsvlm_qa_process_results(doc, results):
    prediction = extract_answer_for_target(results[0], doc["answer"])
    target = str(doc["answer"])
    question = str(doc.get("question", ""))
    token_f1 = _token_f1(prediction, target)
    strict_exact = _normalized_text(prediction) == _normalized_text(target)
    relaxed = strict_exact or _binary_match(prediction, target, question) or _numeric_match(prediction, target, question)
    if not relaxed:
        relaxed = _short_phrase_match(prediction, target) or token_f1 >= 0.55
    return {
        "exact_match": float(relaxed),
        "rsvlm_token_f1": token_f1,
        "rsvlm_strict_exact": float(strict_exact),
    }


def _normalized_text(text):
    return _ANSWER_PROCESSOR(str(text))


def _tokens(text):
    return [token for token in _TOKEN_RE.findall(_normalized_text(text)) if token not in _STOPWORDS]


def _token_f1(prediction, target):
    pred_tokens = _tokens(prediction)
    target_tokens = _tokens(target)
    if not pred_tokens or not target_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(target_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def _binary_match(prediction, target, question):
    if not _is_binary_question(question):
        return False
    pred_binary = _extract_binary(prediction)
    target_binary = _extract_binary(target)
    return pred_binary is not None and pred_binary == target_binary


def _is_binary_question(question):
    question = str(question).strip().lower()
    return question.startswith(("are ", "is ", "do ", "does ", "did ", "can ", "could ", "has ", "have "))


def _extract_binary(text):
    text = _normalized_text(text)
    if re.search(r"\bno\b", text) or re.search(r"\b(?:not|n't|nonexistent|nofind)\b", text):
        return "no"
    if re.search(r"\byes\b", text):
        return "yes"
    return None


def _numeric_match(prediction, target, question):
    question = str(question).lower()
    if not any(marker in question for marker in ("how many", "number of", "total number", "area", "coverage")):
        return False

    pred_area = _extract_area_value(prediction)
    target_area = _extract_area_value(target)
    if pred_area is not None or target_area is not None:
        pred_value = pred_area if pred_area is not None else _extract_number(prediction)
        target_value = target_area if target_area is not None else _extract_number(target)
        return pred_value is not None and target_value is not None and pred_value == target_value

    pred_value = _extract_number(prediction)
    target_value = _extract_number(target)
    return pred_value is not None and target_value is not None and pred_value == target_value


def _extract_area_value(text):
    match = _AREA_RE.search(str(text))
    if match is None:
        return None
    return float(match.group(1))


def _extract_number(text):
    matches = _NUMBER_RE.findall(str(text))
    if not matches:
        return None
    return float(matches[0])


def _short_phrase_match(prediction, target):
    pred_tokens = set(_tokens(prediction))
    target_tokens = set(_tokens(target))
    if not pred_tokens or len(pred_tokens) > 5:
        return False
    return pred_tokens.issubset(target_tokens)
