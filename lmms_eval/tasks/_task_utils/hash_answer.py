import os
import re


VERO_SYSTEM_PROMPT = """You are a helpful assistant answering a question about an image.

Always output exactly two XML-style sections and close every tag.
Start with <think>, close it with </think>, then open <answer>, and close with </answer>.
Do not write anything before <think> or after </answer>.

In <think>, briefly reason through the visual evidence and question.
In <answer>, give a self-contained final answer and exactly one concise final result using LaTeX boxed notation: \\boxed{final_result}."""

HASH_ANSWER_INSTRUCTION = (
    "Use this exact template, replacing only the bracketed text and final_result:\n"
    "<think>[brief reasoning]</think>\n"
    "<answer>[final answer] \\boxed{final_result}</answer>\n"
    "No text before <think>. No text after </answer>."
)
_HASH_RE = re.compile(r"####\s*(.+)", flags=re.DOTALL)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", flags=re.DOTALL)
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_CHOICE_MARKER_RE = re.compile(r"(?i)\b(?:final answer|answer|option|choice)\s*(?:is|:)?\s*(?:\*\*)?\(?([A-Z])\)?(?:\*\*)?\b")
_CHOICE_LINE_RE = re.compile(r"(?i)^\s*(?:[-*]\s*)?(?:\*\*)?\(?([A-Z])\)?(?:\*\*)?(?=\s*$|[\s:.)-])")
_LATEX_TEXT_RE = re.compile(r"\\text\s*\{([^{}]*)\}")


def append_hash_answer_instruction(prompt: str) -> str:
    prompt = str(prompt).rstrip()
    if _use_vero_prompt():
        if VERO_SYSTEM_PROMPT in prompt:
            return prompt
        return f"{VERO_SYSTEM_PROMPT}\n\nQuestion:\n{prompt}\n{HASH_ANSWER_INSTRUCTION}"
    if "####" in prompt:
        return prompt
    return f'{prompt}\nOutput only the final answer after "####".'


def _use_vero_prompt() -> bool:
    value = os.getenv("LMMS_EVAL_USE_VERO_PROMPT", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def extract_hash_answer(text: str) -> str:
    text = str(text)
    boxed = _extract_boxed_answer(text)
    if boxed is not None:
        return boxed
    matches = _HASH_RE.findall(text)
    if not matches:
        return text.strip()
    return matches[-1].strip()


def _extract_boxed_answer(text: str) -> str | None:
    tail = str(text)[-4000:]
    tag_matches = _ANSWER_TAG_RE.findall(tail)
    search_text = tag_matches[-1] if tag_matches else tail
    boxed_matches = _BOXED_RE.findall(search_text)
    if boxed_matches:
        return _strip_latex_text_wrappers(boxed_matches[-1])
    return None


def _strip_latex_text_wrappers(answer: str) -> str:
    """Remove presentation-only ``\\text{...}`` wrappers from boxed answers.

    VQA exact-match tasks compare the extracted boxed content with a plain-text
    target.  ``\\boxed{no}`` and ``\\boxed{\\text{no}}`` express the same answer,
    so preserve the inner text rather than passing the LaTex wrapper to the
    task-specific answer processor.
    """
    previous = None
    normalized = str(answer)
    while normalized != previous:
        previous = normalized
        normalized = _LATEX_TEXT_RE.sub(r"\1", normalized)
    return normalized.strip()


def has_hash_answer(text: str) -> bool:
    return bool(_HASH_RE.search(str(text)))


def extract_choice_answer(text: str, valid_choices: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> str:
    answer = extract_hash_answer(text)
    valid = {choice.upper() for choice in valid_choices}

    boxed_matches = [match.strip().upper() for match in _BOXED_RE.findall(answer)]
    boxed_matches = [match for match in boxed_matches if match in valid]
    if boxed_matches:
        return boxed_matches[-1]

    marker_matches = [match.upper() for match in _CHOICE_MARKER_RE.findall(answer)]
    marker_matches = [match for match in marker_matches if match in valid]
    if marker_matches:
        return marker_matches[-1]

    final_line = _final_nonempty_line(answer)
    line_match = _CHOICE_LINE_RE.match(final_line)
    if line_match is not None and line_match.group(1).upper() in valid:
        return line_match.group(1).upper()

    cleaned = re.sub(r"[*_`#>\[\](){}]", " ", answer)
    cleaned = re.sub(r"^\s*[-+]\s*", "", cleaned.strip())
    if len(cleaned.strip()) == 1 and cleaned.strip().upper() in valid:
        return cleaned.strip().upper()
    return answer.strip()


def _final_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    line = lines[-1]
    line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)
    line = line.strip("*_`#> \t")
    return line


def extract_answer_for_target(text: str, target: str) -> str:
    boxed_answer = _extract_boxed_answer(text)
    if boxed_answer is not None:
        return boxed_answer

    if has_hash_answer(text):
        return extract_hash_answer(text)

    tail = str(text)[-2000:]
    tag_matches = _ANSWER_TAG_RE.findall(tail)
    if tag_matches:
        return tag_matches[-1].strip()

    boxed_matches = _BOXED_RE.findall(tail)
    if boxed_matches:
        return boxed_matches[-1].strip()

    target_text = str(target).strip()
    if target_text.lower() in {"yes", "no"}:
        matches = re.findall(r"\b(yes|no)\b", tail, flags=re.IGNORECASE)
        if matches:
            return matches[-1].lower()

    if _NUMBER_RE.fullmatch(target_text.replace(",", "")) or _NUMBER_RE.fullmatch(target_text):
        matches = _NUMBER_RE.findall(tail)
        if matches:
            return matches[-1].replace(",", "")

    answer_markers = re.findall(
        r"(?i)(?:final answer|answer|therefore|thus|so)\s*(?:is|:)?\s*([^\n.]+)",
        tail,
    )
    if answer_markers:
        return answer_markers[-1].strip("*_`#> \t")

    return _final_nonempty_line(tail)
