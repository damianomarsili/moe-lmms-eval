from lmms_eval.tasks._task_utils.hash_answer import append_hash_answer_instruction
from lmms_eval.tasks._task_utils.medical_vqa_eval import score_medical_vqa_answer


def path_vqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def path_vqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["question"].strip()
    return append_hash_answer_instruction(f"{pre_prompt}{question}{post_prompt}")


def path_vqa_process_results(doc, results):
    return score_medical_vqa_answer(results[0], doc["answer"])
