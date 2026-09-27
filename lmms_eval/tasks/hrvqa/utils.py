from lmms_eval.tasks._task_utils.hash_answer import append_hash_answer_instruction
from lmms_eval.tasks._task_utils.remote_sensing_vqa_eval import score_remote_sensing_vqa_answer


def hrvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def hrvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return append_hash_answer_instruction(f"{pre_prompt}{str(doc['question']).strip()}{post_prompt}")


def hrvqa_process_results(doc, results):
    return score_remote_sensing_vqa_answer(doc, results)
