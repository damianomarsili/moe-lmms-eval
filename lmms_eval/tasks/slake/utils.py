import io
import os
import re
import threading
import zipfile

from huggingface_hub import hf_hub_download
from PIL import Image

from lmms_eval.tasks._task_utils.hash_answer import append_hash_answer_instruction
from lmms_eval.tasks._task_utils.medical_vqa_eval import score_medical_vqa_answer


_REPO_ID = "BoKelvin/SLAKE"
_ZIP_FILENAME = "imgs.zip"
_ZIP_PATH = None
_ZIP_PATH_LOCK = threading.Lock()
_ZIP_INDEX = None
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _imgs_zip_path():
    global _ZIP_PATH
    if _ZIP_PATH is not None:
        return _ZIP_PATH
    with _ZIP_PATH_LOCK:
        if _ZIP_PATH is not None:
            return _ZIP_PATH
        local_path = os.getenv("SLAKE_IMGS_ZIP")
        if local_path:
            _ZIP_PATH = local_path
        else:
            _ZIP_PATH = hf_hub_download(repo_id=_REPO_ID, filename=_ZIP_FILENAME, repo_type="dataset")
        return _ZIP_PATH


def _zip_member_for_image(img_name):
    zip_path = _imgs_zip_path()
    img_name = str(img_name).strip()
    candidates = [
        img_name,
        f"imgs/{img_name}",
        os.path.basename(img_name),
        f"imgs/{os.path.basename(img_name)}",
    ]
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for candidate in candidates:
            if candidate in names:
                return candidate

        global _ZIP_INDEX
        if _ZIP_INDEX is None:
            _ZIP_INDEX = {name: name for name in archive.namelist() if not name.endswith("/")}
            _ZIP_INDEX.update({name.removeprefix("imgs/"): name for name in archive.namelist() if name.startswith("imgs/")})
    if img_name in _ZIP_INDEX:
        return _ZIP_INDEX[img_name]

    raise FileNotFoundError(f"Could not find {img_name!r} in {_imgs_zip_path()}")


def slake_doc_to_visual(doc):
    member = _zip_member_for_image(doc["img_name"])
    with zipfile.ZipFile(_imgs_zip_path()) as archive, archive.open(member) as image_file:
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    return [image]


def _is_english_slake_doc(doc):
    lang = (
        doc.get("q_lang")
        or doc.get("question_lang")
        or doc.get("language")
        or doc.get("lang")
    )
    if lang is not None:
        return str(lang).strip().lower() in {"en", "eng", "english"}

    question = str(doc.get("question", ""))
    return not _CJK_RE.search(question)


def slake_process_docs(dataset):
    """Keep only English SLAKE questions before lmms-eval builds prompts."""
    if hasattr(dataset, "filter"):
        return dataset.filter(_is_english_slake_doc)
    return [doc for doc in dataset if _is_english_slake_doc(doc)]


def slake_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = str(doc["question"]).strip()
    return append_hash_answer_instruction(f"{pre_prompt}{question}{post_prompt}")


def slake_process_results(doc, results):
    return score_medical_vqa_answer(results[0], doc["answer"])
