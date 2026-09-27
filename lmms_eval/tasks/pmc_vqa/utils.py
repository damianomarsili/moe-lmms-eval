import binascii
import hashlib
import io
import os
import re
import struct
import sys
import threading
import zipfile
import zlib
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

from lmms_eval.tasks._task_utils.hash_answer import append_hash_answer_instruction, extract_choice_answer
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


_REPO_ID = "RadGenome/PMC-VQA"
_ZIP_FILENAMES = ("images.zip", "images_2.zip")
_TEST_CSV_FILENAMES = ("test.csv", "test_2.csv")
_CHOICE_LABELS = ("A", "B", "C", "D")
_ANSWER_PROCESSOR = EvalAIAnswerProcessor()
_ZIP_PATHS = None
_ZIP_PATHS_LOCK = threading.Lock()
_ZIP_INDEX = None
_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_DEFAULT_SAMPLE_SIZE = 2000
_DEFAULT_SAMPLE_SEED = "0"
_DATASET_COLUMNS = (
    "Figure_path",
    "Question",
    "Answer",
    "Choice A",
    "Choice B",
    "Choice C",
    "Choice D",
    "Answer_label",
)


def _image_zip_candidates():
    global _ZIP_PATHS
    if _ZIP_PATHS is not None:
        yield from _ZIP_PATHS
        return
    with _ZIP_PATHS_LOCK:
        if _ZIP_PATHS is not None:
            yield from _ZIP_PATHS
            return
        local_paths = os.getenv("PMC_VQA_IMAGES_ZIP")
        if local_paths:
            _ZIP_PATHS = [path for path in local_paths.split(os.pathsep) if path]
        else:
            local_dir = os.getenv("PMC_VQA_IMAGES_DIR")
            if local_dir:
                _ZIP_PATHS = [os.path.join(local_dir, filename) for filename in _ZIP_FILENAMES]
            else:
                cached_paths = _cached_hf_snapshot_files(_ZIP_FILENAMES)
                _ZIP_PATHS = cached_paths or [
                    hf_hub_download(repo_id=_REPO_ID, filename=filename, repo_type="dataset")
                    for filename in _ZIP_FILENAMES
                ]
    yield from _ZIP_PATHS


def _cached_hf_snapshot_files(filenames):
    cache_root = Path(os.path.expanduser("~/.cache/huggingface/hub/datasets--RadGenome--PMC-VQA/snapshots"))
    if not cache_root.is_dir():
        return []
    for snapshot_dir in sorted(cache_root.iterdir(), reverse=True):
        paths = [snapshot_dir / filename for filename in filenames]
        if all(path.exists() for path in paths):
            return [str(path) for path in paths]
    return []


def _images_zips():
    for path in _image_zip_candidates():
        with zipfile.ZipFile(path) as archive:
            yield path, archive


def _build_zip_index():
    global _ZIP_INDEX
    if _ZIP_INDEX is None:
        _ZIP_INDEX = {}
        for archive_path, archive in _images_zips():
            for name in archive.namelist():
                if not name.endswith("/"):
                    _ZIP_INDEX[name] = (archive_path, name)
                    _ZIP_INDEX[os.path.basename(name)] = (archive_path, name)
                    if name.startswith("images/"):
                        _ZIP_INDEX[name.removeprefix("images/")] = (archive_path, name)
    return _ZIP_INDEX


def _zip_member_for_figure(figure_path):
    figure_path = str(figure_path).strip()
    index = _build_zip_index()
    candidates = [
        figure_path,
        f"images/{figure_path}",
        f"figures/{figure_path}",
        os.path.basename(figure_path),
        f"images/{os.path.basename(figure_path)}",
        f"figures/{os.path.basename(figure_path)}",
    ]
    for candidate in candidates:
        if candidate in index:
            return index[candidate]

    basename = os.path.basename(figure_path)
    if basename in index:
        return index[basename]

    suffix = f"/{basename}"
    for archive_path, archive in _images_zips():
        for name in archive.namelist():
            if name == basename or name.endswith(suffix):
                return archive_path, name

    raise FileNotFoundError(f"Could not find {figure_path!r} in {', '.join(_image_zip_candidates())}")


def _figure_exists(figure_path):
    try:
        _zip_member_for_figure(figure_path)
        return True
    except FileNotFoundError:
        return False


def _clean_choice(choice, label):
    choice = str(choice).strip()
    return re.sub(rf"^\s*{label}\s*[:.)-]\s*", "", choice, flags=re.IGNORECASE).strip()


def _choice_texts(doc):
    return {label: _clean_choice(doc.get(f"Choice {label}", ""), label) for label in _CHOICE_LABELS}


def _clean_label(label):
    label = str(label).strip().upper()
    match = re.search(r"[A-D]", label)
    return match.group(0) if match else label


def _label_or_empty(label):
    label = str(label).strip().upper()
    match = re.search(r"[A-D]", label)
    return match.group(0) if match else ""


def _normalize_answer_text(value):
    value = str(value).strip().lower()
    value = re.sub(r"^\s*[a-d]\s*[:.)-]\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .;:,")


def _infer_answer_label(example):
    answer = _normalize_answer_text(example["Answer"])
    direct_label = _label_or_empty(answer)
    if direct_label in _CHOICE_LABELS and len(answer) <= 1:
        return direct_label

    for label in _CHOICE_LABELS:
        choice = _normalize_answer_text(_clean_choice(example.get(f"Choice {label}", ""), label))
        if answer and answer == choice:
            return label
    return ""


def _pmc_vqa_csv_paths(filenames, dataset_kwargs):
    data_dir = os.getenv("PMC_VQA_DATA_DIR")
    if data_dir:
        return [
            os.path.join(data_dir, filename)
            for filename in filenames
            if os.path.exists(os.path.join(data_dir, filename))
        ]

    dataset_kwargs = dataset_kwargs or {}
    return [
        hf_hub_download(
            repo_id=_REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=dataset_kwargs.get("token", None),
            local_files_only=dataset_kwargs.get("local_files_only", False),
        )
        for filename in filenames
    ]


def _load_pmc_vqa_rows(filenames, dataset_kwargs):
    import pandas as pd

    rows = []
    skipped_missing_images = 0
    for path in _pmc_vqa_csv_paths(filenames, dataset_kwargs):
        data = pd.read_csv(path, dtype=str, keep_default_na=False)
        for raw_row in data.to_dict("records"):
            example = {column: str(raw_row.get(column, "")).strip() for column in _DATASET_COLUMNS}
            example["Answer_label"] = _label_or_empty(example["Answer_label"]) or _infer_answer_label(example)
            if all(example[column] for column in ("Figure_path", "Question", "Answer_label")):
                if os.getenv("PMC_VQA_FILTER_MISSING_IMAGES", "1") != "0" and not _figure_exists(example["Figure_path"]):
                    skipped_missing_images += 1
                    continue
                rows.append(example)
    if skipped_missing_images:
        print(
            f"[pmc_vqa] skipped {skipped_missing_images} rows whose Figure_path was not found in image zips",
            file=sys.stderr,
        )
    return rows


def pmc_vqa_load_dataset(dataset_kwargs=None, dataset_path=None, dataset_name=None):
    import datasets

    rows = _load_pmc_vqa_rows(_TEST_CSV_FILENAMES, dataset_kwargs)
    rows = _sample_pmc_vqa_rows(rows)
    features = datasets.Features({column: datasets.Value("string") for column in _DATASET_COLUMNS})
    return datasets.DatasetDict(
        {
            "test": datasets.Dataset.from_list(
                rows,
                features=features,
            ),
        }
    )


def _sample_pmc_vqa_rows(rows):
    sample_size = int(os.getenv("PMC_VQA_SAMPLE_SIZE", str(_DEFAULT_SAMPLE_SIZE)))
    if sample_size <= 0 or sample_size >= len(rows):
        return rows
    seed = os.getenv("PMC_VQA_SAMPLE_SEED", _DEFAULT_SAMPLE_SEED)
    return sorted(rows, key=lambda row: _stable_sample_key(row, seed))[:sample_size]


def _stable_sample_key(row, seed):
    parts = [
        seed,
        row.get("Figure_path", ""),
        row.get("Question", ""),
        row.get("Choice A", ""),
        row.get("Choice B", ""),
        row.get("Choice C", ""),
        row.get("Choice D", ""),
        row.get("Answer_label", ""),
    ]
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def pmc_vqa_doc_to_visual(doc):
    archive_path, member = _zip_member_for_figure(doc["Figure_path"])
    image = Image.open(io.BytesIO(_read_zip_member_bytes(archive_path, member))).convert("RGB")
    return [image]


def _read_zip_member_bytes(archive_path, member):
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member)
        if info.file_size > _MAX_IMAGE_BYTES:
            raise ValueError(f"Refusing to read oversized image member {member!r}: {info.file_size} bytes")
        try:
            with archive.open(info) as image_file:
                return image_file.read()
        except zipfile.BadZipFile as exc:
            if "Overlapped entries" not in str(exc):
                raise
            return _read_overlapped_zip_member(archive_path, info)


def _read_overlapped_zip_member(archive_path, info):
    if info.file_size > _MAX_IMAGE_BYTES:
        raise ValueError(f"Refusing to read oversized image member {info.filename!r}: {info.file_size} bytes")

    with open(archive_path, "rb") as archive_file:
        archive_file.seek(info.header_offset)
        header = archive_file.read(30)
        if len(header) != 30:
            raise zipfile.BadZipFile(f"Truncated local header for {info.filename!r}")
        signature, _, _, _, _, _, _, _, _, filename_len, extra_len = struct.unpack("<IHHHHHIIIHH", header)
        if signature != 0x04034B50:
            raise zipfile.BadZipFile(f"Bad local header for {info.filename!r}")
        archive_file.seek(filename_len + extra_len, os.SEEK_CUR)
        compressed = archive_file.read(info.compress_size)

    if len(compressed) != info.compress_size:
        raise zipfile.BadZipFile(f"Truncated compressed data for {info.filename!r}")
    if info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    else:
        raise NotImplementedError(f"Unsupported compression type {info.compress_type} for {info.filename!r}")
    if len(data) != info.file_size:
        raise zipfile.BadZipFile(f"Bad uncompressed size for {info.filename!r}")
    if (binascii.crc32(data) & 0xFFFFFFFF) != info.CRC:
        raise zipfile.BadZipFile(f"Bad CRC-32 for {info.filename!r}")
    return data


def pmc_vqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = str(doc["Question"]).strip()
    choices = _choice_texts(doc)
    options = "\n".join(f"{label}. {choices[label]}" for label in _CHOICE_LABELS)
    return append_hash_answer_instruction(f"{pre_prompt}{question}\n{options}{post_prompt}")


def _prediction_to_label(prediction, doc):
    prediction = str(prediction).strip()
    label = _clean_label(prediction)
    if label in _CHOICE_LABELS and len(prediction) <= 8:
        return label

    processed_prediction = _ANSWER_PROCESSOR(prediction)
    choices = _choice_texts(doc)
    for choice_label, choice_text in choices.items():
        if processed_prediction == _ANSWER_PROCESSOR(choice_text):
            return choice_label
    if processed_prediction == _ANSWER_PROCESSOR(doc.get("Answer", "")):
        return _clean_label(doc["Answer_label"])
    return label


def pmc_vqa_process_results(doc, results):
    prediction = extract_choice_answer(results[0], valid_choices="ABCD")
    prediction_label = _prediction_to_label(prediction, doc)
    target_label = _clean_label(doc["Answer_label"])
    return {"exact_match": float(prediction_label == target_label)}
