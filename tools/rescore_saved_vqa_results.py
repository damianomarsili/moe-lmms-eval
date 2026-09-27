#!/usr/bin/env python3
"""Rescore saved PathVQA/VQA-RAD generations without model inference.

The evaluator writes one ``*_samples_<task>.jsonl`` file per task.  Each row
already contains the target, raw generation, and prior score, so these files
are sufficient to recompute exact match after an answer-parser change.

By default this script writes a sibling ``*_rescored_patched_results.json``
file and never modifies the original result or sample files.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


SUPPORTED_TASKS = {"path_vqa", "vqa_rad"}


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result_path_for_samples(samples_path: Path) -> Path:
    prefix, marker, _ = samples_path.name.partition("_samples_")
    if not marker:
        raise ValueError(f"Not an lmms-eval samples file: {samples_path}")
    return samples_path.with_name(f"{prefix}_results.json")


def _rescored_result_path(samples_path: Path) -> Path:
    prefix, marker, _ = samples_path.name.partition("_samples_")
    if not marker:
        raise ValueError(f"Not an lmms-eval samples file: {samples_path}")
    return samples_path.with_name(f"{prefix}_rescored_patched_results.json")


def rescore_samples(samples_path: Path, extract_answer: Any, answer_processor: Any) -> tuple[float, int]:
    scores: list[float] = []
    with samples_path.open() as samples_file:
        for line_number, line in enumerate(samples_file, start=1):
            row = json.loads(line)
            try:
                raw_response = row["resps"][0][0]
                target = row["target"]
            except (KeyError, IndexError, TypeError) as error:
                raise ValueError(f"Malformed response at {samples_path}:{line_number}") from error
            prediction = extract_answer(raw_response, target)
            scores.append(float(answer_processor(prediction) == answer_processor(target)))

    if not scores:
        raise ValueError(f"No samples found in {samples_path}")
    return sum(scores) / len(scores), len(scores)


def write_rescored_results(
    original_results_path: Path,
    output_path: Path,
    samples_path: Path,
    task: str,
    score: float,
    num_samples: int,
) -> None:
    with original_results_path.open() as results_file:
        results = copy.deepcopy(json.load(results_file))

    task_results = results["results"][task]
    for key in list(task_results):
        if key.startswith("exact_match,"):
            task_results[key] = score
        elif key.startswith("exact_match_stderr,"):
            task_results[key] = math.sqrt(score * (1.0 - score) / num_samples)

    results["rescored_from_samples"] = samples_path.name
    results["rescored_with"] = "patched hash_answer extractor (unwraps LaTex \\text{...} in boxed answers)"
    results["rescored_num_samples"] = num_samples
    with output_path.open("w") as output_file:
        json.dump(results, output_file, indent=2)
        output_file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path, help="Directory to search recursively for saved samples")
    parser.add_argument("--dry-run", action="store_true", help="Print scores without writing result files")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    hash_answer = _load_module("hash_answer", repo_root / "lmms_eval/tasks/_task_utils/hash_answer.py")
    vqa_metric = _load_module("vqa_eval_metric", repo_root / "lmms_eval/tasks/_task_utils/vqa_eval_metric.py")
    answer_processor = vqa_metric.EvalAIAnswerProcessor()

    samples_paths = sorted(args.results_root.rglob("*_samples_*.jsonl"))
    processed = 0
    for samples_path in samples_paths:
        task = samples_path.parent.parent.name
        if task not in SUPPORTED_TASKS:
            continue
        original_results_path = _result_path_for_samples(samples_path)
        if not original_results_path.is_file():
            raise FileNotFoundError(f"Missing aggregate results file for {samples_path}: {original_results_path}")

        score, num_samples = rescore_samples(samples_path, hash_answer.extract_answer_for_target, answer_processor)
        output_path = _rescored_result_path(samples_path)
        print(f"{task}\t{score:.6f}\t{num_samples}\t{samples_path}")
        if not args.dry_run:
            write_rescored_results(original_results_path, output_path, samples_path, task, score, num_samples)
        processed += 1

    if not processed:
        raise SystemExit(f"No saved PathVQA or VQA-RAD samples found under {args.results_root}")


if __name__ == "__main__":
    main()
