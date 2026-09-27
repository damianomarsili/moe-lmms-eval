#!/usr/bin/env python3
"""Recompute lmms-eval metrics from saved samples without loading a model.

Run with the lmms-eval environment.  The script reloads the original dataset
documents, feeds each saved generation to the task's current ``process_results``
method, and invokes the task's registered aggregation function.  Original
artifacts are preserved; patched copies are named ``*_replayed_results.json``.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
from collections import defaultdict
from pathlib import Path

from datasets import Image
from lmms_eval.tasks import TaskManager, get_task_dict


def sample_task_name(path: Path) -> str:
    prefix, marker, suffix = path.name.partition("_samples_")
    if not marker or not suffix.endswith(".jsonl"):
        raise ValueError(f"Unexpected samples filename: {path}")
    return suffix.removesuffix(".jsonl")


def original_results_path(path: Path) -> Path:
    prefix, marker, _ = path.name.partition("_samples_")
    if not marker:
        raise ValueError(f"Unexpected samples filename: {path}")
    return path.with_name(f"{prefix}_results.json")


def replay_samples(path: Path, task) -> dict[str, list]:
    # Graders consume labels and prompt metadata, not image pixels. Avoid
    # decoding every image so replay remains lightweight and does not exhaust
    # file descriptors on large multimodal datasets.
    docs = task.eval_docs
    non_image_columns = [name for name, feature in docs.features.items() if not isinstance(feature, Image)]
    docs = docs.select_columns(non_image_columns)
    values = defaultdict(list)
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            sample = json.loads(line)
            doc_id = sample["doc_id"]
            try:
                doc = docs[doc_id]
                response = sample["resps"][0][0]
            except (IndexError, KeyError, TypeError) as error:
                raise ValueError(f"Malformed sample at {path}:{line_number}") from error
            for metric, value in task.process_results(doc, [response]).items():
                values[metric].append(value)
    return values


def metric_value(aggregate, values: list):
    """Use the task aggregation exactly as lmms-eval would."""
    parameters = list(inspect.signature(aggregate).parameters.values())
    requires_args = any(
        parameter.name == "args"
        and parameter.default is inspect.Parameter.empty
        for parameter in parameters
    )
    return aggregate(values, None) if requires_args else aggregate(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sample_paths = sorted(args.results_root.rglob("*_samples_*.jsonl"))
    by_results = defaultdict(list)
    for path in sample_paths:
        result_path = original_results_path(path)
        if result_path.is_file():
            by_results[result_path].append(path)

    manager = TaskManager()
    task_cache = {}
    for results_path, paths in by_results.items():
        with results_path.open() as handle:
            replayed = copy.deepcopy(json.load(handle))

        for samples_path in paths:
            task_name = sample_task_name(samples_path)
            if task_name not in task_cache:
                task_cache[task_name] = get_task_dict([task_name], task_manager=manager)[task_name]
            task = task_cache[task_name]
            aggregates = task.aggregation()
            values = replay_samples(samples_path, task)
            for metric, metric_samples in values.items():
                # Submission aggregators write external files and are not scalar
                # evaluation scores.  Preserve the existing submission artifact.
                if metric == "submission" or metric not in aggregates:
                    continue
                value = metric_value(aggregates[metric], metric_samples)
                key = f"{metric},none"
                if task_name not in replayed["results"]:
                    raise KeyError(f"{task_name} absent from {results_path}")
                replayed["results"][task_name][key] = value
                print(f"{task_name}\t{metric}\t{value}\t{samples_path}")

        replayed["replayed_from_saved_samples"] = True
        replayed["replayed_with"] = "current task process_results and aggregation (no model inference)"
        output = results_path.with_name(results_path.name.replace("_results.json", "_replayed_results.json"))
        if not args.dry_run:
            with output.open("w") as handle:
                json.dump(replayed, handle, indent=2)
                handle.write("\n")


if __name__ == "__main__":
    main()
