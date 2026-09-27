"""Sanity-check the clean HRVQA Hugging Face dataset.

Run from the lmms-eval repo root:

    python lmms_eval/tasks/hrvqa/check_hrvqa.py
"""

from datasets import load_dataset

from utils import hrvqa_doc_to_visual


def main() -> None:
    ds = load_dataset("dmarsili/HRVQA-2k", split="validation", token=True)
    print(f"rows={len(ds)}")
    print(f"columns={ds.column_names}")
    for idx in [0, len(ds) // 2, len(ds) - 1]:
        row = ds[int(idx)]
        image = hrvqa_doc_to_visual(row)[0]
        print(f"sample[{idx}] size={image.size} answer={row['answer']!r} question={row['question']!r}")


if __name__ == "__main__":
    main()
