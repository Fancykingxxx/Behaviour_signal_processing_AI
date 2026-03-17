import os
import json
import hashlib
from collections import Counter

# Optional: Hugging Face datasets check
TRY_HF_DATASETS = True

file_paths = [
    r"C:\Users\t54547fj\OneDrive - The University of Manchester\Documents\GitHub\Behaviour_signal_processing_AI\training_data_set\set_1.jsonl",
    r"C:\Users\t54547fj\OneDrive - The University of Manchester\Documents\GitHub\Behaviour_signal_processing_AI\training_data_set\train_from_stats_v2.jsonl",
]

required_fields = ["id", "task_type", "instruction", "input", "output"]


def text_len(x):
    if x is None:
        return 0
    return len(str(x))


def short_hash_record(record):
    """
    Hash content except id, so we can detect duplicated examples
    even if the IDs are different.
    """
    obj = {
        "task_type": record.get("task_type", ""),
        "instruction": record.get("instruction", ""),
        "input": record.get("input", ""),
        "output": record.get("output", ""),
    }
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def validate_jsonl_file(file_path):
    print("=" * 100)
    print(f"Checking file: {file_path}")

    if not os.path.exists(file_path):
        print("ERROR: File does not exist.")
        return None

    records = []
    errors = []
    warnings = []

    ids = []
    content_hashes = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if line == "":
                warnings.append(f"Line {line_num}: empty line")
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: invalid JSON -> {e}")
                continue

            if not isinstance(obj, dict):
                errors.append(f"Line {line_num}: JSON object is not a dictionary")
                continue

            # Check required fields
            missing = [k for k in required_fields if k not in obj]
            if missing:
                errors.append(f"Line {line_num}: missing required fields -> {missing}")
                continue

            # Check field types
            for k in required_fields:
                if not isinstance(obj[k], str):
                    errors.append(
                        f"Line {line_num}: field '{k}' must be a string, got {type(obj[k]).__name__}"
                    )

            # Soft warnings
            if isinstance(obj.get("output"), str) and obj["output"].strip() == "":
                warnings.append(f"Line {line_num}: empty output")

            if isinstance(obj.get("instruction"), str) and len(obj["instruction"].strip()) < 10:
                warnings.append(f"Line {line_num}: very short instruction")

            if isinstance(obj.get("input"), str) and len(obj["input"].strip()) < 20:
                warnings.append(f"Line {line_num}: very short input")

            if isinstance(obj.get("output"), str) and len(obj["output"].strip()) < 20:
                warnings.append(f"Line {line_num}: very short output")

            ids.append(obj["id"])
            content_hashes.append(short_hash_record(obj))
            records.append(obj)

    # Duplicate IDs within file
    id_counter = Counter(ids)
    duplicate_ids = [k for k, v in id_counter.items() if v > 1]
    if duplicate_ids:
        errors.append(f"Duplicate IDs in file: {duplicate_ids[:10]}")

    # Duplicate content within file
    hash_counter = Counter(content_hashes)
    duplicate_contents = [k for k, v in hash_counter.items() if v > 1]
    if duplicate_contents:
        warnings.append(
            f"Duplicate example contents in file: {len(duplicate_contents)} duplicated content groups"
        )

    # Stats
    if records:
        task_counter = Counter(r["task_type"] for r in records)

        instruction_lens = [text_len(r["instruction"]) for r in records]
        input_lens = [text_len(r["input"]) for r in records]
        output_lens = [text_len(r["output"]) for r in records]

        print(f"Total valid records loaded: {len(records)}")
        print("Task type counts:")
        for task, count in task_counter.items():
            print(f"  - {task}: {count}")

        print("Length stats:")
        print(
            f"  instruction: min={min(instruction_lens)}, mean={sum(instruction_lens)/len(instruction_lens):.1f}, max={max(instruction_lens)}"
        )
        print(
            f"  input:       min={min(input_lens)}, mean={sum(input_lens)/len(input_lens):.1f}, max={max(input_lens)}"
        )
        print(
            f"  output:      min={min(output_lens)}, mean={sum(output_lens)/len(output_lens):.1f}, max={max(output_lens)}"
        )
    else:
        print("No valid records loaded.")

    if errors:
        print("\nERRORS:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
    else:
        print("\nNo structural errors found.")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings[:20]:
            print(f"  - {w}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more warnings")
    else:
        print("\nNo warnings.")

    return {
        "file_path": file_path,
        "records": records,
        "errors": errors,
        "warnings": warnings,
        "ids": ids,
        "content_hashes": content_hashes,
    }


def check_cross_file_duplicates(results):
    print("\n" + "=" * 100)
    print("Cross-file checks")

    all_ids = []
    all_hashes = []

    for res in results:
        if res is None:
            continue
        for _id in res["ids"]:
            all_ids.append((_id, os.path.basename(res["file_path"])))
        for h in res["content_hashes"]:
            all_hashes.append((h, os.path.basename(res["file_path"])))

    id_counter = Counter([x[0] for x in all_ids])
    dup_ids = [k for k, v in id_counter.items() if v > 1]

    hash_counter = Counter([x[0] for x in all_hashes])
    dup_hashes = [k for k, v in hash_counter.items() if v > 1]

    if dup_ids:
        print("Duplicate IDs across files:")
        for dup_id in dup_ids[:20]:
            files = [fname for _id, fname in all_ids if _id == dup_id]
            print(f"  - {dup_id}: {files}")
    else:
        print("No duplicate IDs across files.")

    if dup_hashes:
        print(f"Duplicate example contents across files: {len(dup_hashes)} duplicated content groups")
    else:
        print("No duplicate example contents across files.")


def try_load_with_hf_datasets(file_paths):
    print("\n" + "=" * 100)
    print("Hugging Face datasets loading test")

    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets package is not installed. Skipping Hugging Face loading test.")
        print("Install with: pip install datasets")
        return

    for file_path in file_paths:
        try:
            ds = load_dataset("json", data_files=file_path, split="train")
            print(f"Loaded successfully with datasets: {os.path.basename(file_path)}")
            print(ds)
            if len(ds) > 0:
                print("First example:")
                print(ds[0])
        except Exception as e:
            print(f"FAILED to load with datasets: {os.path.basename(file_path)}")
            print(f"Reason: {e}")


def main():
    results = []
    for fp in file_paths:
        results.append(validate_jsonl_file(fp))

    check_cross_file_duplicates(results)

    if TRY_HF_DATASETS:
        try_load_with_hf_datasets(file_paths)

    print("\n" + "=" * 100)
    print("Final verdict")
    any_errors = False
    for res in results:
        if res is None:
            any_errors = True
            continue
        if len(res["errors"]) > 0:
            any_errors = True

    if any_errors:
        print("At least one file has errors. Fix those before training.")
    else:
        print("Both files are structurally valid for instruction-tuning style JSONL.")
        print("You can proceed to dataset loading / train-val split / LoRA training.")


if __name__ == "__main__":
    main()