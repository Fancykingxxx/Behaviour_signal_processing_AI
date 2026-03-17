import json
import random
from pathlib import Path

base = Path(r"C:\Users\t54547fj\OneDrive - The University of Manchester\Documents\GitHub\Behaviour_signal_processing_AI\training_data_set")

input_files = [
    base / "set_1.jsonl",
    base / "train_from_stats_v2.jsonl",
]

merged_path = base / "train_merged.jsonl"
train_path = base / "train.jsonl"
val_path = base / "val.jsonl"
test_path = base / "test.jsonl"

random.seed(42)

records = []
seen = set()

for fp in input_files:
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            key = (
                obj.get("task_type", ""),
                obj.get("instruction", ""),
                obj.get("input", ""),
                obj.get("output", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            records.append(obj)

random.shuffle(records)

n = len(records)
n_test = max(2, int(n * 0.15))
n_val = max(2, int(n * 0.15))
n_train = n - n_val - n_test

train_records = records[:n_train]
val_records = records[n_train:n_train + n_val]
test_records = records[n_train + n_val:]

for out_path, subset in [
    (merged_path, records),
    (train_path, train_records),
    (val_path, val_records),
    (test_path, test_records),
]:
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in subset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Total: {len(records)}")
print(f"Train: {len(train_records)}")
print(f"Val:   {len(val_records)}")
print(f"Test:  {len(test_records)}")