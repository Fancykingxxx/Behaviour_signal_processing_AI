import os
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat


# =========================================================
# 1. User settings
# =========================================================
mat_path = "/Users/jinfengxi/Downloads/jupyter_tem/data/animal2/combined_data.mat"
seq_path = "/Users/jinfengxi/Downloads/jupyter_tem/data/animal2/vest_sequence.txt"

# Output folder
output_dir = "/Users/jinfengxi/Documents/GitHub/Behaviour_signal_processing_AI/sample data"
os.makedirs(output_dir, exist_ok=True)

output_json_path = os.path.join(output_dir, "processed_features.json")
output_csv_path = os.path.join(output_dir, "processed_features_summary.csv")

# Data settings
frames_per_trial = 80
n_trials_expected = 600

# Each trial lasts 4 seconds
trial_duration_sec = 4.0
dt = trial_duration_sec / frames_per_trial  # 0.05 s per frame


# =========================================================
# 2. Helper functions
# =========================================================
def load_sequence_txt(seq_path):
    """
    Load the sequence file.
    Supports:
    - one number per line
    - space-separated values
    - mixed whitespace separators
    """
    with open(seq_path, "r") as f:
        text = f.read().strip()

    tokens = text.replace(",", " ").split()
    seq = [int(x) for x in tokens]
    return np.array(seq, dtype=int)


def matlab_array_to_2d(arr, name, expected_shape=(80, 600)):
    """
    Convert MATLAB array to a 2D NumPy array and validate shape.
    """
    arr = np.array(arr)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(f"{name} is not 2D after squeeze. shape={arr.shape}")

    if arr.shape != expected_shape:
        raise ValueError(
            f"{name} shape mismatch. Expected {expected_shape}, got {arr.shape}"
        )

    return arr.astype(float)


def map_condition_label(label):
    """
    Map experimental condition labels:
    0 = no tilt
    1 = tilt left
    2 = tilt right
    """
    mapping = {
        0: "no_tilt",
        1: "tilt_left",
        2: "tilt_right"
    }
    return mapping.get(int(label), f"unknown_{label}")


def safe_stats(x, baseline_frames=10, response_start=10, response_end=40):
    """
    Compute basic statistics for a single trial time series.
    """
    x = np.asarray(x, dtype=float)

    baseline = x[:baseline_frames]
    response = x[response_start:response_end]

    def safe_mean(a):
        return float(np.nanmean(a)) if len(a) > 0 else np.nan

    def safe_std(a):
        return float(np.nanstd(a)) if len(a) > 0 else np.nan

    def safe_max(a):
        return float(np.nanmax(a)) if len(a) > 0 else np.nan

    def safe_min(a):
        return float(np.nanmin(a)) if len(a) > 0 else np.nan

    stats = {
        "mean": safe_mean(x),
        "std": safe_std(x),
        "max": safe_max(x),
        "min": safe_min(x),
        "baseline_mean": safe_mean(baseline),
        "baseline_std": safe_std(baseline),
        "response_mean": safe_mean(response),
        "response_max": safe_max(response),
        "response_min": safe_min(response),
        "delta_response_baseline": (
            safe_mean(response) - safe_mean(baseline)
            if len(response) > 0 and len(baseline) > 0 else np.nan
        ),
        "peak_index": int(np.nanargmax(x)) if np.any(~np.isnan(x)) else -1,
        "trough_index": int(np.nanargmin(x)) if np.any(~np.isnan(x)) else -1,
    }

    dx = np.diff(x)
    stats["mean_abs_diff"] = float(np.nanmean(np.abs(dx))) if len(dx) > 0 else np.nan
    stats["max_abs_diff"] = float(np.nanmax(np.abs(dx))) if len(dx) > 0 else np.nan

    # Convert peak/trough frame indices to time in seconds
    stats["peak_time_sec"] = stats["peak_index"] * dt if stats["peak_index"] >= 0 else np.nan
    stats["trough_time_sec"] = stats["trough_index"] * dt if stats["trough_index"] >= 0 else np.nan

    return stats


def build_trial_record(trial_idx, seq_label, aom_trial, pom_trial, pupil_trial, eye_trial, dt=1.0):
    """
    Build a processed feature record for one trial.
    """
    time_axis = (np.arange(len(aom_trial)) * dt).tolist()

    record = {
        "trial_id": f"T{trial_idx + 1:03d}",
        "trial_index": int(trial_idx),
        "condition_label": int(seq_label),
        "condition_name": map_condition_label(seq_label),
        "trial_duration_sec": float(len(aom_trial) * dt),
        "n_frames": int(len(aom_trial)),
        "dt_sec": float(dt),
        "time_axis_sec": time_axis,

        "signals": {
            "aom": [float(x) for x in aom_trial],
            "pom": [float(x) for x in pom_trial],
            "avg_pupil": [float(x) for x in pupil_trial],
            "avg_eye": [float(x) for x in eye_trial],
        },

        "features": {
            "aom": safe_stats(aom_trial),
            "pom": safe_stats(pom_trial),
            "avg_pupil": safe_stats(pupil_trial),
            "avg_eye": safe_stats(eye_trial),
        }
    }

    return record


# =========================================================
# 3. Load MAT data
# =========================================================
mat_data = loadmat(mat_path)

required_keys = ["aom", "pom", "avg_pupil", "avg_eye"]
for key in required_keys:
    if key not in mat_data:
        raise KeyError(f"Key '{key}' not found in MAT file. Available keys: {list(mat_data.keys())}")

aom = matlab_array_to_2d(mat_data["aom"], "aom", expected_shape=(frames_per_trial, n_trials_expected))
pom = matlab_array_to_2d(mat_data["pom"], "pom", expected_shape=(frames_per_trial, n_trials_expected))
avg_pupil = matlab_array_to_2d(mat_data["avg_pupil"], "avg_pupil", expected_shape=(frames_per_trial, n_trials_expected))
avg_eye = matlab_array_to_2d(mat_data["avg_eye"], "avg_eye", expected_shape=(frames_per_trial, n_trials_expected))


# =========================================================
# 4. Load sequence labels
# =========================================================
sequence = load_sequence_txt(seq_path)

if len(sequence) != n_trials_expected:
    raise ValueError(
        f"Sequence length mismatch. Expected {n_trials_expected}, got {len(sequence)}"
    )


# =========================================================
# 5. Build processed feature records
# =========================================================
records = []

for trial_idx in range(n_trials_expected):
    record = build_trial_record(
        trial_idx=trial_idx,
        seq_label=sequence[trial_idx],
        aom_trial=aom[:, trial_idx],
        pom_trial=pom[:, trial_idx],
        pupil_trial=avg_pupil[:, trial_idx],
        eye_trial=avg_eye[:, trial_idx],
        dt=dt
    )
    records.append(record)


# =========================================================
# 6. Export summary CSV for quick inspection
# =========================================================
summary_rows = []

for rec in records:
    row = {
        "trial_id": rec["trial_id"],
        "trial_index": rec["trial_index"],
        "condition_label": rec["condition_label"],
        "condition_name": rec["condition_name"],
        "trial_duration_sec": rec["trial_duration_sec"],
        "dt_sec": rec["dt_sec"],
    }

    for signal_name in ["aom", "pom", "avg_pupil", "avg_eye"]:
        feats = rec["features"][signal_name]
        row[f"{signal_name}_mean"] = feats["mean"]
        row[f"{signal_name}_std"] = feats["std"]
        row[f"{signal_name}_max"] = feats["max"]
        row[f"{signal_name}_min"] = feats["min"]
        row[f"{signal_name}_baseline_mean"] = feats["baseline_mean"]
        row[f"{signal_name}_response_mean"] = feats["response_mean"]
        row[f"{signal_name}_delta_response_baseline"] = feats["delta_response_baseline"]
        row[f"{signal_name}_peak_index"] = feats["peak_index"]
        row[f"{signal_name}_peak_time_sec"] = feats["peak_time_sec"]
        row[f"{signal_name}_mean_abs_diff"] = feats["mean_abs_diff"]

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)


# =========================================================
# 7. Save outputs
# =========================================================
with open(output_json_path, "w") as f:
    json.dump(records, f, indent=2)

summary_df.to_csv(output_csv_path, index=False)

print("Done.")
print(f"Processed JSON saved to: {output_json_path}")
print(f"Summary CSV saved to:   {output_csv_path}")
print()
print("Example trial record:")
print(json.dumps(records[0], indent=2)[:2000])
print()
print("Summary head:")
print(summary_df.head())