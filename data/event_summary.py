import os
import json
import numpy as np
import pandas as pd


# =========================================================
# 1. User settings
# =========================================================
json_path = r"C:\Users\t54547fj\OneDrive - The University of Manchester\Documents\GitHub\Behaviour_signal_processing_AI\sample data\processed_features.json"
output_csv_path = os.path.join(os.path.dirname(json_path), "events_summary.csv")

baseline_start_sec = 0.0
baseline_end_sec = 2.0
response_start_sec = 2.0
response_end_sec = 4.0


# =========================================================
# 2. Helper functions
# =========================================================
def get_time_masks(time_axis, baseline_start=0.0, baseline_end=2.0, response_start=2.0, response_end=4.0):
    time_axis = np.asarray(time_axis, dtype=float)

    baseline_mask = (time_axis >= baseline_start) & (time_axis < baseline_end)
    response_mask = (time_axis >= response_start) & (time_axis <= response_end)

    return baseline_mask, response_mask


def has_valid_values(x):
    x = np.asarray(x, dtype=float)
    return np.any(~np.isnan(x))


def safe_nanmin(x):
    return float(np.nanmin(x)) if has_valid_values(x) else np.nan


def safe_nanmax(x):
    return float(np.nanmax(x)) if has_valid_values(x) else np.nan


def safe_nanmean(x):
    return float(np.nanmean(x)) if has_valid_values(x) else np.nan


def safe_nanargmax(x):
    return int(np.nanargmax(x)) if has_valid_values(x) else -1


def safe_nanargmin(x):
    return int(np.nanargmin(x)) if has_valid_values(x) else -1


def detect_event_by_baseline_min_and_response_max(signal, time_axis, increase_ratio=0.05):
    """
    Event rule:
    response_max > baseline_min * (1 + increase_ratio)

    Used for pupil dilation.
    """
    signal = np.asarray(signal, dtype=float)
    time_axis = np.asarray(time_axis, dtype=float)

    baseline_mask, response_mask = get_time_masks(
        time_axis,
        baseline_start=baseline_start_sec,
        baseline_end=baseline_end_sec,
        response_start=response_start_sec,
        response_end=response_end_sec
    )

    if not np.any(baseline_mask) or not np.any(response_mask):
        return {
            "event_happened": False,
            "baseline_min": np.nan,
            "response_max": np.nan,
            "threshold_value": np.nan,
            "event_time_sec": np.nan,
            "event_frame": -1,
            "peak_time_sec": np.nan,
            "peak_frame": -1,
            "valid_trial": False
        }

    baseline_signal = signal[baseline_mask]
    response_signal = signal[response_mask]
    response_time = time_axis[response_mask]
    response_global_idx = np.where(response_mask)[0]

    if (not has_valid_values(baseline_signal)) or (not has_valid_values(response_signal)):
        return {
            "event_happened": False,
            "baseline_min": np.nan,
            "response_max": np.nan,
            "threshold_value": np.nan,
            "event_time_sec": np.nan,
            "event_frame": -1,
            "peak_time_sec": np.nan,
            "peak_frame": -1,
            "valid_trial": False
        }

    baseline_min = safe_nanmin(baseline_signal)
    response_max = safe_nanmax(response_signal)
    threshold_value = baseline_min * (1.0 + increase_ratio)

    peak_local_idx = safe_nanargmax(response_signal)
    if peak_local_idx >= 0:
        peak_time_sec = float(response_time[peak_local_idx])
        peak_frame = int(response_global_idx[peak_local_idx])
    else:
        peak_time_sec = np.nan
        peak_frame = -1

    event_happened = bool(response_max > threshold_value)

    crossing_idx = np.where(response_signal > threshold_value)[0]
    if len(crossing_idx) > 0:
        first_idx = int(crossing_idx[0])
        event_time_sec = float(response_time[first_idx])
        event_frame = int(response_global_idx[first_idx])
    else:
        event_time_sec = np.nan
        event_frame = -1

    return {
        "event_happened": event_happened,
        "baseline_min": baseline_min,
        "response_max": response_max,
        "threshold_value": threshold_value,
        "event_time_sec": event_time_sec,
        "event_frame": event_frame,
        "peak_time_sec": peak_time_sec,
        "peak_frame": peak_frame,
        "valid_trial": True
    }


def detect_event_by_baseline_mean_and_response_max(signal, time_axis, increase_ratio=0.50):
    """
    Event rule:
    response_max > baseline_mean * (1 + increase_ratio)

    Used for:
    - active locomotion
    - passive locomotion
    - eye movement
    """
    signal = np.asarray(signal, dtype=float)
    time_axis = np.asarray(time_axis, dtype=float)

    baseline_mask, response_mask = get_time_masks(
        time_axis,
        baseline_start=baseline_start_sec,
        baseline_end=baseline_end_sec,
        response_start=response_start_sec,
        response_end=response_end_sec
    )

    if not np.any(baseline_mask) or not np.any(response_mask):
        return {
            "event_happened": False,
            "baseline_mean": np.nan,
            "response_max": np.nan,
            "threshold_value": np.nan,
            "event_time_sec": np.nan,
            "event_frame": -1,
            "peak_time_sec": np.nan,
            "peak_frame": -1,
            "valid_trial": False
        }

    baseline_signal = signal[baseline_mask]
    response_signal = signal[response_mask]
    response_time = time_axis[response_mask]
    response_global_idx = np.where(response_mask)[0]

    if (not has_valid_values(baseline_signal)) or (not has_valid_values(response_signal)):
        return {
            "event_happened": False,
            "baseline_mean": np.nan,
            "response_max": np.nan,
            "threshold_value": np.nan,
            "event_time_sec": np.nan,
            "event_frame": -1,
            "peak_time_sec": np.nan,
            "peak_frame": -1,
            "valid_trial": False
        }

    baseline_mean = safe_nanmean(baseline_signal)
    response_max = safe_nanmax(response_signal)
    threshold_value = baseline_mean * (1.0 + increase_ratio)

    peak_local_idx = safe_nanargmax(response_signal)
    if peak_local_idx >= 0:
        peak_time_sec = float(response_time[peak_local_idx])
        peak_frame = int(response_global_idx[peak_local_idx])
    else:
        peak_time_sec = np.nan
        peak_frame = -1

    event_happened = bool(response_max > threshold_value)

    crossing_idx = np.where(response_signal > threshold_value)[0]
    if len(crossing_idx) > 0:
        first_idx = int(crossing_idx[0])
        event_time_sec = float(response_time[first_idx])
        event_frame = int(response_global_idx[first_idx])
    else:
        event_time_sec = np.nan
        event_frame = -1

    return {
        "event_happened": event_happened,
        "baseline_mean": baseline_mean,
        "response_max": response_max,
        "threshold_value": threshold_value,
        "event_time_sec": event_time_sec,
        "event_frame": event_frame,
        "peak_time_sec": peak_time_sec,
        "peak_frame": peak_frame,
        "valid_trial": True
    }


def detect_freezing_by_response_min(signal, time_axis, decrease_ratio=0.10):
    """
    Temporary freezing rule:
    response_min < baseline_mean * (1 - decrease_ratio)
    """
    signal = np.asarray(signal, dtype=float)
    time_axis = np.asarray(time_axis, dtype=float)

    baseline_mask, response_mask = get_time_masks(
        time_axis,
        baseline_start=baseline_start_sec,
        baseline_end=baseline_end_sec,
        response_start=response_start_sec,
        response_end=response_end_sec
    )

    if not np.any(baseline_mask) or not np.any(response_mask):
        return {
            "event_happened": False,
            "baseline_mean": np.nan,
            "response_min": np.nan,
            "threshold_value": np.nan,
            "event_time_sec": np.nan,
            "event_frame": -1,
            "trough_time_sec": np.nan,
            "trough_frame": -1,
            "valid_trial": False
        }

    baseline_signal = signal[baseline_mask]
    response_signal = signal[response_mask]
    response_time = time_axis[response_mask]
    response_global_idx = np.where(response_mask)[0]

    if (not has_valid_values(baseline_signal)) or (not has_valid_values(response_signal)):
        return {
            "event_happened": False,
            "baseline_mean": np.nan,
            "response_min": np.nan,
            "threshold_value": np.nan,
            "event_time_sec": np.nan,
            "event_frame": -1,
            "trough_time_sec": np.nan,
            "trough_frame": -1,
            "valid_trial": False
        }

    baseline_mean = safe_nanmean(baseline_signal)
    response_min = safe_nanmin(response_signal)
    threshold_value = baseline_mean * (1.0 - decrease_ratio)

    trough_local_idx = safe_nanargmin(response_signal)
    if trough_local_idx >= 0:
        trough_time_sec = float(response_time[trough_local_idx])
        trough_frame = int(response_global_idx[trough_local_idx])
    else:
        trough_time_sec = np.nan
        trough_frame = -1

    event_happened = bool(response_min < threshold_value)

    crossing_idx = np.where(response_signal < threshold_value)[0]
    if len(crossing_idx) > 0:
        first_idx = int(crossing_idx[0])
        event_time_sec = float(response_time[first_idx])
        event_frame = int(response_global_idx[first_idx])
    else:
        event_time_sec = np.nan
        event_frame = -1

    return {
        "event_happened": event_happened,
        "baseline_mean": baseline_mean,
        "response_min": response_min,
        "threshold_value": threshold_value,
        "event_time_sec": event_time_sec,
        "event_frame": event_frame,
        "trough_time_sec": trough_time_sec,
        "trough_frame": trough_frame,
        "valid_trial": True
    }


# =========================================================
# 3. Load JSON
# =========================================================
with open(json_path, "r") as f:
    records = json.load(f)


# =========================================================
# 4. Extract events
# =========================================================
rows = []

for rec in records:
    trial_id = rec["trial_id"]
    trial_index = rec["trial_index"]
    condition_label = rec["condition_label"]
    condition_name = rec["condition_name"]
    time_axis = rec["time_axis_sec"]

    aom = rec["signals"]["aom"]
    pom = rec["signals"]["pom"]
    avg_pupil = rec["signals"]["avg_pupil"]
    avg_eye = rec["signals"]["avg_eye"]

    pupil_result = detect_event_by_baseline_min_and_response_max(
        avg_pupil, time_axis, increase_ratio=0.05
    )
    eye_result = detect_event_by_baseline_mean_and_response_max(
        avg_eye, time_axis, increase_ratio=0.50
    )
    aom_result = detect_event_by_baseline_mean_and_response_max(
        aom, time_axis, increase_ratio=0.50
    )
    pom_result = detect_event_by_baseline_mean_and_response_max(
        pom, time_axis, increase_ratio=0.50
    )
    active_freezing_result = detect_freezing_by_response_min(
        aom, time_axis, decrease_ratio=0.10
    )
    passive_freezing_result = detect_freezing_by_response_min(
        pom, time_axis, decrease_ratio=0.10
    )

    row = {
        "trial_id": trial_id,
        "trial_index": trial_index,
        "condition_label": condition_label,
        "condition_name": condition_name,

        "pupil_valid_trial": pupil_result["valid_trial"],
        "eye_valid_trial": eye_result["valid_trial"],
        "aom_valid_trial": aom_result["valid_trial"],
        "pom_valid_trial": pom_result["valid_trial"],

        "pupil_dilation_happened": pupil_result["event_happened"],
        "pupil_baseline_min": pupil_result["baseline_min"],
        "pupil_response_max": pupil_result["response_max"],
        "pupil_dilation_threshold": pupil_result["threshold_value"],
        "pupil_dilation_time_sec": pupil_result["event_time_sec"],
        "pupil_peak_time_sec": pupil_result["peak_time_sec"],

        "eye_movement_happened": eye_result["event_happened"],
        "eye_baseline_mean": eye_result["baseline_mean"],
        "eye_response_max": eye_result["response_max"],
        "eye_movement_threshold": eye_result["threshold_value"],
        "eye_movement_time_sec": eye_result["event_time_sec"],
        "eye_peak_time_sec": eye_result["peak_time_sec"],

        "active_locomotion_happened": aom_result["event_happened"],
        "aom_baseline_mean": aom_result["baseline_mean"],
        "aom_response_max": aom_result["response_max"],
        "active_locomotion_threshold": aom_result["threshold_value"],
        "active_locomotion_time_sec": aom_result["event_time_sec"],
        "aom_peak_time_sec": aom_result["peak_time_sec"],

        "passive_locomotion_happened": pom_result["event_happened"],
        "pom_baseline_mean": pom_result["baseline_mean"],
        "pom_response_max": pom_result["response_max"],
        "passive_locomotion_threshold": pom_result["threshold_value"],
        "passive_locomotion_time_sec": pom_result["event_time_sec"],
        "pom_peak_time_sec": pom_result["peak_time_sec"],

        "active_freezing_happened": active_freezing_result["event_happened"],
        "active_freezing_time_sec": active_freezing_result["event_time_sec"],
        "aom_trough_time_sec": active_freezing_result["trough_time_sec"],

        "passive_freezing_happened": passive_freezing_result["event_happened"],
        "passive_freezing_time_sec": passive_freezing_result["event_time_sec"],
        "pom_trough_time_sec": passive_freezing_result["trough_time_sec"],
    }

    rows.append(row)


# =========================================================
# 5. Save CSV
# =========================================================
events_df = pd.DataFrame(rows)
events_df.to_csv(output_csv_path, index=False)

print("Done.")
print(f"Updated events summary saved to: {output_csv_path}")
print()
print(events_df.head())
print()
print("Invalid trial counts:")
print(events_df[["pupil_valid_trial", "eye_valid_trial", "aom_valid_trial", "pom_valid_trial"]].apply(lambda x: (~x).sum()))