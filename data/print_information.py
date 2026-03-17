import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats

# =========================================================
# 1. User settings
# =========================================================
csv_path = r"C:\Users\t54547fj\OneDrive - The University of Manchester\Documents\GitHub\Behaviour_signal_processing_AI\sample data\events_summary.csv"

output_txt_path = os.path.join(
    os.path.dirname(csv_path),
    "essential_statistical_summary.txt"
)

# Expected columns
event_cols_default = [
    "pupil_dilation_happened",
    "eye_movement_happened",
    "active_locomotion_happened",
    "passive_locomotion_happened",
    "active_freezing_happened",
    "passive_freezing_happened",
]

time_cols_default = [
    "pupil_dilation_time_sec",
    "eye_movement_time_sec",
    "active_locomotion_time_sec",
    "passive_locomotion_time_sec",
    "active_freezing_time_sec",
    "passive_freezing_time_sec",
]

pretty_names = {
    "pupil_dilation_happened": "Pupil dilation",
    "eye_movement_happened": "Eye movement",
    "active_locomotion_happened": "Active locomotion",
    "passive_locomotion_happened": "Passive locomotion",
    "active_freezing_happened": "Active freezing",
    "passive_freezing_happened": "Passive freezing",
    "pupil_dilation_time_sec": "Pupil dilation time",
    "eye_movement_time_sec": "Eye movement time",
    "active_locomotion_time_sec": "Active locomotion time",
    "passive_locomotion_time_sec": "Passive locomotion time",
    "active_freezing_time_sec": "Active freezing time",
    "passive_freezing_time_sec": "Passive freezing time",
}


# =========================================================
# 2. Helper functions
# =========================================================
def ensure_bool01(series):
    """Convert event columns to numeric 0/1 robustly."""
    if series.dtype == bool:
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "1": 1, "0": 0
    }
    return s.map(mapping).astype(float)


def sem(x):
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))


def format_float(x, ndigits=4):
    if pd.isna(x):
        return "NaN"
    return f"{x:.{ndigits}f}"


def safe_mannwhitneyu(x, y):
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    return stat, p


def safe_fisher_exact(success1, total1, success2, total2):
    if min(total1, total2) == 0:
        return np.nan, np.nan
    table = np.array([
        [success1, total1 - success1],
        [success2, total2 - success2]
    ], dtype=int)
    try:
        odds_ratio, p = stats.fisher_exact(table)
        return odds_ratio, p
    except Exception:
        return np.nan, np.nan


def build_sequence_for_row(row, time_cols_present):
    """Build event sequence based on non-NaN event times."""
    mapping = {
        "pupil_dilation_time_sec": "pupil",
        "eye_movement_time_sec": "eye",
        "active_locomotion_time_sec": "active",
        "passive_locomotion_time_sec": "passive",
        "active_freezing_time_sec": "active_freeze",
        "passive_freezing_time_sec": "passive_freeze",
    }

    pairs = []
    for col in time_cols_present:
        val = row.get(col, np.nan)
        if pd.notna(val):
            pairs.append((mapping.get(col, col), float(val)))

    if len(pairs) == 0:
        return "no_event"

    pairs = sorted(pairs, key=lambda x: x[1])
    return " -> ".join([x[0] for x in pairs])


def write_line(f, text=""):
    print(text)
    f.write(text + "\n")


# =========================================================
# 3. Load data
# =========================================================
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found:\n{csv_path}")

df = pd.read_csv(csv_path)

if "condition_name" not in df.columns:
    raise ValueError("The CSV must contain a 'condition_name' column.")

event_cols = [c for c in event_cols_default if c in df.columns]
time_cols = [c for c in time_cols_default if c in df.columns]

if len(event_cols) == 0:
    raise ValueError("No expected event occurrence columns were found.")

for col in event_cols:
    df[col] = ensure_bool01(df[col])

for col in time_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

condition_order = list(df["condition_name"].dropna().unique())

# Derived timing columns
derived_time_pairs = [
    ("pupil_dilation_time_sec", "eye_movement_time_sec", "pupil_minus_eye"),
    ("pupil_dilation_time_sec", "active_locomotion_time_sec", "pupil_minus_active"),
    ("pupil_dilation_time_sec", "passive_locomotion_time_sec", "pupil_minus_passive"),
    ("passive_locomotion_time_sec", "active_locomotion_time_sec", "passive_minus_active"),
    ("eye_movement_time_sec", "active_locomotion_time_sec", "eye_minus_active"),
]

derived_cols = []
for c1, c2, new_name in derived_time_pairs:
    if c1 in df.columns and c2 in df.columns:
        df[new_name] = df[c1] - df[c2]
        derived_cols.append(new_name)

# Sequence
if len(time_cols) > 0:
    df["sequence"] = df.apply(lambda row: build_sequence_for_row(row, time_cols), axis=1)
else:
    df["sequence"] = "no_timing_columns"


# =========================================================
# 4. Generate text summary
# =========================================================
with open(output_txt_path, "w", encoding="utf-8") as f:
    write_line(f, "=" * 80)
    write_line(f, "ESSENTIAL STATISTICAL SUMMARY")
    write_line(f, "=" * 80)
    write_line(f, f"Input CSV: {csv_path}")
    write_line(f, f"Total trials: {len(df)}")
    write_line(f, f"Conditions: {', '.join(condition_order)}")
    write_line(f)

    # -----------------------------------------------------
    # Section 1: Trial counts
    # -----------------------------------------------------
    write_line(f, "-" * 80)
    write_line(f, "1. TRIAL COUNTS BY CONDITION")
    write_line(f, "-" * 80)
    counts = df["condition_name"].value_counts().reindex(condition_order)
    for cond, n in counts.items():
        write_line(f, f"{cond}: {n}")
    write_line(f)

    # -----------------------------------------------------
    # Section 2: Event occurrence rates
    # -----------------------------------------------------
    write_line(f, "-" * 80)
    write_line(f, "2. EVENT OCCURRENCE RATES BY CONDITION")
    write_line(f, "-" * 80)
    for col in event_cols:
        write_line(f, f"{pretty_names.get(col, col)}:")
        for cond in condition_order:
            vals = df.loc[df["condition_name"] == cond, col].dropna()
            rate = vals.mean() if len(vals) > 0 else np.nan
            n_yes = int(vals.sum()) if len(vals) > 0 else 0
            n_total = len(vals)
            write_line(
                f,
                f"  {cond}: rate={format_float(rate, 4)} "
                f"({n_yes}/{n_total})"
            )
        write_line(f)

    # -----------------------------------------------------
    # Section 3: Event timing summary
    # -----------------------------------------------------
    if len(time_cols) > 0:
        write_line(f, "-" * 80)
        write_line(f, "3. EVENT TIMING SUMMARY BY CONDITION")
        write_line(f, "-" * 80)
        for col in time_cols:
            write_line(f, f"{pretty_names.get(col, col)}:")
            for cond in condition_order:
                vals = df.loc[df["condition_name"] == cond, col].dropna()
                write_line(
                    f,
                    f"  {cond}: "
                    f"n={len(vals)}, "
                    f"mean={format_float(vals.mean(), 4)}, "
                    f"median={format_float(vals.median(), 4)}, "
                    f"std={format_float(vals.std(ddof=1), 4)}, "
                    f"sem={format_float(sem(vals), 4)}"
                )
            write_line(f)

    # -----------------------------------------------------
    # Section 4: Derived timing differences
    # -----------------------------------------------------
    if len(derived_cols) > 0:
        write_line(f, "-" * 80)
        write_line(f, "4. DERIVED TIMING DIFFERENCES")
        write_line(f, "-" * 80)
        for col in derived_cols:
            write_line(f, f"{col}:")
            for cond in condition_order:
                vals = df.loc[df["condition_name"] == cond, col].dropna()
                write_line(
                    f,
                    f"  {cond}: "
                    f"n={len(vals)}, "
                    f"mean={format_float(vals.mean(), 4)}, "
                    f"median={format_float(vals.median(), 4)}, "
                    f"std={format_float(vals.std(ddof=1), 4)}, "
                    f"sem={format_float(sem(vals), 4)}"
                )
            write_line(f)

    # -----------------------------------------------------
    # Section 5: Most common sequences
    # -----------------------------------------------------
    write_line(f, "-" * 80)
    write_line(f, "5. MOST COMMON EVENT SEQUENCES BY CONDITION")
    write_line(f, "-" * 80)
    for cond in condition_order:
        seq_counts = df.loc[df["condition_name"] == cond, "sequence"].value_counts().head(10)
        write_line(f, f"{cond}:")
        for seq_name, n in seq_counts.items():
            write_line(f, f"  {seq_name}: {n}")
        write_line(f)

    # -----------------------------------------------------
    # Section 6: Correlation matrix
    # -----------------------------------------------------
    corr_candidates = [c for c in (time_cols + derived_cols) if c in df.columns]
    if len(corr_candidates) >= 2:
        write_line(f, "-" * 80)
        write_line(f, "6. CORRELATION MATRIX OF TIMING VARIABLES")
        write_line(f, "-" * 80)
        corr_df = df[corr_candidates].corr()
        write_line(f, corr_df.to_string(float_format=lambda x: f"{x:0.4f}"))
        write_line(f)

    # -----------------------------------------------------
    # Section 7: Event occurrence statistical tests
    # -----------------------------------------------------
    write_line(f, "-" * 80)
    write_line(f, "7. EVENT OCCURRENCE STATISTICAL TESTS (FISHER EXACT)")
    write_line(f, "-" * 80)
    for col in event_cols:
        write_line(f, f"{pretty_names.get(col, col)}:")
        for cond1, cond2 in combinations(condition_order, 2):
            g1 = df.loc[df["condition_name"] == cond1, col].dropna()
            g2 = df.loc[df["condition_name"] == cond2, col].dropna()

            success1, total1 = int(g1.sum()), len(g1)
            success2, total2 = int(g2.sum()), len(g2)

            odds_ratio, p = safe_fisher_exact(success1, total1, success2, total2)

            write_line(
                f,
                f"  {cond1} vs {cond2}: "
                f"rate1={format_float(g1.mean(), 4)}, "
                f"rate2={format_float(g2.mean(), 4)}, "
                f"odds_ratio={format_float(odds_ratio, 4)}, "
                f"p={format_float(p, 6)}"
            )
        write_line(f)

    # -----------------------------------------------------
    # Section 8: Timing statistical tests
    # -----------------------------------------------------
    if len(time_cols) > 0 or len(derived_cols) > 0:
        write_line(f, "-" * 80)
        write_line(f, "8. TIMING STATISTICAL TESTS (MANN-WHITNEY U)")
        write_line(f, "-" * 80)

        for col in time_cols + derived_cols:
            if col not in df.columns:
                continue
            write_line(f, f"{pretty_names.get(col, col)}:")
            for cond1, cond2 in combinations(condition_order, 2):
                g1 = df.loc[df["condition_name"] == cond1, col].dropna()
                g2 = df.loc[df["condition_name"] == cond2, col].dropna()

                stat, p = safe_mannwhitneyu(g1, g2)

                write_line(
                    f,
                    f"  {cond1} vs {cond2}: "
                    f"n1={len(g1)}, mean1={format_float(g1.mean(), 4)}, median1={format_float(g1.median(), 4)}; "
                    f"n2={len(g2)}, mean2={format_float(g2.mean(), 4)}, median2={format_float(g2.median(), 4)}; "
                    f"U={format_float(stat, 4)}, p={format_float(p, 6)}"
                )
            write_line(f)

    # -----------------------------------------------------
    # Section 9: Quick biological highlights
    # -----------------------------------------------------
    write_line(f, "-" * 80)
    write_line(f, "9. QUICK BIOLOGICAL HIGHLIGHTS")
    write_line(f, "-" * 80)

    # Highest event rate by condition
    for col in event_cols:
        group_means = df.groupby("condition_name")[col].mean()
        if len(group_means.dropna()) > 0:
            top_cond = group_means.idxmax()
            top_val = group_means.max()
            write_line(
                f,
                f"Highest {pretty_names.get(col, col).lower()} rate: "
                f"{top_cond} ({format_float(top_val, 4)})"
            )

    # Earliest mean timing by condition
    for col in time_cols:
        group_means = df.groupby("condition_name")[col].mean().dropna()
        if len(group_means) > 0:
            earliest_cond = group_means.idxmin()
            earliest_val = group_means.min()
            write_line(
                f,
                f"Earliest {pretty_names.get(col, col).lower()}: "
                f"{earliest_cond} ({format_float(earliest_val, 4)} s)"
            )

    if "passive_minus_active" in df.columns:
        group_means = df.groupby("condition_name")["passive_minus_active"].mean().dropna()
        if len(group_means) > 0:
            write_line(f)
            write_line(f, "Interpretation of passive_minus_active:")
            for cond, val in group_means.items():
                direction = "passive earlier than active" if val < 0 else "active earlier than passive"
                write_line(f, f"  {cond}: mean={format_float(val, 4)} s ({direction})")

    write_line(f)
    write_line(f, "=" * 80)
    write_line(f, f"Summary saved to: {output_txt_path}")
    write_line(f, "=" * 80)