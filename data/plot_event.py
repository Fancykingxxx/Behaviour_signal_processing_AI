import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================
# 1. User settings
# =========================================================
csv_path = r"C:\Users\t54547fj\OneDrive - The University of Manchester\Documents\GitHub\Behaviour_signal_processing_AI\sample data\events_summary.csv"

output_dir = os.path.join(os.path.dirname(csv_path), "event_statistics_outputs")
os.makedirs(output_dir, exist_ok=True)

# Whether to save figures
save_figures = True

# Whether to show figures interactively
show_figures = True

# Event / timing columns expected
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

# Pretty labels for plotting
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
def sem(x):
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))


def ensure_bool01(series):
    """
    Convert event columns robustly to numeric 0/1 when possible.
    """
    if series.dtype == bool:
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "1": 1, "0": 0
    }
    return s.map(mapping).astype(float)


def safe_mannwhitneyu(x, y):
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    return stat, p


def safe_fisher_or_chi2(success1, total1, success2, total2):
    """
    Use Fisher exact for 2x2 occurrence comparison.
    """
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
    """
    Build event sequence string based on available non-NaN times.
    """
    event_pairs = []
    mapping = {
        "pupil_dilation_time_sec": "pupil",
        "eye_movement_time_sec": "eye",
        "active_locomotion_time_sec": "active",
        "passive_locomotion_time_sec": "passive",
        "active_freezing_time_sec": "active_freeze",
        "passive_freezing_time_sec": "passive_freeze",
    }

    for col in time_cols_present:
        val = row.get(col, np.nan)
        if pd.notna(val):
            event_pairs.append((mapping.get(col, col), float(val)))

    if len(event_pairs) == 0:
        return "no_event"

    event_pairs = sorted(event_pairs, key=lambda x: x[1])
    return " -> ".join([x[0] for x in event_pairs])


def save_figure(fig, filename):
    if save_figures:
        fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")


# =========================================================
# 3. Load data
# =========================================================
df = pd.read_csv(csv_path)

if "condition_name" not in df.columns:
    raise ValueError("The CSV must contain a 'condition_name' column.")

# Detect available columns
event_cols = [c for c in event_cols_default if c in df.columns]
time_cols = [c for c in time_cols_default if c in df.columns]

if len(event_cols) == 0:
    raise ValueError("No expected event occurrence columns were found.")

if len(time_cols) == 0:
    print("Warning: No expected timing columns were found. Timing plots will be skipped.")

# Clean event columns
for col in event_cols:
    df[col] = ensure_bool01(df[col])

# Clean time columns
for col in time_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Optional condition ordering
condition_order = list(df["condition_name"].dropna().unique())


# =========================================================
# 4. Summary statistics tables
# =========================================================
# 4.1 Event rate summary
event_rate_summary = (
    df.groupby("condition_name")[event_cols]
    .agg(["mean", "sum", "count"])
)

event_rate_summary.to_csv(os.path.join(output_dir, "event_rate_summary.csv"))

# 4.2 Timing summary
if len(time_cols) > 0:
    timing_summary = (
        df.groupby("condition_name")[time_cols]
        .agg(["mean", "median", "std", sem, "count"])
    )
    timing_summary.to_csv(os.path.join(output_dir, "event_timing_summary.csv"))

# 4.3 Sequence summary
df["sequence"] = df.apply(lambda row: build_sequence_for_row(row, time_cols), axis=1)
sequence_summary = (
    df.groupby("condition_name")["sequence"]
    .value_counts()
    .rename("count")
    .reset_index()
)
sequence_summary.to_csv(os.path.join(output_dir, "sequence_summary.csv"), index=False)


# =========================================================
# 5. Plot 1: Event occurrence rate by condition
# =========================================================
rate_df = df.groupby("condition_name")[event_cols].mean().reindex(condition_order)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(rate_df.index))
width = 0.12 if len(event_cols) >= 5 else 0.18

for i, col in enumerate(event_cols):
    ax.bar(
        x + (i - (len(event_cols)-1)/2) * width,
        rate_df[col].values,
        width=width,
        label=pretty_names.get(col, col)
    )

ax.set_xticks(x)
ax.set_xticklabels(rate_df.index, rotation=20)
ax.set_ylabel("Event probability")
ax.set_title("Event occurrence rate by condition")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False)
fig.tight_layout()

save_figure(fig, "plot_01_event_occurrence_rate_by_condition.png")
if show_figures:
    plt.show()
else:
    plt.close(fig)


# =========================================================
# 6. Plot 2: Event onset times by condition (boxplots)
# =========================================================
if len(time_cols) > 0:
    n_plots = len(time_cols)
    fig, axes = plt.subplots(
        nrows=int(np.ceil(n_plots / 2)),
        ncols=2,
        figsize=(12, 4 * int(np.ceil(n_plots / 2)))
    )
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, time_cols):
        data_to_plot = [
            df.loc[df["condition_name"] == cond, col].dropna().values
            for cond in condition_order
        ]

        ax.boxplot(data_to_plot, tick_labels=condition_order, showfliers=False)
        ax.set_title(pretty_names.get(col, col))
        ax.set_ylabel("Time (s)")
        ax.tick_params(axis='x', rotation=20)

    # Hide unused axes
    for ax in axes[len(time_cols):]:
        ax.axis("off")

    fig.suptitle("Event onset times by condition", y=0.995)
    fig.tight_layout()

    save_figure(fig, "plot_02_event_onset_times_by_condition.png")
    if show_figures:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# 7. Plot 3: Mean event time by condition
# =========================================================
if len(time_cols) > 0:
    mean_time_df = df.groupby("condition_name")[time_cols].mean().reindex(condition_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(mean_time_df.index))
    width = 0.12 if len(time_cols) >= 5 else 0.18

    for i, col in enumerate(time_cols):
        ax.bar(
            x + (i - (len(time_cols)-1)/2) * width,
            mean_time_df[col].values,
            width=width,
            label=pretty_names.get(col, col)
        )

    ax.set_xticks(x)
    ax.set_xticklabels(mean_time_df.index, rotation=20)
    ax.set_ylabel("Mean onset time (s)")
    ax.set_title("Mean event onset time by condition")
    ax.legend(frameon=False)
    fig.tight_layout()

    save_figure(fig, "plot_03_mean_event_onset_time_by_condition.png")
    if show_figures:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# 8. Derived timing differences
# =========================================================
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

if len(derived_cols) > 0:
    # Summary
    derived_summary = (
        df.groupby("condition_name")[derived_cols]
        .agg(["mean", "median", "std", sem, "count"])
    )
    derived_summary.to_csv(os.path.join(output_dir, "derived_timing_difference_summary.csv"))

    # Plot
    fig, axes = plt.subplots(
        nrows=int(np.ceil(len(derived_cols) / 2)),
        ncols=2,
        figsize=(12, 4 * int(np.ceil(len(derived_cols) / 2)))
    )
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, derived_cols):
        data_to_plot = [
            df.loc[df["condition_name"] == cond, col].dropna().values
            for cond in condition_order
        ]
        ax.boxplot(data_to_plot, tick_labels=condition_order, showfliers=False)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(col.replace("_", " "))
        ax.set_ylabel("Time difference (s)")
        ax.tick_params(axis='x', rotation=20)

    for ax in axes[len(derived_cols):]:
        ax.axis("off")

    fig.suptitle("Derived timing differences by condition", y=0.995)
    fig.tight_layout()

    save_figure(fig, "plot_04_derived_timing_differences.png")
    if show_figures:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# 9. Plot 5: Sequence counts by condition
# =========================================================
top_n_sequences = 8
top_sequences = df["sequence"].value_counts().head(top_n_sequences).index.tolist()

seq_plot_df = (
    df[df["sequence"].isin(top_sequences)]
    .groupby(["condition_name", "sequence"])
    .size()
    .unstack(fill_value=0)
    .reindex(condition_order)
)

if seq_plot_df.shape[1] > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    seq_plot_df.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Number of trials")
    ax.set_title(f"Top {top_n_sequences} event sequences by condition")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    save_figure(fig, "plot_05_top_sequences_by_condition.png")
    if show_figures:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# 10. Plot 6: Correlation heatmap for timing variables
# =========================================================
corr_candidates = time_cols + derived_cols
corr_candidates = [c for c in corr_candidates if c in df.columns]

if len(corr_candidates) >= 2:
    corr_df = df[corr_candidates].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_df.values, aspect="auto")
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_xticklabels([pretty_names.get(c, c) for c in corr_df.columns], rotation=45, ha="right")
    ax.set_yticklabels([pretty_names.get(c, c) for c in corr_df.index])
    ax.set_title("Correlation matrix of timing variables")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add correlation text
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center")

    fig.tight_layout()

    save_figure(fig, "plot_06_timing_correlation_heatmap.png")
    if show_figures:
        plt.show()
    else:
        plt.close(fig)

    corr_df.to_csv(os.path.join(output_dir, "timing_correlation_matrix.csv"))


# =========================================================
# 11. Statistical tests
# =========================================================
stats_rows = []

# 11.1 Event occurrence tests between conditions
for col in event_cols:
    for cond1, cond2 in combinations(condition_order, 2):
        g1 = df.loc[df["condition_name"] == cond1, col].dropna()
        g2 = df.loc[df["condition_name"] == cond2, col].dropna()

        success1 = int(g1.sum())
        total1 = int(len(g1))
        success2 = int(g2.sum())
        total2 = int(len(g2))

        odds_ratio, p = safe_fisher_or_chi2(success1, total1, success2, total2)

        stats_rows.append({
            "analysis_type": "event_occurrence",
            "variable": col,
            "condition_1": cond1,
            "condition_2": cond2,
            "n1": total1,
            "n2": total2,
            "mean_1": g1.mean() if total1 > 0 else np.nan,
            "mean_2": g2.mean() if total2 > 0 else np.nan,
            "effect_direction": f"{cond1}>{cond2}" if g1.mean() > g2.mean() else f"{cond2}>{cond1}",
            "test": "fisher_exact",
            "statistic": odds_ratio,
            "p_value": p
        })

# 11.2 Event timing tests between conditions
for col in time_cols + derived_cols:
    if col not in df.columns:
        continue
    for cond1, cond2 in combinations(condition_order, 2):
        g1 = df.loc[df["condition_name"] == cond1, col].dropna()
        g2 = df.loc[df["condition_name"] == cond2, col].dropna()

        stat, p = safe_mannwhitneyu(g1, g2)

        stats_rows.append({
            "analysis_type": "timing",
            "variable": col,
            "condition_1": cond1,
            "condition_2": cond2,
            "n1": len(g1),
            "n2": len(g2),
            "mean_1": g1.mean() if len(g1) > 0 else np.nan,
            "mean_2": g2.mean() if len(g2) > 0 else np.nan,
            "median_1": g1.median() if len(g1) > 0 else np.nan,
            "median_2": g2.median() if len(g2) > 0 else np.nan,
            "effect_direction": f"{cond1}>{cond2}" if g1.mean() > g2.mean() else f"{cond2}>{cond1}",
            "test": "mannwhitneyu",
            "statistic": stat,
            "p_value": p
        })

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(os.path.join(output_dir, "statistical_test_results.csv"), index=False)


# =========================================================
# 12. Console summary
# =========================================================
print("Done.")
print(f"Input CSV: {csv_path}")
print(f"Output folder: {output_dir}")
print()
print("Available event columns:")
print(event_cols)
print()
print("Available timing columns:")
print(time_cols)
print()
print("Saved files:")
for fn in sorted(os.listdir(output_dir)):
    print(f" - {fn}")