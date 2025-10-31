import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from numpy.polynomial.polynomial import polyfit
from scipy import stats

# ============================================================
# Utility: annotate points on scatter
# ============================================================
def label_points(ax, x_vals, y_vals, labels):
    for x, y, lab in zip(x_vals, y_vals, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6))

# ============================================================
# Scatter plot with least-squares line and optional labels
# ============================================================
def plot_scatter(x_vals: List[float],
                 y_vals: List[float],
                 title: str,
                 x_label: str,
                 y_label: str,
                 filename: str,
                 labels: List[str] = None):
    fig, ax = plt.subplots()
    ax.scatter(x_vals, y_vals)
    if labels:
        label_points(ax, x_vals, y_vals, labels)

    # Least-squares fit line (degree 1)
    coefs = polyfit(x_vals, y_vals, 1)  # returns [intercept, slope]
    fit_y = coefs[0] + coefs[1] * np.array(x_vals)
    ax.plot(x_vals, fit_y)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    try:
        r, p = stats.pearsonr(x_vals, y_vals)
        ax.set_title(f"{title} (r={r:.2f}, p={p:.3f})")
    except Exception:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

# ============================================================
# Box plot helper: quality per respondent per model
# ============================================================
def boxplot_quality(ax,
                    per_model_per_respondent: Dict[str, List[float]],
                    title: str):
    labels = list(per_model_per_respondent.keys())
    data = [per_model_per_respondent[m] for m in labels]

    bp = ax.boxplot(data, labels=labels, showmeans=True, meanline=False)
    ax.set_ylabel("Quality Score")
    ax.set_title(title)

# ============================================================
# Box plot helper: generation times per run per model
# ============================================================
def boxplot_times(ax,
                  per_model_times: Dict[str, List[float]],
                  title: str):
    labels = list(per_model_times.keys())
    data = [per_model_times[m] for m in labels]
    bp = ax.boxplot(data, labels=labels, showmeans=True, meanline=False)
    ax.set_ylabel("Generation Time (s)")
    ax.set_title(title)

# ============================================================
# Load per-respondent quality distributions by model
# ============================================================
def load_quality_distributions(file_path: str) -> Dict[str, List[float]]:
    """
    Build per-respondent average quality for each model:
    A -> Claude few
    B -> GPT zero
    D -> Claude zero
    E -> GPT few
    Each respondent's score for a model is the mean across 4 aspects,
    each aspect composed of 4 Likert questions.

    Returns a dict: model -> list of respondent means.
    """
    df = pd.read_csv(file_path)
    df = df.iloc[2:].copy()

    aspects = [
        ["Q47_1", "Q48_1", "Q49_1", "Q50_1"],  # correctness
        ["Q51_1", "Q52_1", "Q53_1", "Q54_1"],  # ease
        ["Q55_1", "Q56_1", "Q57_1", "Q58_1"],  # completeness
        ["Q60_1", "Q61_1", "Q62_1", "Q64_1"],  # hallucination absence
    ]

    models = ["Claude few", "GPT zero", "Claude zero", "GPT few"]
    results = {m: [] for m in models}

    # For each respondent, compute model means
    for _, row in df.iterrows():
        # Per aspect, we get a vector of 4 values that map to A,B,D,E
        # We then aggregate across aspects per model.
        per_model_values = {m: [] for m in models}
        for aspect in aspects:
            vals = [pd.to_numeric(pd.Series([row[c]]), errors="coerce").values[0] for c in aspect]
            # vals order corresponds to A,B,D,E -> Claude few, GPT zero, Claude zero, GPT few
            per_model_values["Claude few"].append(vals[0])
            per_model_values["GPT zero"].append(vals[1])
            per_model_values["Claude zero"].append(vals[2])
            per_model_values["GPT few"].append(vals[3])

        # mean across the 4 aspects for each model for this respondent
        for m in models:
            arr = [v for v in per_model_values[m] if pd.notna(v)]
            if len(arr) > 0:
                results[m].append(float(np.mean(arr)))

    return results

# ============================================================
# Load mean quality by model (overall averages)
# ============================================================
def load_quality_means(file_path: str) -> Dict[str, float]:
    df = pd.read_csv(file_path)
    df = df.iloc[2:].copy()

    aspects = [
        ["Q47_1", "Q48_1", "Q49_1", "Q50_1"],
        ["Q51_1", "Q52_1", "Q53_1", "Q54_1"],
        ["Q55_1", "Q56_1", "Q57_1", "Q58_1"],
        ["Q60_1", "Q61_1", "Q62_1", "Q64_1"],
    ]
    models = ["Claude few", "GPT zero", "Claude zero", "GPT few"]
    per_model_scores = {m: [] for m in models}

    for aspect in aspects:
        cols = aspect
        vals = [pd.to_numeric(df[c], errors="coerce") for c in cols]
        for model_name, series in zip(models, vals):
            per_model_scores[model_name].append(series.mean())

    mean_scores = {m: float(np.nanmean(per_model_scores[m])) for m in models}
    return mean_scores

# ============================================================
# Parse generation times from TimesLLM.txt
# ============================================================
def parse_times(path: str) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, float]]]:
    """
    Returns:
      per_run_times: run -> model -> list of times
      per_run_means: run -> model -> mean time
    """
    text = open(path, "r", encoding="utf-8").read().splitlines()
    per_run_times: Dict[str, Dict[str, List[float]]] = {}
    current_run = None
    current_llm = None

    for raw in text:
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("run"):
            current_run = line
            per_run_times[current_run] = {}
            current_llm = None
            continue
        if current_run and (line.lower().startswith("gpt") or line.lower().startswith("claude")):
            model_name = line.split(":")[0].strip()
            current_llm = model_name
            per_run_times[current_run][current_llm] = []
            continue
        if line.lower().startswith("gem"):
            continue
        if current_run and current_llm:
            try:
                per_run_times[current_run][current_llm].append(float(line))
            except ValueError:
                pass

    per_run_means: Dict[str, Dict[str, float]] = {}
    for run, d in per_run_times.items():
        per_run_means[run] = {k: float(np.mean(v)) for k, v in d.items() if v}
    return per_run_times, per_run_means

# ============================================================
# Main analysis
# ============================================================
def run_analysis():
    data_dir = os.getcwd()

    # Map dataset names
    path_2048 = os.path.join(data_dir, "game_survey_binary.csv")
    path_sumtree = os.path.join(data_dir, "sumtree_survey_binary.csv")
    times_path = os.path.join(data_dir, "TimesLLM.txt")

    # Quality means for scatter
    quality_2048_means = load_quality_means(path_2048)
    quality_sumtree_means = load_quality_means(path_sumtree)

    # Quality distributions for box plots
    quality_2048_dists = load_quality_distributions(path_2048)
    quality_sumtree_dists = load_quality_distributions(path_sumtree)

    # Times per run and mean times
    per_run_times, per_run_means = parse_times(times_path)

    models = ["Claude few", "GPT zero", "Claude zero", "GPT few"]

    # Print summary means
    print("Per-model quality means (2048 / Sumtree):")
    for m in models:
        print(f"  {m}: {quality_2048_means[m]:.3f} / {quality_sumtree_means[m]:.3f}")

    # Average time across runs per model (for pooled scatter)
    avg_time_per_model = {}
    for m in models:
        vals = []
        for run, d in per_run_means.items():
            for k, v in d.items():
                if k.lower().replace(" ", "") == m.lower().replace(" ", ""):
                    vals.append(v)
        avg_time_per_model[m] = float(np.mean(vals)) if vals else np.nan

    print("\nAverage generation times across runs (s):")
    for m in models:
        print(f"  {m}: {avg_time_per_model[m]:.3f}")

    # Scatter: 2048
    run1 = "Run 1"
    run1_map = {k.lower().replace(" ", ""): v for k, v in per_run_means[run1].items()}
    times_2048 = [run1_map[m.lower().replace(" ", "")] for m in models]
    qualities_2048 = [quality_2048_means[m] for m in models]
    r1, p1 = stats.pearsonr(times_2048, qualities_2048)
    print(f"\n2048 correlation time-quality: r={r1:.3f}, p={p1:.3f}")

    plot_scatter(
        times_2048,
        qualities_2048,
        "2048: Generation Time vs Quality",
        "Generation Time (s)",
        "Quality Score",
        os.path.join(data_dir, "2048_scatter_labeled.png"),
        labels=models,
    )

    # Scatter: Sumtree
    run2 = "Run 2"
    run2_map = {k.lower().replace(" ", ""): v for k, v in per_run_means[run2].items()}
    times_sumtree = [run2_map[m.lower().replace(" ", "")] for m in models]
    qualities_sumtree = [quality_sumtree_means[m] for m in models]
    r2, p2 = stats.pearsonr(times_sumtree, qualities_sumtree)
    print(f"Sumtree correlation time-quality: r={r2:.3f}, p={p2:.3f}")

    plot_scatter(
        times_sumtree,
        qualities_sumtree,
        "Sumtree: Generation Time vs Quality",
        "Generation Time (s)",
        "Quality Score",
        os.path.join(data_dir, "Sumtree_scatter_labeled.png"),
        labels=models,
    )

    # Scatter: pooled (average time and average quality)
    pooled_t = [avg_time_per_model[m] for m in models]
    pooled_q = [(quality_2048_means[m] + quality_sumtree_means[m]) / 2.0 for m in models]
    rp, pp = stats.pearsonr(pooled_t, pooled_q)
    print(f"Pooled correlation time-quality: r={rp:.3f}, p={pp:.3f}")

    plot_scatter(
        pooled_t,
        pooled_q,
        "Pooled: Generation Time vs Quality",
        "Average Generation Time (s)",
        "Average Quality Score",
        os.path.join(data_dir, "Pooled_scatter_labeled.png"),
        labels=models,
    )

    # Box plots: quality, per dataset
    fig, ax = plt.subplots()
    boxplot_quality(ax, quality_2048_dists, "2048: Quality by Model (Box = quartiles, Triangle = mean)")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "2048_quality_box.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    boxplot_quality(ax, quality_sumtree_dists, "Sumtree: Quality by Model (Box = quartiles, Triangle = mean)")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "Sumtree_quality_box.png"), dpi=150)
    plt.close(fig)

    # Box plots: generation times, per run
    # Build per-model lists for each run, with labels in consistent order
    def normalize_time_lists(run_key: str) -> Dict[str, List[float]]:
        raw = per_run_times.get(run_key, {})
        out = {}
        for m in models:
            for key, vals in raw.items():
                if key.lower().replace(" ", "") == m.lower().replace(" ", ""):
                    out[m] = vals
                    break
            if m not in out:
                out[m] = []
        return out

    times_2048_lists = normalize_time_lists("Run 1")
    fig, ax = plt.subplots()
    boxplot_times(ax, times_2048_lists, "2048: Generation Time by Model (Box = quartiles, Triangle = mean)")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "2048_time_box.png"), dpi=150)
    plt.close(fig)

    times_sumtree_lists = normalize_time_lists("Run 2")
    fig, ax = plt.subplots()
    boxplot_times(ax, times_sumtree_lists, "Sumtree: Generation Time by Model (Box = quartiles, Triangle = mean)")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "Sumtree_time_box.png"), dpi=150)
    plt.close(fig)

    print("\nNotes:")
    print(" - r is Pearson's correlation: r = cov(X,Y) / (sd(X)*sd(Y))")
    print(" - p is from t = r*sqrt((n-2)/(1-r^2)), df = n-2 (two-sided test)")
    print(" - Box plots show median (line), quartiles (box), whiskers, and mean (dot).")

if __name__ == "__main__":
    run_analysis()
