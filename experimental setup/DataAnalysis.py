import argparse
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import statsmodels.api as sm  # noqa: F401  # imported for MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multitest import multipletests


# Configure matplotlib global settings
plt.rcParams["figure.dpi"] = 140


def ensure_outdir(p: Path) -> None:
    """Ensure that the output directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def cronbach_alpha(df: pd.DataFrame) -> float:
    """Compute Cronbach's alpha for a set of item columns.
    A higher alpha (≥ 0.7) suggests acceptable internal consistency. If the
    DataFrame is empty or has fewer than two columns, returns NaN.
    """
    if df is None or df.empty or df.shape[1] < 2:
        return np.nan
    # Drop any rows with missing values for reliability calculation
    clean = df.dropna(axis=0, how="any")
    k = clean.shape[1]
    if k < 2 or clean.empty:
        return np.nan
    item_vars = clean.var(axis=0, ddof=1)
    total_var = clean.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return np.nan
    return float((k / (k - 1)) * (1 - item_vars.sum() / total_var))


def partial_eta_squared(aov: pd.DataFrame, effect_row: str) -> float:
    """Compute partial eta squared from an ANOVA table for the specified effect."""
    if aov is None or aov.empty:
        return np.nan
    if effect_row not in aov.index or "Residual" not in aov.index:
        return np.nan
    ss_effect = aov.loc[effect_row, "sum_sq"]
    ss_error = aov.loc["Residual", "sum_sq"]
    if (ss_effect + ss_error) <= 0:
        return np.nan
    return float(ss_effect / (ss_effect + ss_error))


def holm_adjust(pvals: List[float]) -> List[float]:
    """Apply Holm correction to a list of p‑values and return adjusted values."""
    return multipletests(pvals, method="holm")[1].tolist()


def try_float(x) -> float:
    """Safely convert a value to float; return NaN if conversion fails."""
    try:
        return float(x)
    except Exception:
        return np.nan


def load_qualtrics_files(files: List[Path], group_labels: List[str]) -> pd.DataFrame:

    frames = []
    for f, label in zip(files, group_labels):
        df = pd.read_csv(f)
        # Add group label before dropping rows; this ensures it is available
        df["__batch_group__"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def get_rank_columns(df_raw: pd.DataFrame) -> List[str]:

    rank_cols: List[str] = []
    for c in df_raw.columns:
        if isinstance(c, str) and re.match(r"^q39_\d+$", c.strip().lower()):
            rank_cols.append(c)
    if rank_cols:
        return rank_cols
    if not df_raw.empty:
        row0 = df_raw.iloc[0]
        for c in df_raw.columns:
            txt = str(row0.get(c, "")).lower()
            if "rank the code documentation" in txt:
                rank_cols.append(c)
    return rank_cols


# ---- Column detection heuristics ----
def _siblings_for_prefix(columns: List[str], base: str) -> List[str]:
    """Given a base prefix (e.g., 'Q10'), return sibling columns with suffixes '_1'..'_10'."""
    sibs: List[str] = []
    if base in columns:
        sibs.append(base)
    for i in range(1, 11):
        candidate = f"{base}_{i}"
        if candidate in columns:
            sibs.append(candidate)
    return sibs


def _numeric_1_5_fraction(series: pd.Series) -> float:
    """Compute the fraction of non‑NaN values in a series that are between 1 and 5 inclusive."""
    x = pd.to_numeric(series, errors="coerce")
    x = x.dropna()
    if x.empty:
        return 0.0
    valid = x.between(1, 5).sum()
    return float(valid) / float(len(x))


def _structure_based_blocks(df: pd.DataFrame, min_block: int = 3, max_block: int = 8) -> Dict[str, List[str]]:
    blocks: Dict[str, List[str]] = {}
    cols = [c for c in df.columns if isinstance(c, str)]
    # Identify Qxx prefixes
    seen: set = set()
    for c in cols:
        m = re.match(r"^([A-Za-z]*\d+)(?:_[0-9]+)?$", c.strip())
        if not m:
            continue
        prefix = m.group(1)
        if prefix in seen:
            continue
        seen.add(prefix)
        sibs = _siblings_for_prefix(cols, prefix)
        if len(sibs) < min_block or len(sibs) > max_block:
            continue
        # Convert each sibling series to numeric separately; DataFrame conversion is not allowed
        arr = df[sibs].apply(pd.to_numeric, errors="coerce")
        # Evaluate the fraction of valid Likert responses per column
        fractions = [
            _numeric_1_5_fraction(arr[s]) for s in sibs
        ]
        if np.mean(fractions) >= 0.6:  # At least 60 % of values in 1..5
            blocks[prefix] = sibs
    return blocks


def detect_primary_metric_blocks(df_raw: pd.DataFrame, df_data: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, List[str]]]:

    cols = [c for c in df_raw.columns if isinstance(c, str)]
    if not df_raw.empty:
        row0 = df_raw.iloc[0]
    else:
        row0 = pd.Series(dtype=object)

    # Keyword lists for semantic matching (lowercased)
    acc_keywords = [
        "accur", "correct", "juist", "nauwkeur", "precis", "foutloos",
        "klopt", "betrouwbaar", "nauwkeurigheid", "juistheid"
    ]
    und_keywords = [
        "understand", "begrijp", "duidel", "leesbaar", "helder",
        "ease", "easy", "perceive",
        "gemakkelijk te begrijpen", "clarity", "begrijpelijkheid"
    ]
    comp_keywords = [
        "complete", "volledig", "volledigheid", "alles", "coverage", "compleet"
    ]
    halluc_keywords = [
        "halluc", "verzonnen", "off-topic", "irrelevant", "fabric", "niet relevant"
    ]

    def match_by_keywords(keywords: List[str]) -> List[str]:
        """Find columns whose question text contains any of the given keywords."""
        hits: List[str] = []
        for c in cols:
            txt = str(row0.get(c, "")).lower()
            if any(k in txt for k in keywords):
                # Expand to sibling block by prefix
                m = re.match(r"^([A-Za-z]*\d+)(?:_[0-9]+)?$", c.strip())
                if m:
                    sibs = _siblings_for_prefix(cols, m.group(1))
                    hits.extend(sibs if sibs else [c])
                else:
                    hits.append(c)
        # Deduplicate while preserving order
        seen_set: set = set()
        unique_hits: List[str] = []
        for h in hits:
            if h not in seen_set:
                unique_hits.append(h)
                seen_set.add(h)
        return unique_hits

    # Semantic matches
    acc_cols = match_by_keywords(acc_keywords)
    und_cols = match_by_keywords(und_keywords)
    comp_cols = match_by_keywords(comp_keywords)
    nh_cols = match_by_keywords(halluc_keywords)

    # If failed to find one or both primary metrics, fall back to structure
    if not acc_cols or not und_cols:
        blocks = _structure_based_blocks(df_data)
        if blocks:
            # Score each block by average variance of its columns; choose top two
            block_scores: List[Tuple[str, float]] = []
            for pref, sibs in blocks.items():
                arr = df_data[sibs].apply(pd.to_numeric, errors="coerce")
                # Compute mean variance across columns; skip NaN
                var_vals = arr.var(axis=0, ddof=1).dropna()
                avg_var = var_vals.mean() if not var_vals.empty else 0.0
                block_scores.append((pref, float(avg_var)))
            block_scores.sort(key=lambda t: t[1], reverse=True)
            # Pick top two blocks as candidates
            top_blocks = [b for b, _ in block_scores[:2]]
            if top_blocks:
                if not acc_cols and len(top_blocks) >= 1:
                    acc_cols = blocks[top_blocks[0]]
                if not und_cols and len(top_blocks) >= 2:
                    und_cols = blocks[top_blocks[1]]

    # Clean the lists: keep only actual columns in df_data and ensure they contain numeric data
    def clean(col_list: List[str]) -> List[str]:
        cleaned: List[str] = []
        for c in col_list:
            if c in df_data.columns:
                series = pd.to_numeric(df_data[c], errors="coerce")
                if series.notna().any():
                    cleaned.append(c)
        return cleaned

    acc_cols = clean(acc_cols)
    und_cols = clean(und_cols)
    comp_cols = clean(comp_cols)
    nh_cols = clean(nh_cols)

    secondary = {
        "completeness": comp_cols,
        "no_hallucination": nh_cols,
    }

    return acc_cols, und_cols, secondary


def manova_2dv_between_groups(df_av: pd.DataFrame, dv1: str, dv2: str, group: str) -> pd.DataFrame:
    """Perform MANOVA for two dependent variables with a single between‑subject factor."""
    sub = df_av[[dv1, dv2, group]].dropna()
    if sub.empty or sub[group].nunique() < 2:
        return pd.DataFrame()
    ma = MANOVA.from_formula(f"{dv1} + {dv2} ~ {group}", data=sub)
    res = ma.mv_test()
    stat = res.results[group]["stat"]
    out = stat[["F Value", "Pr > F"]].rename(columns={"F Value": "F", "Pr > F": "p"})
    out.index.name = "test"
    return out.reset_index()


def one_way_anova(df: pd.DataFrame, dv: str, group: str) -> Tuple[float, float, float]:
    """Perform a one‑way between‑subjects ANOVA and return (F, p, partial eta squared)."""
    sub = df[[dv, group]].dropna()
    if sub.empty or sub[group].nunique() < 2:
        return np.nan, np.nan, np.nan
    model = ols(f"{dv} ~ C({group})", data=sub).fit()
    aov = anova_lm(model, typ=2)
    effect_label = f"C({group})"
    if effect_label not in aov.index:
        return np.nan, np.nan, np.nan
    F = float(aov.loc[effect_label, "F"])
    p = float(aov.loc[effect_label, "PR(>F)"])
    pes = partial_eta_squared(aov, effect_label)
    return F, p, pes


def friedman_within_subjects(matrix: pd.DataFrame) -> Tuple[float, float]:
    """Perform Friedman’s test across k conditions (columns) within subjects (rows)."""
    mat = matrix.dropna(axis=0, how="any")
    if mat.empty or mat.shape[0] < 2 or mat.shape[1] < 2:
        return np.nan, np.nan
    stat, p = stats.friedmanchisquare(*[mat.iloc[:, i] for i in range(mat.shape[1])])
    return float(stat), float(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run statistical analysis on Qualtrics survey exports.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to Qualtrics CSV files")
    parser.add_argument("--outdir", required=True, help="Directory to write outputs")
    parser.add_argument("--time-file", default="", help="Optional CSV with generation time")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    # Determine group labels from filenames (anything containing 'V2' is V2, else Software)
    input_paths = [Path(p) for p in args.inputs]
    batch_labels = ["V2" if "V2" in p.name else "Software" for p in input_paths]

    # Load full raw data (including header rows) for semantic matching
    df_raw_all = load_qualtrics_files(input_paths, batch_labels)

    # Copy raw metadata (used for semantic detection). We'll drop the first two rows later.
    df_raw_meta = df_raw_all.copy()

    # Drop the two Qualtrics metadata rows for analysis
    df_data = df_raw_all.iloc[2:].reset_index(drop=True).copy()
    # Move the batch group to proper column name
    df_data.rename(columns={"__batch_group__": "batch_group"}, inplace=True)

    # Assign participant IDs if not present
    if "ResponseId" in df_data.columns:
        df_data["participant_id"] = df_data["ResponseId"]
    else:
        df_data["participant_id"] = np.arange(len(df_data))

    # Detect metric columns (primary and secondary)
    acc_cols, und_cols, secondary_cols = detect_primary_metric_blocks(df_raw_meta, df_data)

    # Convert candidate columns to numeric
    df_num = df_data.copy()
    for c in acc_cols + und_cols:
        if c in df_num.columns:
            df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
    for col_list in secondary_cols.values():
        for c in col_list:
            if c in df_num.columns:
                df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    # Build participant‑level averages for primary metrics
    df_av = pd.DataFrame({
        "participant_id": df_data["participant_id"],
        "group": df_data["batch_group"],
        "M1_accuracy": df_num[acc_cols].mean(axis=1, skipna=True),
        "M2_understandability": df_num[und_cols].mean(axis=1, skipna=True),
    })
    # Drop rows where both metrics are NaN (participant missing data)
    df_av = df_av.dropna(subset=["M1_accuracy", "M2_understandability"], how="all")

    # Reliability assessment
    irr_results = {
        "alpha_accuracy": cronbach_alpha(df_num[acc_cols]) if acc_cols else np.nan,
        "alpha_understandability": cronbach_alpha(df_num[und_cols]) if und_cols else np.nan,
    }
    pd.DataFrame([irr_results]).to_csv(outdir / "irr_cronbach.csv", index=False)

    # MANOVA across primary metrics by group
    manova_tbl = manova_2dv_between_groups(df_av, "M1_accuracy", "M2_understandability", "group")
    if not manova_tbl.empty:
        manova_tbl.to_csv(outdir / "manova_primary.csv", index=False)

    # Univariate ANOVAs for each primary metric
    anova_rows = []
    for dv in ["M1_accuracy", "M2_understandability"]:
        F, p, pes = one_way_anova(df_av, dv, "group")
        anova_rows.append({"dv": dv, "F": F, "p": p, "partial_eta2": pes})
    anova_df = pd.DataFrame(anova_rows)
    if not anova_df.empty:
        anova_df["p_holm"] = holm_adjust(anova_df["p"].fillna(1).tolist())
        anova_df.to_csv(outdir / "anova_primary.csv", index=False)

    # Secondary metrics (exploratory)
    secondary_results: List[Dict[str, float]] = []
    for sec_name, cols in secondary_cols.items():
        if not cols:
            continue
        # Compute per‑participant average for this secondary metric
        sec_mean_col = f"sec_{sec_name}"
        df_av[sec_mean_col] = df_num[cols].mean(axis=1, skipna=True)
        F, p, pes = one_way_anova(df_av.rename(columns={sec_mean_col: "dvtmp"}), "dvtmp", "group")
        secondary_results.append({"dv": sec_name, "F": F, "p": p, "partial_eta2": pes})
    if secondary_results:
        sec_df = pd.DataFrame(secondary_results)
        sec_df["p_holm"] = holm_adjust(sec_df["p"].fillna(1).tolist())
        sec_df.to_csv(outdir / "anova_secondary.csv", index=False)

    # Ranking analyses
    rank_cols = get_rank_columns(df_raw_meta)
    # Fallback: also recognise columns starting with q39_ in the cleaned data
    if not rank_cols:
        rank_cols = [c for c in df_data.columns if re.match(r"^q?39_\d+$", c.strip().lower())]

    if rank_cols:
        # Coerce ranking columns to numeric
        for c in rank_cols:
            if c in df_num.columns:
                df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

        # Friedman test across files (within participants)
        rank_matrix = (
            df_num[["participant_id"] + rank_cols]
            .dropna(subset=rank_cols, how="any")
            .drop_duplicates(subset=["participant_id"])
        )
        rank_summary: List[Dict[str, float]] = []
        if not rank_matrix.empty and rank_matrix["participant_id"].nunique() >= 2:
            chi2, p_fr = friedman_within_subjects(rank_matrix[rank_cols])
            rank_summary.append({"test": "Friedman_all_files", "chi2": chi2, "p": p_fr})

        # Between‑group Mann–Whitney per file
        mw_rows: List[Dict[str, float]] = []
        for c in rank_cols:
            # Compare groups only if both groups have at least two participants with data
            a = df_num.loc[df_num["batch_group"] == "Software", c].dropna().apply(try_float)
            b = df_num.loc[df_num["batch_group"] == "V2", c].dropna().apply(try_float)
            if len(a) >= 2 and len(b) >= 2:
                U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                mw_rows.append({"file_item": c, "U": float(U), "p": float(p)})
        if mw_rows:
            mw_df = pd.DataFrame(mw_rows)
            mw_df["p_holm"] = holm_adjust(mw_df["p"].tolist())
            mw_df.to_csv(outdir / "rank_between_groups.csv", index=False)
            rank_summary.append({
                "test": "MW_per_file",
                "n_files": len(mw_rows),
                "any_sig_uncorrected": bool((mw_df["p"] < 0.05).any()),
                "any_sig_holm": bool((mw_df["p_holm"] < 0.05).any()),
            })
        if rank_summary:
            pd.DataFrame(rank_summary).to_csv(outdir / "rank_summary.csv", index=False)

        # Plot average rank per file (lower is better)
        # Compute by group (Software vs V2)
        rank_cols_numeric = [c for c in rank_cols if c in df_num.columns]
        avg_rank = (
            df_num.groupby("batch_group")[rank_cols_numeric]
            .mean(numeric_only=True)
        )
        if not avg_rank.empty and avg_rank.shape[1] > 0:
            # Build a mapping from ranking column to human‑readable file label using the question text.
            rank_labels: Dict[str, str] = {}
            if not df_raw_meta.empty:
                qtext_row = df_raw_meta.iloc[0]
                for c in rank_cols_numeric:
                    raw_txt = str(qtext_row.get(c, ""))
                    # Attempt to extract the file letter after "File "
                    m = re.search(r"-\s*File\s*([A-Z])", raw_txt)
                    if m:
                        rank_labels[c] = f"File {m.group(1)}"
                    else:
                        rank_labels[c] = c
            # Rename index of avg_rank_T with human labels for readability
            avg_rank_T = avg_rank.T
            if rank_labels:
                avg_rank_T = avg_rank_T.rename(index=rank_labels)
            fig, ax = plt.subplots(figsize=(7, 3.6))
            avg_rank_T.plot(kind="bar", ax=ax)
            ax.set_title("Average rank per file (lower is better)")
            ax.set_xlabel("Documentation file")
            ax.set_ylabel("Average rank")
            fig.tight_layout()
            fig.savefig(outdir / "ranking_bar.png", dpi=200)
        else:
            print("Skipping ranking_bar: no numeric ranking data available after cleaning.")
    else:
        print("Skipping ranking analyses: no ranking columns detected.")

    # --------- Plots for primary metrics ---------
    def mean_ci(arr: pd.Series) -> Tuple[float, float, float]:
        x = pd.Series(arr).dropna()
        if x.empty:
            return np.nan, np.nan, np.nan
        m = x.mean()
        se = x.std(ddof=1) / max(1, np.sqrt(len(x)))
        ci_half = stats.t.ppf(0.975, df=max(1, len(x) - 1)) * se
        return m, m - ci_half, m + ci_half

    # Prepare data for mean + CI plot
    mean_rows: List[Dict[str, object]] = []
    for dv, label in [("M1_accuracy", "Accuracy"), ("M2_understandability", "Understandability")]:
        for group_name, group_data in df_av.groupby("group"):
            m, lo, hi = mean_ci(group_data[dv])
            if not np.isnan(m):
                mean_rows.append({
                    "Metric": label,
                    "Group": group_name,
                    "Mean": m,
                    "Lo": lo,
                    "Hi": hi,
                })
    mean_df = pd.DataFrame(mean_rows)
    if not mean_df.empty:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        metrics = mean_df["Metric"].unique().tolist()
        groups = mean_df["Group"].unique().tolist()
        width = 0.35
        for i, gname in enumerate(groups):
            seg = mean_df[mean_df["Group"] == gname]
            x = np.arange(len(metrics)) + i * width
            ax.bar(x, seg["Mean"].values, width=width, label=gname)
            ax.vlines(x, seg["Lo"].values, seg["Hi"].values, lw=2)
        ax.set_xticks(np.arange(len(metrics)) + width / 2)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Mean (95% CI)")
        ax.set_title("Primary metrics by group")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / "primary_means_ci.png", dpi=200)
    else:
        print("Skipping primary_means_ci: no non-NaN values available for plotting.")

    # Boxplots per metric
    valid = df_av.dropna(subset=["M1_accuracy", "M2_understandability"], how="all")
    if not valid.empty:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=False)
        for ax, dv, title in zip(
            axes,
            ["M1_accuracy", "M2_understandability"],
            ["M1_accuracy", "M2_understandability"],
        ):
            groups = sorted(valid["group"].dropna().unique())
            data = [valid.loc[valid["group"] == g, dv].dropna().values for g in groups]
            if any(len(d) > 0 for d in data):
                ax.boxplot(data, labels=groups)
                ax.set_title(title)
                ax.set_xlabel("group")
                ax.set_ylabel("Score")
            else:
                ax.set_title(f"{title} (no data)")
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(outdir / "primary_boxplots.png", dpi=200)
    else:
        print("Skipping primary_boxplots: no data after cleaning.")


    if len(acc_cols) >= 2 and len(und_cols) >= 2:

        acc_sorted = sorted(acc_cols)
        und_sorted = sorted(und_cols)

        # Prepare per‑file mean table by group
        perfile_acc = df_num.groupby("batch_group")[acc_sorted].mean(numeric_only=True)
        perfile_und = df_num.groupby("batch_group")[und_sorted].mean(numeric_only=True)
        file_labels = None
        # Build mapping from ranking Q39_* columns to file label
        if rank_cols:
            # Get ranking labels from question text row
            qtext_row = df_raw_meta.iloc[0] if not df_raw_meta.empty else None
            if qtext_row is not None:
                labels = []
                for rc in sorted([c for c in rank_cols if c in df_raw_meta.columns]):
                    raw = str(qtext_row.get(rc, ""))
                    m = re.search(r"-\s*File\s*([A-Z])", raw)
                    if m:
                        labels.append(f"File {m.group(1)}")
                if len(labels) == len(acc_sorted):
                    file_labels = labels

        # If we have labels, apply them; else use QIDs
        acc_plot_names = file_labels if file_labels else acc_sorted
        und_plot_names = file_labels if file_labels else und_sorted

        # Plot per‑file accuracy means by group
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
        # Accuracy plot
        ax = axes[0]
        for gname in perfile_acc.index:
            ax.plot(acc_plot_names, perfile_acc.loc[gname].values, marker='o', label=gname)
        ax.set_title("Per‑file Accuracy (higher=better)")
        ax.set_xlabel("Documentation file")
        ax.set_ylabel("Mean score")
        ax.legend(frameon=False)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        # Understandability plot
        ax2 = axes[1]
        for gname in perfile_und.index:
            ax2.plot(und_plot_names, perfile_und.loc[gname].values, marker='o', label=gname)
        ax2.set_title("Per‑file Understandability (higher=better)")
        ax2.set_xlabel("Documentation file")
        ax2.set_ylabel("Mean score")
        ax2.legend(frameon=False)
        ax2.grid(axis='y', linestyle='--', alpha=0.4)
        fig.tight_layout()
        fig.savefig(outdir / "perfile_metrics.png", dpi=200)

        # Perform Friedman tests within each group (Software and V2) across files
        friedman_rows = []
        for metric_name, cols_sorted in [("accuracy", acc_sorted), ("understandability", und_sorted)]:
            for gname in df_data["batch_group"].unique():
                sub = df_num.loc[df_num["batch_group"] == gname, cols_sorted]
                # drop rows with any missing values across the file columns
                sub_clean = sub.dropna(axis=0, how="any")
                if sub_clean.shape[0] >= 2:
                    stat, p = friedman_within_subjects(sub_clean)
                    friedman_rows.append({
                        "metric": metric_name,
                        "group": gname,
                        "chi2": stat,
                        "p": p,
                    })
        if friedman_rows:
            pd.DataFrame(friedman_rows).to_csv(outdir / "friedman_within_group.csv", index=False)
    print(f"Done. Results saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()