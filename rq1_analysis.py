# file: analyse_llm_survey.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
from typing import Dict


metrics_dict = {
    "accuracy": ["Q48_1", "Q47_1", "Q49_1", "Q18_1", "Q50_1"],
    "understandability": ["Q53_1", "Q54_1", "Q20_1", "Q52_1", "Q51_1"],
    "completeness": ["Q58_1", "Q57_1", "Q56_1", "Q45_1", "Q55_1"],
    "relevance": ["Q60_1", "Q64_1", "QID63_1", "Q62_1", "Q61_1"]
}

variable_dict = {
    "claude-few": ["Q48_1", "Q53_1", "Q58_1", "Q60_1"],
    "chat-zero": ["Q47_1", "Q54_1", "Q57_1", "Q64_1"],
    "groundtruth": ["Q49_1", "Q20_1", "Q56_1", "QID63_1"],
    "claude-zero": ["Q18_1", "Q52_1", "Q45_1", "Q62_1"],
    "chat-few": ["Q50_1", "Q51_1", "Q55_1", "Q61_1"]
}

rank_dict = {
    "1": "Q39_1",
    "2": "Q39_2",
    "3": "Q39_3",
    "4": "Q39_4",
    "5": "Q39_5",
}

python_knowledge_dict = {
    "experience": "Q5",
    "use": "Q6",
    "purpose": "Q7"
}

label_mappings = {
    "experience": {
        1: "No experience",
        2: "<1 month",
        3: "1–6 months",
        4: "6 mo – 1 yr",
        5: "1–2 years",
        6: ">2 years"
    },
    "use": {
        1: "Never",
        2: "Daily",
        3: "Weekly",
        4: "Monthly",
        5: "Irregularly"
    },
    "purpose": {
        1: "Education",
        2: "Personal/Learning",
        3: "Work",
        5: "Other"
    }
}



def load_survey(file_path: str) -> pd.DataFrame:
    """Load a single survey CSV file"""
    df = pd.read_csv(file_path)
    return df

def compute_summary(df: pd.DataFrame, survey_name: str):
    overall_summary = {}
    per_metric_summary = {}

    # Overall scores per model (average of all 4 metrics)
    for model, questions in variable_dict.items():
        overall_scores = df[questions].mean(axis=1)
        overall_summary[model] = {
            "mean": np.mean(overall_scores),
            "median": np.median(overall_scores),
            "std": np.std(overall_scores),
            "min": np.min(overall_scores),
            "max": np.max(overall_scores)
        }

    # Per-metric scores per model
    for metric, q_list in metrics_dict.items():
        per_metric_summary[metric] = {}
        for model, model_q in variable_dict.items():
            overlap = list(set(q_list) & set(model_q))
            if overlap:
                metric_scores = df[overlap].mean(axis=1)
                per_metric_summary[metric][model] = {
                    "mean": np.mean(metric_scores),
                    "median": np.median(metric_scores),
                    "std": np.std(metric_scores),
                    "min": np.min(metric_scores),
                    "max": np.max(metric_scores)
                }

    # Convert dicts to DataFrames for easier inspection
    overall_df = pd.DataFrame(overall_summary).T.round(2)
    per_metric_df = {
        metric: pd.DataFrame(per_metric_summary[metric]).T.round(2)
        for metric in per_metric_summary
    }

    # Print results nicely
    print(f"\n=== Overall Summary Statistics for {survey_name} ===")
    print(overall_df)

    print(f"\n=== Per-Metric Summary Statistics for {survey_name} ===")
    for metric, df_metric in per_metric_df.items():
        print(f"\n-- {metric.upper()} --")
        print(df_metric)

    return {"overall": overall_df, "per_metric": per_metric_df}

# Plot functions
def plot_box_chart(df, survey_name):
    plt.figure(figsize=(8,6))
    data = [df[questions].mean(axis=1) for questions in variable_dict.values()]
    plt.boxplot(data, labels=variable_dict.keys())
    plt.title(f"{survey_name}: Box Plot of LLM Percieved Quality Scores")
    plt.ylabel("Score (1–5)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_radar_chart(df, survey_name):
    metrics = list(metrics_dict.keys())
    models = variable_dict.keys()
    avg_scores = {}

    for model, questions in variable_dict.items():
        model_scores = []
        for metric, q_list in metrics_dict.items():
            overlap = list(set(questions) & set(q_list))
            score = df[overlap].mean(axis=1).mean() if overlap else 0
            model_scores.append(score)
        avg_scores[model] = model_scores

    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(8,8))
    for model, values in avg_scores.items():
        values += values[:1]
        plt.polar(angles, values, label=model)
        plt.fill(angles, values, alpha=0.1)

    plt.xticks(angles[:-1], metrics)
    plt.title(f"{survey_name}: Radar Chart of Average Metric Scores")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

def plot_grouped_bar_chart(df, survey_name):
    metrics = list(metrics_dict.keys())
    models = variable_dict.keys()
    data = {m: [] for m in metrics}

    for metric, q_list in metrics_dict.items():
        for model, model_q in variable_dict.items():
            overlap = list(set(q_list) & set(model_q))
            score = df[overlap].mean(axis=1).mean() if overlap else 0
            data[metric].append(score)

    df_plot = pd.DataFrame(data, index=models)
    df_plot.plot(kind="bar", figsize=(10,6))
    plt.title(f"{survey_name}: Grouped Bar Chart (Scores by Metric and Model)")
    plt.ylabel("Score (1–5)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# Not needed, only when all metrics are plotted, which is not the case now
def plot_column_chart(df, survey_name):
    means = {m: df[q].mean().mean() for m, q in variable_dict.items()}
    plt.figure(figsize=(8,6))
    plt.bar(means.keys(), means.values())
    plt.title(f"{survey_name}: Column Chart of Mean Scores")
    plt.ylabel("Average Score (1–5)")
    plt.ylim(0,5)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_divergent_stacked_bar_chart(df, survey_name):
    models = variable_dict.keys()
    dist = {}
    for model, questions in variable_dict.items():
        vals = df[questions].values.flatten()
        counts = pd.Series(vals).value_counts(normalize=True).sort_index()
        dist[model] = counts
    dist_df = pd.DataFrame(dist).T.fillna(0)

    pos = dist_df[[4,5]].sum(axis=1)
    neg = -dist_df[[1,2]].sum(axis=1)
    neutral = dist_df[3]

    plt.figure(figsize=(10,6))
    plt.barh(dist_df.index, neg, color="red", label="Low (1–2)")
    plt.barh(dist_df.index, neutral, left=neg, color="gray", label="Neutral (3)")
    plt.barh(dist_df.index, pos, left=neg+neutral, color="green", label="High (4–5)")
    plt.axvline(0, color="black")
    plt.title(f"{survey_name}: Divergent Stacked Bar Chart (Response Distribution of Perceived Quality Scores)")
    plt.xlabel("Proportion of Responses")
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_survey(file_path: str, survey_name: str):
    df = load_survey(file_path)
    compute_summary(df, survey_name)

    # Generate all plots
    plot_box_chart(df, survey_name)
    plot_radar_chart(df, survey_name)
    plot_grouped_bar_chart(df, survey_name)
    plot_column_chart(df, survey_name)
    plot_divergent_stacked_bar_chart(df, survey_name)

# Make record ranking visual
def load_rank_data(filepath: str, rank_dict: dict) -> pd.DataFrame:
    """
    Load a Qualtrics CSV file and return a tidy DataFrame of rankings.
    Each row = participant, each column = approach, cell = rank number.
    """
    df = pd.read_csv(filepath)

    # Reverse the mapping: question -> rank number
    question_to_rank = {v: int(k) for k, v in rank_dict.items()}

    # Extract relevant columns
    df_ranks = df[list(question_to_rank.keys())].copy()

    # Build a participant x approach table
    tidy = pd.DataFrame()
    for q_col, rank in question_to_rank.items():
        # Each Q39_x column contains the approach name that was ranked this rank
        tidy[q_col] = df[q_col]

    # Melt so each participant’s rankings become rows
    melted = tidy.melt(ignore_index=False, var_name="Question", value_name="Approach")
    melted["Rank"] = melted["Question"].map(question_to_rank)
    melted = melted.drop(columns="Question")

    # Pivot to wide format: participant x approach
    ranking_table = melted.pivot_table(index=melted.index, columns="Approach", values="Rank")

    return ranking_table


def create_summary_table(ranking_table: pd.DataFrame):
    """
    Create and display a summary table with mean rank, range, and #1 votes.
    """
    summary = pd.DataFrame({
        "Mean Rank": ranking_table.mean(),
        "Range": ranking_table.apply(lambda x: f"{int(x.min())}-{int(x.max())}", axis=0),
        "#1 Votes": (ranking_table == 1).sum(),
    }).sort_values("Mean Rank")

    print("\nSummary Table:")
    print(summary.round(2))
    return summary


def create_heatmap(ranking_table: pd.DataFrame, title: str = "Participant Rankings Heatmap"):
    """
    Create a heatmap showing each participant’s ranking per approach.
    """
    plt.figure(figsize=(8, len(ranking_table) * 0.6 + 2))
    sns.heatmap(
        ranking_table,
        annot=True,
        cmap="YlGnBu_r",
        cbar_kws={'label': 'Rank (1 = Best)'},
        linewidths=0.5,
    )
    plt.title(title)
    plt.xlabel("LLM")
    plt.ylabel("Participant")
    plt.tight_layout()
    plt.show()

# Experience participants
def _get_order_and_labels(topic: str):
    """
    Returns (ordered_keys, labels_in_order) based on label_mappings for the topic.
    ordered_keys is a list of ints (category codes in the CSV).
    labels_in_order is a list of strings (human readable labels).
    """
    mapping = label_mappings.get(topic, {})
    ordered_keys = sorted(mapping.keys())
    labels = [mapping[k] for k in ordered_keys]
    return ordered_keys, labels

def plot_python_knowledge(df: pd.DataFrame, survey_name: str, show_plot: bool = True):
    """
    For each topic in python_knowledge_dict, plot the percentage distribution
    of respondents across the integer categories recorded in the CSV column.
    The function uses label_mappings to convert integer codes into human labels.
    """
    for topic, col in python_knowledge_dict.items():
        if col not in df.columns:
            print(f"Warning: column {col} for topic '{topic}' not found in dataframe. Skipping.")
            continue

        # dropna then cast to int where possible (some CSVs might have stray floats/strings)
        vals = df[col].dropna()
        try:
            vals = vals.astype(int)
        except Exception:
            # if casting fails, keep as-is but try to coerce numeric values
            vals = pd.to_numeric(vals, errors='coerce').dropna().astype(int)

        ordered_keys, labels = _get_order_and_labels(topic)

        # compute normalized counts for the expected categories, fill missing with 0
        counts = vals.value_counts(normalize=True).reindex(ordered_keys, fill_value=0) * 100  # percent

        # If there are unexpected codes (not in label_mappings), aggregate them under "Other"
        unexpected = vals[~vals.isin(ordered_keys)]
        other_pct = (unexpected.count() / len(vals)) * 100 if len(vals) > 0 else 0.0
        display_labels = labels.copy()
        display_counts = counts.copy()

        if other_pct > 0:
            display_labels = display_labels + ["Other"]
            display_counts = list(display_counts.values) + [other_pct]

        # Plot
        plt.figure(figsize=(8, 4.8))
        bars = plt.bar(display_labels, display_counts, edgecolor="black")
        plt.title(f"{survey_name}: Python {topic.capitalize()} Distribution")
        plt.ylabel("Percentage of respondents (%)")
        plt.ylim(0, 100)
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # label each bar with percent value
        for b_idx, b in enumerate(bars):
            height = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2, height + 1.0, f"{height:.1f}%", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        if show_plot:
            plt.show()

        # Ensure display_counts is always a list of numeric values
        display_counts_list = list(display_counts) if not isinstance(display_counts, list) else display_counts
        display_labels_list = list(display_labels)

        # Build the result as a clean Series (no index alignment)
        result_series = pd.Series(data=display_counts_list, index=display_labels_list, dtype=float)

        print(f"\n{survey_name} – Python {topic.capitalize()} (counts as %):")
        print(result_series.to_frame(name="percent").round(2))
        print("-" * 60)
   

if __name__ == "__main__":
    game_df = pd.read_csv("qualtrics results/game_survey_binary.csv")
    sumtree_df = pd.read_csv("qualtrics results/sumtree_survey_binary.csv")

    # Participant's results Game
    plot_python_knowledge(game_df, "Participants game2048")

    # Participant's results Sumtree
    plot_python_knowledge(sumtree_df, "Participants sumtree")

    game_df = pd.read_csv("qualtrics results/game_survey_text.csv")
    sumtree_df = pd.read_csv("qualtrics results/sumtree_survey_text.csv")

    # Metrics results
    analyze_survey("qualtrics results/game_survey_text.csv", "Game2048")
    analyze_survey("qualtrics results/sumtree_survey_text.csv", "Sumtree")

    # Game rankings
    ranking_df = load_rank_data("qualtrics results/game_survey_text.csv", rank_dict)
    summary = create_summary_table(ranking_df)
    create_heatmap(ranking_df, title="Game2048 Rankings")
    
    # Sumtree rankings
    ranking_df = load_rank_data("qualtrics results/sumtree_survey_text.csv", rank_dict)
    summary = create_summary_table(ranking_df)
    create_heatmap(ranking_df, title="Sumtree Rankings")