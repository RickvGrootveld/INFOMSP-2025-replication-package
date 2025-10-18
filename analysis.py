import pandas as pd
from statsmodels.multivariate.manova import MANOVA
import argparse
import os

"""
mapping clarification from experiment group to file
game_input = "file S"
game_groundtruth = "file C"
game_chat_zero = "file B"
game_chat_few = "file E"
game_claude_zero = "file D"
game_claude_few = "file A"

sumtree_input = "file S"
sumtree_groundtruth = "file C"
sumtree_chat_zero = "file B"
sumtree_chat_few = "file E"
sumtree_claude_zero = "file D"
sumtree_claude_few = "file A"
"""

# Define your metric and variable mappings
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

# Helper function to load and process data
def load_experiment_data(csv_name):
    """Load and process the CSV for a given experiment."""
    folder = "qualtrics results"
    file_path = os.path.join(folder, f"{csv_name}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {csv_name}.csv in the working directory.")

    df = pd.read_csv(file_path)

    # Build a tidy dataframe with columns: participant, LLM, Prompting, Accuracy, Understandability, Completeness, Relevance
    records = []

    for var_name, questions in variable_dict.items():
        # Determine factor levels
        llm = "ChatGPT" if "chat" in var_name else "Claude"
        prompting = "Few-shot" if "few" in var_name else "Zero-shot"

        # Extract responses for this condition
        subset = df[questions]

        # Compute per-participant mean scores for each metric
        for i, row in subset.iterrows():
            # Skip the groundtruth condition
            if var_name == "groundtruth":
                continue
            record = {"participant": i, "LLM": llm, "Prompting": prompting}
            for metric, q_list in metrics_dict.items():
                # Only include relevant questions for this variable
                relevant_qs = [q for q in q_list if q in questions]
                record[metric] = row[relevant_qs].mean()
            records.append(record)

    tidy_df = pd.DataFrame(records)
    return tidy_df

# Run MANOVA analysis
def run_manova(df, experiment_name):
    """Perform a 2x2 MANOVA on the given dataframe."""
    print(f"\nRunning MANOVA for {experiment_name} experiment...")
    formula = "accuracy + understandability + completeness + relevance ~ LLM * Prompting"
    maov = MANOVA.from_formula(formula, data=df)
    print(maov.mv_test())

# Main function
def main():
    parser = argparse.ArgumentParser(description="Run MANOVA for LLM experiments.")
    parser.add_argument("--dataset", choices=["game", "sumtree", "both"], default="both",
                        help="Choose which experiment(s) to analyze.")
    args = parser.parse_args()

    if args.dataset in ["game", "both"]:
        game_df = load_experiment_data("game_survey_text")
        run_manova(game_df, "Game")

    if args.dataset in ["sumtree", "both"]:
        sumtree_df = load_experiment_data("sumtree_survey_text")
        run_manova(sumtree_df, "Sumtree")

    if args.dataset == "both":
        combined_df = pd.concat([game_df, sumtree_df], ignore_index=True)
        run_manova(combined_df, "Combined (Game + Sumtree)")

if __name__ == "__main__":
    main()

import statsmodels.api as sm
from statsmodels.formula.api import ols

for metric in ["accuracy", "understandability", "completeness", "relevance"]:
    model = ols(f"{metric} ~ LLM * Prompting", data=combined_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f"\nUnivariate ANOVA for {metric}")
    print(anova_table)