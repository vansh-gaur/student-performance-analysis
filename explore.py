"""Exploratory analysis: summary statistics, correlations and plots."""

import matplotlib.pyplot as plt
import pandas as pd

INPUT_PATH = "data/student_performance_clean.csv"
FIGURES_DIR = "reports/figures"


def save_histogram(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df["final_score"], bins=20)
    plt.xlabel("Final score")
    plt.ylabel("Number of students")
    plt.title("Distribution of final scores")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/final_score_distribution.png", dpi=150)
    plt.close()


def save_scatter(df, x, title, filename):
    plt.figure(figsize=(8, 5))
    plt.scatter(df[x], df["final_score"], alpha=0.5)
    plt.xlabel(x.replace("_", " "))
    plt.ylabel("Final score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150)
    plt.close()


def save_correlation_heatmap(df):
    corr = df.drop(columns=["student_id"]).corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation between features and final score")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()
    return corr


def main():
    df = pd.read_csv(INPUT_PATH)

    summary = df.describe().round(2)
    summary.to_csv("reports/summary_stats.csv")
    print(summary.to_string())

    corr = save_correlation_heatmap(df)
    corr.round(3).to_csv("reports/correlations.csv")
    print("\nCorrelation with final_score:")
    print(corr["final_score"].sort_values(ascending=False).round(3).to_string())

    save_histogram(df)
    save_scatter(df, "study_hours_per_week", "Study hours vs final score", "study_hours_vs_final.png")
    save_scatter(df, "previous_score", "Previous score vs final score", "previous_score_vs_final.png")
    print(f"\nFigures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
