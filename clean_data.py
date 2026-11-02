"""Basic cleaning checks for the student performance dataset.

The generated file is already clean, so this script is mostly a safety
net: it reports missing values and duplicates, clips values to their
valid ranges, and writes a cleaned copy for the other scripts to use.
"""

import pandas as pd

INPUT_PATH = "data/student_performance.csv"
OUTPUT_PATH = "data/student_performance_clean.csv"

RANGES = {
    "study_hours_per_week": (0, 60),
    "attendance_percent": (0, 100),
    "previous_score": (0, 100),
    "sleep_hours": (0, 16),
    "assignments_completed_percent": (0, 100),
    "social_media_hours_per_day": (0, 24),
    "final_score": (0, 100),
}


def main():
    df = pd.read_csv(INPUT_PATH)

    print(f"Rows in: {len(df)}")
    print("Missing values per column:")
    print(df.isna().sum().to_string())
    print(f"Duplicate rows: {df.duplicated().sum()}")

    df = df.drop_duplicates()
    for column, (low, high) in RANGES.items():
        df[column] = df[column].clip(low, high)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Rows out: {len(df)}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
