"""Predict a final score for a single student.

Example:
    python predict.py --study-hours 20 --attendance 90 --previous-score 75 \\
        --sleep 7 --assignments 85 --social-media 2
"""

import argparse

import joblib
import pandas as pd

MODEL_PATH = "models/student_performance_model.pkl"

FEATURES = [
    "study_hours_per_week",
    "attendance_percent",
    "previous_score",
    "sleep_hours",
    "assignments_completed_percent",
    "social_media_hours_per_day",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a student's final score.")
    parser.add_argument("--study-hours", type=float, default=15)
    parser.add_argument("--attendance", type=float, default=85)
    parser.add_argument("--previous-score", type=float, default=70)
    parser.add_argument("--sleep", type=float, default=7)
    parser.add_argument("--assignments", type=float, default=80)
    parser.add_argument("--social-media", type=float, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    model = joblib.load(MODEL_PATH)

    student = pd.DataFrame(
        [
            {
                "study_hours_per_week": args.study_hours,
                "attendance_percent": args.attendance,
                "previous_score": args.previous_score,
                "sleep_hours": args.sleep,
                "assignments_completed_percent": args.assignments,
                "social_media_hours_per_day": args.social_media,
            }
        ]
    )

    prediction = float(model.predict(student[FEATURES])[0])
    print(f"Predicted final score: {min(max(prediction, 0), 100):.1f} / 100")


if __name__ == "__main__":
    main()
