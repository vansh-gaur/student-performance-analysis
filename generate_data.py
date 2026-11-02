"""Generate the synthetic student performance dataset.

The data is synthetic and is only meant for practising analysis and
modelling. The relationships are deliberately simple and noisy, so a model
trained on it will not look unrealistically accurate.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_STUDENTS = 500
OUTPUT_PATH = "data/student_performance.csv"


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    study_hours = np.clip(rng.normal(15, 7, N_STUDENTS), 2, 40)
    attendance = np.clip(rng.normal(85, 10, N_STUDENTS), 50, 100)
    previous_score = np.clip(rng.normal(70, 12, N_STUDENTS), 30, 100)
    sleep_hours = np.clip(rng.normal(7, 1.2, N_STUDENTS), 4, 10)
    assignments = np.clip(rng.normal(80, 15, N_STUDENTS), 0, 100)
    social_media = np.clip(rng.normal(3, 1.5, N_STUDENTS), 0, 8)

    # Sleep helps up to a point, then extra hours add nothing.
    sleep_effect = -0.8 * (sleep_hours - 7.5) ** 2

    final_score = (
        15
        + 0.35 * previous_score
        + 1.0 * study_hours
        + 0.15 * attendance
        + 0.12 * assignments
        - 1.5 * social_media
        + sleep_effect
        + rng.normal(0, 7, N_STUDENTS)
    )
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame(
        {
            "student_id": range(1, N_STUDENTS + 1),
            "study_hours_per_week": np.round(study_hours, 1),
            "attendance_percent": np.round(attendance, 1),
            "previous_score": np.round(previous_score, 1),
            "sleep_hours": np.round(sleep_hours, 1),
            "assignments_completed_percent": np.round(assignments, 1),
            "social_media_hours_per_day": np.round(social_media, 1),
            "final_score": np.round(final_score, 1),
        }
    )

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
