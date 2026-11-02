"""Train and compare regression models for final score prediction."""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

INPUT_PATH = "data/student_performance_clean.csv"
MODEL_PATH = "models/student_performance_model.pkl"

FEATURES = [
    "study_hours_per_week",
    "attendance_percent",
    "previous_score",
    "sleep_hours",
    "assignments_completed_percent",
    "social_media_hours_per_day",
]
TARGET = "final_score"

MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "rmse": round(mean_squared_error(y_test, predictions) ** 0.5, 3),
        "mae": round(mean_absolute_error(y_test, predictions), 3),
        "r2": round(r2_score(y_test, predictions), 3),
    }


def main():
    df = pd.read_csv(INPUT_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    fitted = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        fitted[name] = model

        cv_rmse = -cross_val_score(
            model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error"
        ).mean()
        test_metrics = evaluate(model, X_test, y_test)
        results[name] = {"cv_rmse": round(float(cv_rmse), 3), **test_metrics}
        print(f"{name}: {results[name]}")

    best_name = min(results, key=lambda name: results[name]["cv_rmse"])
    joblib.dump(fitted[best_name], MODEL_PATH)
    print(f"\nBest model by cross-validation RMSE: {best_name}")
    print(f"Saved to {MODEL_PATH}")

    pd.DataFrame(results).T.to_csv("reports/model_results.csv")
    with open("reports/model_results.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    # Feature importance from the tree models, for comparison.
    importance_rows = []
    for name in ["Random Forest", "Gradient Boosting"]:
        for feature, value in zip(FEATURES, fitted[name].feature_importances_):
            importance_rows.append({"model": name, "feature": feature, "importance": round(float(value), 3)})
    importance = pd.DataFrame(importance_rows)
    importance.to_csv("reports/feature_importance.csv", index=False)
    print("\nFeature importance:")
    print(importance.to_string(index=False))

    rf_importance = importance[importance["model"] == "Random Forest"].sort_values("importance")
    plt.figure(figsize=(8, 5))
    plt.barh(rf_importance["feature"], rf_importance["importance"])
    plt.xlabel("Importance")
    plt.title("Random Forest feature importance")
    plt.tight_layout()
    plt.savefig("reports/figures/feature_importance.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
