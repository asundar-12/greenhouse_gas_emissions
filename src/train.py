import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.feature_engineering import build_preprocessor, split_features_and_target


PROCESSED_DATA_PATH = Path("data/processed/modeling_dataset.csv")
MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("metrics.json")

EXPERIMENT_NAME = "ghg-emissions-random-forest"


MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
}


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the processed modeling dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {path}. Run src/data_processing.py first."
        )

    return pd.read_csv(path)


def build_model_pipeline() -> Pipeline:
    """Build a full sklearn pipeline with preprocessing and model steps."""
    preprocessor = build_preprocessor()

    model = RandomForestRegressor(**MODEL_PARAMS)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_model(model: Pipeline, X_test, y_test) -> dict:
    """Evaluate a trained model and return regression metrics."""
    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }

    return metrics


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    """Save the trained model pipeline to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_metrics(metrics: dict, path: Path = METRICS_PATH) -> None:
    """Save model evaluation metrics for DVC tracking."""
    with path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")


def train_model() -> dict:
    """Train, evaluate, log, and save the model."""
    df = load_processed_data()
    X, y = split_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_model_pipeline()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(MODEL_PARAMS)

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics)

        save_model(pipeline)
        save_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, name="model")

    return metrics


if __name__ == "__main__":
    training_metrics = train_model()

    print("Training complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("Metrics:")

    for metric_name, metric_value in training_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
