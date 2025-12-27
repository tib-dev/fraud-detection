import mlflow
from pathlib import Path
from fraud_detection.utils.project_root import get_project_root


def set_mlflow_tracking(
    experiment_name: str = "fraud_detection_models",
):
    """
    Configure MLflow tracking to use project-root/mlruns (Windows-safe).
    """

    project_root = get_project_root()
    mlruns_path = project_root / "mlruns"

    # Ensure directory exists
    mlruns_path.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: file:/// + forward slashes
    tracking_uri = f"file:///{mlruns_path.resolve().as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return tracking_uri
