import mlflow
from fraud_detection.utils.project_root import get_project_root


def set_mlflow_tracking():
    # Get project root reliably
    project_root = get_project_root()
    mlruns_path = project_root / "mlruns"

    # Make folder if it doesn't exist
    mlruns_path.mkdir(exist_ok=True)

    # Use file:// URI with forward slashes
    mlflow.set_tracking_uri(f"file:///{mlruns_path.resolve().as_posix()}")
    mlflow.set_experiment("fraud_detection_models")
