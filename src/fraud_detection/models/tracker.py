from pathlib import Path
import mlflow
from fraud_detection.utils.project_root import get_project_root

def set_mlflow_tracking(experiment_name: str, backend: str = "filesystem", artifact_subdir: str = "mlruns") -> str:
    project_root: Path = get_project_root()

    if backend == "sqlite":
        db_path = project_root / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path.resolve().as_posix()}"
    elif backend == "filesystem":
        mlruns_path = project_root / artifact_subdir
        mlruns_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:///{mlruns_path.resolve().as_posix()}"
    else:
        raise ValueError("backend must be 'filesystem' or 'sqlite'")

    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    return tracking_uri
