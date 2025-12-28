from fraud_detection.utils.project_root import get_project_root
import mlflow


def set_mlflow_tracking(
    experiment_name: str = "fraud_detection_models",
    use_sqlite: bool = False,
):
    """
    Configure MLflow tracking.
    
    Options:
    - Default: project-root/mlruns (filesystem, Windows-safe)
    - use_sqlite=True: uses SQLite backend for future-proofing
    """

    if use_sqlite:
        # SQLite backend
        project_root = get_project_root()
        db_path = project_root / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path.resolve().as_posix()}"
    else:
        # Filesystem backend
        project_root = get_project_root()
        mlruns_path = project_root / "mlruns"
        mlruns_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:///{mlruns_path.resolve().as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return tracking_uri
