from pathlib import Path
from typing import Literal
import mlflow
from fraud_detection.utils.project_root import get_project_root

def set_mlflow_tracking(
    experiment_name: str,
    backend: Literal["filesystem", "sqlite"] = "filesystem",
    artifact_subdir: str = "mlruns"
) -> str:
    """
    Configures the MLflow tracking URI and initializes the experiment.

    This function sets up where MLflow stores its metadata and artifacts. It
    automatically handles directory creation and converts system paths into
    the URI format required by MLflow.

    Args:
        experiment_name: The name of the MLflow experiment to create or use.
        backend: The storage backend to use.
            - 'filesystem': Stores data in local files (default).
            - 'sqlite': Stores metadata in a local SQLite database file.
        artifact_subdir: The name of the directory/file relative to project root
            where MLflow data will be stored.

    Returns:
        The resolved tracking URI string (e.g., 'file:///path/to/mlruns').

    Raises:
        ValueError: If a backend other than 'filesystem' or 'sqlite' is provided.
    """
    project_root: Path = get_project_root()

    if backend == "sqlite":
        # SQLite URI format: sqlite:///absolute/path/to/db
        db_path = project_root / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path.resolve().as_posix()}"

    elif backend == "filesystem":
        # File URI format: file:///absolute/path/to/mlruns
        mlruns_path = project_root / artifact_subdir
        mlruns_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:///{mlruns_path.resolve().as_posix()}"

    else:
        raise ValueError("backend must be 'filesystem' or 'sqlite'")

    mlflow.set_tracking_uri(tracking_uri)

    # Check existence to avoid duplicate creation errors in some MLflow versions
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)

    return tracking_uri
