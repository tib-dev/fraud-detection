import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Literal, Optional
import logging

# Assuming settings is imported from your project structure
from fraud_detection.core.settings import settings

logger = logging.getLogger(__name__)


class DataHandler:
    """
    DataHandler manages all I/O operations for the project.
    It integrates with the central settings to resolve paths and 
    handles loading/saving for DataFrames and plots.
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        file_type: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the handler with a specific path.
        If file_type is not provided, it infers it from the extension.
        """
        self.filepath = Path(filepath)
        self.file_type = file_type.lower() if file_type else self.filepath.suffix.replace(".", "")
        self.kwargs = kwargs

    # -------------------------
    # Core Data I/O
    # -------------------------

    def load(self) -> pd.DataFrame:
        """Loads data from the initialized filepath into a DataFrame."""
        try:
            if self.file_type == "csv":
                return pd.read_csv(self.filepath, **self.kwargs)
            if self.file_type == "parquet":
                return pd.read_parquet(self.filepath, **self.kwargs)
            if self.file_type in {"excel", "xlsx"}:
                return pd.read_excel(self.filepath, **self.kwargs)
            if self.file_type == "json":
                return pd.read_json(self.filepath, **self.kwargs)

            raise ValueError(f"Unsupported load type: {self.file_type}")
        except Exception as e:
            logger.error(f"Failed to load file at {self.filepath}: {e}")
            raise

    def save(self, df: pd.DataFrame):
        """Saves a DataFrame to the initialized filepath."""
        # Ensure the parent directory exists before saving
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            if self.file_type == "csv":
                # Default index=False for CSVs unless specified
                index = self.kwargs.get("index", False)
                df.to_csv(self.filepath, index=index, **self.kwargs)
            elif self.file_type == "parquet":
                df.to_parquet(self.filepath, **self.kwargs)
            elif self.file_type in {"excel", "xlsx"}:
                df.to_excel(self.filepath, index=self.kwargs.get(
                    "index", False), **self.kwargs)
            elif self.file_type == "json":
                df.to_json(self.filepath, **self.kwargs)
            else:
                raise ValueError(f"Unsupported save type: {self.file_type}")

            logger.info(f"File successfully saved to {self.filepath}")
        except Exception as e:
            logger.error(f"Failed to save file at {self.filepath}: {e}")
            raise

    # -------------------------
    # Factory Methods
    # -------------------------

    @classmethod
    def from_registry(
        cls,
        section: Literal["DATA", "REPORTS"],
        path_key: str,
        filename: str,
        **kwargs
    ):
        """
        Factory method to create a handler using paths defined in settings.
        Example: DataHandler.from_registry("DATA", "raw_dir", "data.csv")
        """
        # Resolve the base path from settings (e.g., settings.paths.DATA['raw_dir'])
        try:
            registry_section = getattr(settings.paths, section.upper())
            base_path = registry_section[path_key]
            full_path = base_path / filename
            return cls(filepath=full_path, **kwargs)
        except (AttributeError, KeyError) as e:
            logger.error(
                f"Path not found in registry: {section} -> {path_key}. Error: {e}")
            raise

    # -------------------------
    # Visualizations
    # -------------------------

    @staticmethod
    def save_plot(filename: str, fig: Optional[plt.Figure] = None, **kwargs):
        """
        Saves a matplotlib/seaborn figure to the reports/plots directory.
        """
        # Resolve plot directory from settings
        plot_dir = settings.paths.REPORTS["plots_dir"]
        plot_dir.mkdir(parents=True, exist_ok=True)

        save_path = plot_dir / filename

        # Use provided figure or the current active one
        if fig:
            fig.savefig(save_path, bbox_inches='tight', **kwargs)
        else:
            plt.savefig(save_path, bbox_inches='tight', **kwargs)

        logger.info(f"Plot saved to: {save_path}")
        return save_path
