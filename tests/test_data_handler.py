import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import matplotlib.pyplot as plt

from fraud_detection.data.loader import DataHandler
from fraud_detection.core.settings import settings


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )


@pytest.fixture
def sample_model():
    return LogisticRegression(max_iter=100)


# -------------------------------------------------------------------
# CSV
# -------------------------------------------------------------------

def test_csv_save_and_load(tmp_path, sample_df):
    path = tmp_path / "data.csv"
    handler = DataHandler(path)

    handler.save(sample_df)
    loaded = handler.load()

    pd.testing.assert_frame_equal(sample_df, loaded)


# -------------------------------------------------------------------
# Parquet
# -------------------------------------------------------------------

def test_parquet_save_and_load(tmp_path, sample_df):
    path = tmp_path / "data.parquet"
    handler = DataHandler(path)

    handler.save(sample_df)
    loaded = handler.load()

    pd.testing.assert_frame_equal(sample_df, loaded)


# -------------------------------------------------------------------
# Pickle / Joblib (models)
# -------------------------------------------------------------------

def test_pkl_save_and_load(tmp_path, sample_model):
    path = tmp_path / "model.pkl"
    handler = DataHandler(path)

    handler.save(sample_model)
    loaded = handler.load()

    assert isinstance(loaded, LogisticRegression)
    assert loaded.max_iter == sample_model.max_iter


def test_joblib_save_and_load(tmp_path, sample_model):
    path = tmp_path / "model.joblib"
    handler = DataHandler(path)

    handler.save(sample_model)
    loaded = handler.load()

    assert isinstance(loaded, LogisticRegression)


# -------------------------------------------------------------------
# Unsupported file type
# -------------------------------------------------------------------

def test_unsupported_file_type_raises(tmp_path, sample_df):
    path = tmp_path / "data.txt"
    handler = DataHandler(path)

    with pytest.raises(ValueError):
        handler.save(sample_df)


# -------------------------------------------------------------------
# Registry resolution
# -------------------------------------------------------------------

def test_from_registry_resolves_path(monkeypatch, tmp_path):
    """
    Ensure registry paths are correctly resolved
    """

    class DummyPaths:
        DATA = {"raw_dir": tmp_path}

    monkeypatch.setattr(settings, "paths", DummyPaths())

    handler = DataHandler.from_registry(
        section="DATA",
        path_key="raw_dir",
        filename="test.csv",
    )

    assert handler.filepath == tmp_path / "test.csv"


# -------------------------------------------------------------------
# Plot saving
# -------------------------------------------------------------------

def test_save_plot(tmp_path, monkeypatch):
    """
    Save a matplotlib plot without displaying it
    """

    class DummyPaths:
        REPORTS = {"plots_dir": tmp_path}

    monkeypatch.setattr(settings, "paths", DummyPaths())

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])

    save_path = DataHandler.save_plot("test_plot.png", fig=fig)

    assert save_path.exists()
    assert save_path.suffix == ".png"
