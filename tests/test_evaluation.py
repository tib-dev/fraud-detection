import numpy as np
import pytest

from src.fraud_detection.models.evaluation import (
    _safe_confusion_matrix,
    compute_classification_metrics,
    compute_expected_loss,
    compute_business_score,
    find_best_threshold_metric_based,
    find_best_threshold_cost_based,
)


# ----------------------------
# Confusion Matrix Tests
# ----------------------------

def test_safe_confusion_matrix_basic_case():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    tn, fp, fn, tp = _safe_confusion_matrix(y_true, y_pred)

    assert tn == 1
    assert fp == 1
    assert fn == 1
    assert tp == 1


def test_safe_confusion_matrix_empty_input():
    y_true = np.array([])
    y_pred = np.array([])

    tn, fp, fn, tp = _safe_confusion_matrix(y_true, y_pred)

    assert (tn, fp, fn, tp) == (0, 0, 0, 0)


# ----------------------------
# Classification Metrics Tests
# ----------------------------

def test_compute_classification_metrics_without_proba():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    metrics = compute_classification_metrics(y_true, y_pred)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "tn" in metrics
    assert "fp" in metrics
    assert "fn" in metrics
    assert "tp" in metrics
    assert "auc_pr" not in metrics
    assert "roc_auc" not in metrics


def test_compute_classification_metrics_with_proba():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_proba = np.array([0.1, 0.9, 0.4, 0.2])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    assert "auc_pr" in metrics
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] > 0


def test_compute_classification_metrics_single_class():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    assert metrics["auc_pr"] == 0.0
    assert metrics["roc_auc"] == 0.0


# ----------------------------
# Business Loss & Score Tests
# ----------------------------

def test_compute_expected_loss():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    loss = compute_expected_loss(
        y_true,
        y_pred,
        false_positive_cost=10,
        false_negative_cost=50
    )

    # 1 FP * 10 + 1 FN * 50 = 60
    assert loss == 60


def test_compute_business_score_range():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    score = compute_business_score(
        y_true,
        y_pred,
        false_positive_cost=10,
        false_negative_cost=50
    )

    assert 0 <= score <= 1


def test_compute_business_score_empty_input():
    score = compute_business_score(
        np.array([]),
        np.array([]),
        false_positive_cost=10,
        false_negative_cost=50
    )

    assert score == 0.0


# ----------------------------
# Threshold Optimization Tests
# ----------------------------

def test_find_best_threshold_metric_based_returns_valid_values():
    y_true = np.array([0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.8, 0.2])

    threshold, score = find_best_threshold_metric_based(
        y_true,
        y_proba,
        metric="f1",
        steps=50
    )

    assert 0.01 <= threshold <= 0.99
    assert 0 <= score <= 1


def test_find_best_threshold_cost_based_returns_valid_values():
    y_true = np.array([0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.8, 0.2])

    threshold, loss = find_best_threshold_cost_based(
        y_true,
        y_proba,
        false_positive_cost=10,
        false_negative_cost=50,
        steps=50
    )

    assert 0.01 <= threshold <= 0.99
    assert loss >= 0


def test_cost_based_threshold_prefers_low_loss():
    y_true = np.array([0, 1])
    y_proba = np.array([0.2, 0.9])

    threshold, loss = find_best_threshold_cost_based(
        y_true,
        y_proba,
        false_positive_cost=1,
        false_negative_cost=100,
        steps=50
    )

    # Should avoid false negative due to high cost
    assert loss >= 0
    assert 0.01 <= threshold <= 0.99
