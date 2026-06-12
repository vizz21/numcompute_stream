from pathlib import Path
import numpy as np
import pytest
from numcompute_stream.metrics import accuracy, auc, confusion_matrix, f1, mse, precision, recall, roc_curve

DATA_DIR = Path(__file__).resolve().parent / "data" / "metrics"


def test_classification_metrics_binary() -> None:
    y_true = np.load(DATA_DIR / "cls_y_true.npy")
    y_pred = np.load(DATA_DIR / "cls_y_pred.npy")

    assert accuracy(y_true, y_pred) == float(np.mean(y_true == y_pred))

    tp = 3
    fp = 1
    fn = 1
    expected_p = tp / (tp + fp)
    expected_r = tp / (tp + fn)
    expected_f1 = 2 * expected_p * expected_r / (expected_p + expected_r)
    assert precision(y_true, y_pred) == expected_p
    assert recall(y_true, y_pred) == expected_r
    assert f1(y_true, y_pred) == expected_f1


def test_confusion_matrix_multiclass() -> None:
    y_true = np.load(DATA_DIR / "multi_y_true.npy")
    y_pred = np.load(DATA_DIR / "multi_y_pred.npy")
    cm = confusion_matrix(y_true, y_pred)
    expected = np.array(
        [
            [2, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ],
        dtype=int,
    )
    assert np.array_equal(cm, expected)


def test_confusion_matrix_with_explicit_labels() -> None:
    y_true = np.load(DATA_DIR / "multi_y_true.npy")
    y_pred = np.load(DATA_DIR / "multi_y_pred.npy")

    labels = [2, 1, 0]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    expected = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 2],
        ],
        dtype=int,
    )
    assert np.array_equal(cm, expected)


def test_mse_matches_numpy() -> None:
    y_true = np.load(DATA_DIR / "reg_y_true.npy")
    y_pred = np.load(DATA_DIR / "reg_y_pred.npy")
    expected = float(np.mean((y_true - y_pred) ** 2))
    assert mse(y_true, y_pred) == expected


def test_roc_curve_and_auc() -> None:
    y_true = np.load(DATA_DIR / "roc_y_true.npy")
    y_score = np.load(DATA_DIR / "roc_y_score.npy")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    assert thresholds[0] == np.inf
    assert np.all(np.diff(fpr) >= 0)
    assert np.all(np.diff(tpr) >= 0)
    area = auc(fpr, tpr)
    assert np.isclose(area, 0.75)


def test_zero_division_behavior() -> None:
    y_true = np.array([0, 0, 0], dtype=int)
    y_pred = np.array([0, 0, 0], dtype=int)
    assert precision(y_true, y_pred, pos_label=1, zero_division=0.0) == 0.0
    assert recall(y_true, y_pred, pos_label=1, zero_division=0.0) == 0.0
    assert f1(y_true, y_pred, pos_label=1, zero_division=0.0) == 0.0


def test_invalid_shapes_raise() -> None:
    y_true = np.array([1, 0, 1], dtype=int)
    y_pred = np.array([[1, 0, 1]], dtype=int)
    with pytest.raises(ValueError, match="1D"):
        accuracy(y_true, y_pred)
    with pytest.raises(ValueError, match="same length"):
        mse(np.array([1, 2], dtype=float), np.array([1], dtype=float))


def test_auc_requires_at_least_two_points() -> None:
    with pytest.raises(ValueError, match="at least two"):
        auc(np.array([0.0]), np.array([1.0]))


def test_roc_requires_both_classes() -> None:
    y_true = np.array([1, 1, 1], dtype=int)
    y_score = np.array([0.2, 0.5, 0.8], dtype=float)
    with pytest.raises(ValueError, match="positive and negative"):
        roc_curve(y_true, y_score)
