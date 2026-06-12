from pathlib import Path
import numpy as np
import pytest
from numcompute_stream.stats import mean,variance,welford,histogram,quantile

DATA_DIR = Path(__file__).resolve().parent/ "data"/ "stats"


def test_mean_normal_case()-> None:
    X = np.load(DATA_DIR/ "stats_normal.npy")
    assert np.isclose(mean(X), 3.0)


def test_mean_ignores_nan() -> None:
    X = np.load(DATA_DIR/ "stats_nan.npy")
    result = mean(X, ignore_nan=True)
    assert np.isclose(result, 3.0)

def test_mean_with_nan_returns_nan() -> None:
    X = np.load(DATA_DIR/ "stats_nan.npy")
    result = mean(X, ignore_nan=False)
    assert np.isnan(result)


def test_mean_empty_array()-> None:
    with pytest.raises(ValueError, match="Empty"):
        mean(np.array([]))


def test_variance_normal_case()-> None:
    X = np.load(DATA_DIR/ "stats_normal.npy")
    assert np.isclose(variance(X), np.var(X))

def test_variance_sample_ddof()-> None:
    X = np.load(DATA_DIR/ "stats_normal.npy")
    assert np.isclose(variance(X, ddof=1), np.var(X, ddof=1))

def test_variance_empty_raises()-> None:
    with pytest.raises(ValueError, match="Empty"):
        variance(np.array([]))


def test_welford_matches_numpy()-> None:
    X = np.load(DATA_DIR/ "stats_normal.npy")
    w_mean, w_var = welford(X)
    clean = X[~np.isnan(X)]
    assert np.isclose(w_mean, np.mean(clean))

def test_welford_ignores_nan()-> None:
    X = np.load(DATA_DIR/ "stats_nan.npy")
    w_mean,w_var = welford(X)
    clean = X[~np.isnan(X)]
    assert np.isclose(w_mean, np.mean(clean))

def test_welford_empty_array_raise()-> None:
    with pytest.raises(ValueError, match="Empty"):
        welford(np.array([]))


def test_histogram_correct_shape() -> None:
    x = np.load(DATA_DIR / "stats_histogram.npy")
    hist, bin_edges = histogram(x, bins=4)
    assert len(hist) == 4
    assert len(bin_edges) == 5

def test_histogram_counts_sum_to_total() -> None:
    x = np.load(DATA_DIR / "stats_histogram.npy")
    hist, _ = histogram(x, bins=4)
    assert np.sum(hist) == len(x)

def test_histogram_empty_array_raises() -> None:
    with pytest.raises(ValueError, match="Empty"):
        histogram(np.array([]))

def test_quantile_median() -> None:
    x = np.load(DATA_DIR / "stats_normal.npy")
    assert np.isclose(quantile(x, 0.5), np.quantile(x, 0.5))

def test_quantile_multiple_values() -> None:
    x = np.load(DATA_DIR / "stats_normal.npy")
    qs = [0.25, 0.5, 0.75]
    assert np.allclose(quantile(x, qs), np.quantile(x, qs))

def test_quantile_invalid_q_raises() -> None:
    x = np.load(DATA_DIR / "stats_normal.npy")
    with pytest.raises(ValueError, match="q must"):
        quantile(x, q=1.5)

def test_quantile_empty_array_raises() -> None:
    with pytest.raises(ValueError, match="Empty"):
        quantile(np.array([]), q=0.5)