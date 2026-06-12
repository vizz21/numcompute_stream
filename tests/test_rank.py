from pathlib import Path

import numpy as np
import pytest

from numcompute_stream.rank import percentile, rank

DATA_DIR = Path(__file__).resolve().parent / "data" / "rank"


def test_rank_average_handles_ties() -> None:
    values = np.load(DATA_DIR / "rank_values.npy")
    out = rank(values, method="average")

    expected = np.array([6.0, 4.5, 4.5, 2.0, 2.0, 2.0], dtype=float)
    assert np.array_equal(out, expected)


def test_rank_dense_handles_ties() -> None:
    values = np.load(DATA_DIR / "rank_values.npy")
    out = rank(values, method="dense")

    expected = np.array([3.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=float)
    assert np.array_equal(out, expected)


def test_rank_ordinal_uses_stable_order_for_ties() -> None:
    values = np.load(DATA_DIR / "rank_values.npy")
    out = rank(values, method="ordinal")

    expected = np.array([6.0, 4.0, 5.0, 1.0, 2.0, 3.0], dtype=float)
    assert np.array_equal(out, expected)


def test_rank_invalid_inputs_raise() -> None:
    values = np.load(DATA_DIR / "rank_values.npy")

    with pytest.raises(ValueError, match="1D"):
        rank(values.reshape(2, 3))

    with pytest.raises(ValueError, match="non-empty"):
        rank(np.array([], dtype=float))

    with pytest.raises(ValueError, match="method"):
        rank(values, method="min")


def test_percentile_scalar_linear() -> None:
    values = np.load(DATA_DIR / "percentile_values.npy")
    out = percentile(values, 25.0, interpolation="linear")

    expected = float(np.percentile(values, 25.0, method="linear"))
    assert out == expected


def test_percentile_vector_midpoint() -> None:
    values = np.load(DATA_DIR / "percentile_values.npy")
    qs = np.array([10.0, 50.0, 90.0], dtype=float)
    out = percentile(values, qs, interpolation="midpoint")

    expected = np.percentile(values, qs, method="midpoint")
    assert np.array_equal(out, expected)


def test_percentile_lower_and_higher_modes() -> None:
    values = np.load(DATA_DIR / "percentile_values.npy")

    out_lower = percentile(values, 33.0, interpolation="lower")
    out_higher = percentile(values, 33.0, interpolation="higher")

    expected_lower = float(np.percentile(values, 33.0, method="lower"))
    expected_higher = float(np.percentile(values, 33.0, method="higher"))

    assert out_lower == expected_lower
    assert out_higher == expected_higher


def test_percentile_invalid_inputs_raise() -> None:
    values = np.load(DATA_DIR / "percentile_values.npy")

    with pytest.raises(ValueError, match="1D"):
        percentile(values.reshape(2, 4), 50.0)

    with pytest.raises(ValueError, match="non-empty"):
        percentile(np.array([], dtype=float), 50.0)

    with pytest.raises(ValueError, match="range"):
        percentile(values, -1.0)

    with pytest.raises(ValueError, match="range"):
        percentile(values, 101.0)

    with pytest.raises(ValueError, match="interpolation"):
        percentile(values, 50.0, interpolation="nearest")
