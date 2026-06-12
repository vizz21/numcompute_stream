from pathlib import Path
import numpy as np
import pytest

from numcompute_stream.io import load_csv

DATA_DIR = Path(__file__).resolve().parent / "data" / "io"


def test_load_csv_fill_missing_with_default_nan() -> None:
    data = load_csv(DATA_DIR / "io_basic_missing.csv")

    assert data.shape == (4, 3)
    assert np.array_equal(data[0], np.array([1.0, 2.0, 3.0]))
    assert np.isnan(data[1, 1])
    assert np.array_equal(data[2], np.array([7.0, 8.0, 9.0]))
    assert np.isnan(data[3, 0])


def test_load_csv_fill_missing_with_custom_value() -> None:
    data = load_csv(DATA_DIR / "io_basic_missing.csv", fill_value=-1.0)

    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, -1.0, 6.0],
            [7.0, 8.0, 9.0],
            [-1.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    assert np.array_equal(data, expected)


def test_load_csv_skip_rows_and_custom_delimiter() -> None:
    data = load_csv(DATA_DIR / "io_two_header_rows.csv", skip_rows=2)

    expected = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, np.nan, 2.0],
            [3.0, 3.0, 3.0],
        ],
        dtype=float,
    )
    assert data.shape == expected.shape
    assert np.array_equal(np.isnan(data), np.isnan(expected))
    assert np.array_equal(np.nan_to_num(data, nan=-999.0), np.nan_to_num(expected, nan=-999.0))


def test_load_csv_no_missing() -> None:
    data = load_csv(DATA_DIR / "io_no_missing.csv", skip_rows=1)

    expected = np.array(
        [
            0.1, 1.1, 2.1,
            3.1, 4.1, 5.1,
            6.1, 7.1, 8.1
        ],
        dtype=float,
    )
    assert np.array_equal(data.flatten(), expected)


def test_load_csv_semicolon_delimiter() -> None:
    data = load_csv(DATA_DIR / "io_semicolon_missing.csv", delimiter=";", fill_value=0.0)

    expected = np.array(
        [
            [10.0, 20.0, 30.0],
            [40.0, 0.0, 60.0],
            [70.0, 80.0, 90.0],
        ],
        dtype=float,
    )
    assert np.array_equal(data, expected)


def test_load_csv_accepts_str_and_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("h1,h2\n1,2\n3,4\n", encoding="utf-8")

    expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    assert np.array_equal(load_csv(csv_path, skip_rows=1), expected)
    assert np.array_equal(load_csv(str(csv_path), skip_rows=1), expected)


def test_load_csv_invalid_missing_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Invalid missing_strategy"):
        load_csv(DATA_DIR / "io_no_missing.csv", missing_strategy="drop")
