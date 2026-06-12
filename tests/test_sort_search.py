from pathlib import Path

import numpy as np
import pytest

from numcompute_stream.sort_search import binary_search, multi_key_sort, quickselect, stable_sort, topk

DATA_DIR = Path(__file__).resolve().parent / "data" / "sort_search"


def test_stable_sort_1d() -> None:
    values = np.load(DATA_DIR / "sort_values_1d.npy")
    out = stable_sort(values)

    expected = np.sort(values, kind="stable")
    assert np.array_equal(out, expected)


def test_stable_sort_2d_axis_0() -> None:
    values = np.load(DATA_DIR / "sort_values_2d.npy")
    out = stable_sort(values, axis=0)

    expected = np.sort(values, axis=0, kind="stable")
    assert np.array_equal(out, expected)


def test_multi_key_sort_by_primary_then_secondary() -> None:
    data = np.load(DATA_DIR / "multi_key_data.npy")
    out = multi_key_sort(data, keys=[0, 1])

    expected = data[np.lexsort((data[:, 1], data[:, 0]))]
    assert np.array_equal(out, expected)


def test_multi_key_sort_return_indices() -> None:
    data = np.load(DATA_DIR / "multi_key_data.npy")
    sorted_data, indices = multi_key_sort(data, keys=[0, 1, 2], return_indices=True)

    assert sorted_data.shape == data.shape
    assert indices.shape == (data.shape[0],)
    assert np.array_equal(sorted_data, data[indices])


def test_multi_key_sort_invalid_inputs_raise() -> None:
    data = np.load(DATA_DIR / "multi_key_data.npy")

    with pytest.raises(ValueError, match="2D"):
        multi_key_sort(data[:, 0], keys=[0])

    with pytest.raises(ValueError, match="at least one"):
        multi_key_sort(data, keys=[])

    with pytest.raises(IndexError, match="out of bounds"):
        multi_key_sort(data, keys=[5])


def test_topk_largest_values_and_indices() -> None:
    values = np.load(DATA_DIR / "topk_values.npy")
    k = 3
    out_values, out_indices = topk(values, k=k, largest=True, return_indices=True)

    expected_indices = np.argsort(-values, kind="stable")[:k]
    expected_values = values[expected_indices]

    assert np.array_equal(out_values, expected_values)
    assert np.array_equal(out_indices, expected_indices)


def test_topk_smallest_values_only() -> None:
    values = np.load(DATA_DIR / "topk_values.npy")
    k = 4
    out_values = topk(values, k=k, largest=False, return_indices=False)

    expected_values = np.sort(values, kind="stable")[:k]
    assert np.array_equal(out_values, expected_values)


def test_topk_invalid_inputs_raise() -> None:
    values = np.load(DATA_DIR / "topk_values.npy")

    with pytest.raises(ValueError, match="1D"):
        topk(values.reshape(3, 3), k=2)

    with pytest.raises(ValueError, match="k must"):
        topk(values, k=0)

    with pytest.raises(ValueError, match="k must"):
        topk(values, k=values.size + 1)


def test_quickselect_smallest_and_largest_rank() -> None:
    values = np.load(DATA_DIR / "topk_values.npy")

    assert quickselect(values, k=0, largest=False) == float(np.min(values))
    assert quickselect(values, k=0, largest=True) == float(np.max(values))


def test_quickselect_middle_rank_matches_sorted_value() -> None:
    values = np.load(DATA_DIR / "sort_values_1d.npy")
    k = 3
    out = quickselect(values, k=k, largest=False)

    expected = float(np.sort(values)[k])
    assert out == expected


def test_quickselect_invalid_inputs_raise() -> None:
    values = np.load(DATA_DIR / "topk_values.npy")

    with pytest.raises(ValueError, match="1D"):
        quickselect(values.reshape(3, 3), k=1)

    with pytest.raises(ValueError, match="k must"):
        quickselect(values, k=-1)

    with pytest.raises(ValueError, match="k must"):
        quickselect(values, k=values.size)


def test_binary_search_hit_returns_first_duplicate_index() -> None:
    arr = np.load(DATA_DIR / "search_sorted.npy")
    idx, exists = binary_search(arr, 2.0)

    assert exists is True
    assert idx == int(np.searchsorted(arr, 2.0, side="left"))


def test_binary_search_miss_returns_insertion_index() -> None:
    arr = np.load(DATA_DIR / "search_sorted.npy")
    idx, exists = binary_search(arr, 2.5)

    assert exists is False
    assert idx == int(np.searchsorted(arr, 2.5, side="left"))


def test_binary_search_invalid_input_raises() -> None:
    arr = np.load(DATA_DIR / "search_sorted.npy")

    with pytest.raises(ValueError, match="1D"):
        binary_search(arr.reshape(2, 3), 2.0)
