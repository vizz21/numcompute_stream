import numpy as np
from typing import Sequence, Tuple, Union

def stable_sort(data, axis=-1):
    """
    Perform a stable sort on an array.

    Parameters
    data : array-like
    axis : int, optional

    Returns
    np.ndarray
        Sorted array with same shape as input.

    Time Complexity
    O(n log n)

    Space Complexity
    O(n)
    """
    data = np.sort(data, axis=axis, kind="stable")
    return data


def multi_key_sort(
        data: np.ndarray,
        keys: Sequence[int],
        return_indices: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Sort a 2D array using multiple column keys (stable).

    Parameters
    data : np.ndarray of shape (n_rows, n_cols)
    keys : Sequence[int]
        Column indices in priority order.
    return_indices : bool, optional

    Returns
    np.ndarray of shape (n_rows, n_cols)
    or (np.ndarray, np.ndarray)
        Sorted array and optionally row indices.

    Raises
    ValueError
        If input is not 2D or keys empty.
    IndexError
        If key is out of bounds.

    Time Complexity
    O(k * n log n)

    Space Complexity
    O(n)
    """
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("multi_key_sort expects a 2D array.")
    if len(keys) == 0:
        raise ValueError("keys must contain at least one column index.")

    n_rows, n_cols = arr.shape
    indices = np.arange(n_rows)

    for key in reversed(keys):
        if key < 0 or key >= n_cols:
            raise IndexError(f"Column index {key} is out of bounds for shape {arr.shape}.")
        order = np.argsort(arr[indices, key], kind="stable")
        indices = indices[order]

    sorted_data = arr[indices]
    if return_indices:
        return sorted_data, indices
    return sorted_data



def topk(values, k, largest=True, return_indices=True):
    """
    Select top-k elements from a 1D array.

    Parameters
    values : np.ndarray of shape (n,)
    k : int
    largest : bool, optional
    return_indices : bool, optional

    Returns
    np.ndarray of shape (k,)
    or (np.ndarray, np.ndarray)
        Top-k values and optionally indices.

    Raises
    ValueError
        If input is not 1D or k is invalid.

    Time Complexity
    O(n)

    Space Complexity
    O(k)
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("topk expects a 1D array.")
    if k <= 0 or k > values.size:
        raise ValueError("k must be between 1 and len(values)")

    if largest:
        partition_idx = values.size - k
        idx = np.argpartition(values, partition_idx)[partition_idx:]
        order = np.argsort(values[idx])[::-1]
    else:
        partition_idx = k
        idx = np.argpartition(values, partition_idx)[:k]
        order = np.argsort(values[idx])

    idx = idx[order]

    if return_indices:
        return values[idx], idx
    return values[idx]


def quickselect(values: np.ndarray, k: int, largest: bool = False) -> float:
    """
    Select k-th element using quickselect.

    Parameters
    values : np.ndarray of shape (n,)
    k : int
        Zero-based rank.
    largest : bool, optional

    Returns
    float
        Selected value.

    Raises
    ValueError
        If input is not 1D or k is out of range.

    Time Complexity
    O(n) average, O(n^2) worst-case

    Space Complexity
    O(n)
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("quickselect expects a 1D array.")
    if k < 0 or k >= arr.size:
        raise ValueError(f"k must be in [0, {arr.size - 1}], got {k}.")

    work = arr.copy()
    target = arr.size - 1 - k if largest else k

    left = 0
    right = work.size - 1

    while True:
        if left == right:
            return float(work[left])

        pivot = work[(left + right) // 2]
        i, j = left, right

        while i <= j:
            while work[i] < pivot:
                i += 1
            while work[j] > pivot:
                j -= 1
            if i <= j:
                work[i], work[j] = work[j], work[i]
                i += 1
                j -= 1

        if target <= j:
            right = j
        elif target >= i:
            left = i
        else:
            return float(work[target])


def binary_search(sorted_array: np.ndarray, x: float) -> Tuple[int, bool]:
    """
    Perform binary search on a sorted array.

    Parameters
    sorted_array : np.ndarray of shape (n,)
    x : float

    Returns
    tuple[int, bool]
        (insertion index, exists flag).

    Raises
    ValueError
        If input is not 1D.

    Time Complexity
    O(log n)

    Space Complexity
    O(1)
    """
    arr = np.asarray(sorted_array)
    if arr.ndim != 1:
        raise ValueError("binary_search expects a 1D array.")

    idx = int(np.searchsorted(arr, x, side="left"))
    exists = idx < arr.size and arr[idx] == x
    return idx, bool(exists)