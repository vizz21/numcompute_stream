from pathlib import Path
import numpy as np
import pytest
from numcompute_stream.benchmarking import (
    benchmark,
    compare_functions,
    mean_loop,
    mean_vectorised,
    top_k_loop,
    top_k_vectorised,
)


def test_mean_loop_correct() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(mean_loop(x), 3.0)


def test_mean_loop_matches_vectorised() -> None:
    x = np.random.rand(1000)
    assert np.isclose(mean_loop(x), mean_vectorised(x))


def test_top_k_loop_returns_k_elements() -> None:
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    result = top_k_loop(x, k=2)
    assert len(result) == 2


def test_top_k_loop_returns_largest_values() -> None:
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    result = top_k_loop(x, k=2)
    assert set(result) == {4.0, 5.0}


def test_top_k_vectorised_returns_k_indices() -> None:
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    result = top_k_vectorised(x, k=2)
    assert len(result) == 2


def test_benchmark_returns_positive_time() -> None:
    f = lambda x: np.mean(x)
    x = np.random.rand(1000)
    t = benchmark(f, x)
    assert t > 0.0


def test_compare_functions_returns_correct_count() -> None:
    x = np.random.rand(1000)
    results = compare_functions(
        funcs=[mean_vectorised, mean_loop],
        args=(x,),
        labels=["vectorised", "loop"]
    )
    assert len(results) == 2
    assert results[0][0] == "vectorised"
    assert results[1][0] == "loop"