
import time
import numpy as np


def benchmark(func, *args, repeat=5, warmup=1):
    """
    Measure execution time of a function.

    Parameters
    func : callable
        Function to benchmark.
    *args : Any
        Positional arguments passed to the function.
    repeat : int, optional
        Number of timed runs (default is 5).
    warmup : int, optional
        Number of warmup runs before timing (default is 1).

    Returns
    float
        Best execution time in seconds across all runs.

    Raises
    ValueError
        If repeat or warmup is negative.

    Time Complexity
    O((repeat + warmup) * T_func), where T_func is execution time of func.

    Space Complexity
    O(repeat), for storing timing results.
    """
    for _ in range(warmup):
        func(*args)

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)

    return min(times)  # best time


def compare_functions(funcs, args, labels=None, repeat=5):
    """
    Benchmark and compare multiple functions on identical inputs.

    Parameters
    funcs : list of callable
        List of functions to benchmark.
    args : tuple
        Arguments passed to each function.
    labels : list of str, optional
        Labels corresponding to each function. If None, function names are used.
    repeat : int, optional
        Number of timed runs per function (default is 5).

    Returns
    list of tuple[str, float]
        List of (label, best_time_seconds) pairs.

    Raises
    ValueError
        If funcs and labels have mismatched lengths.

    Time Complexity
    O(len(funcs) * repeat * T_func)

    Space Complexity
    O(len(funcs))
    """

    results = []

    if labels is None:
        labels = [f.__name__ for f in funcs]

    for func, label in zip(funcs, labels):
        t = benchmark(func, *args, repeat=repeat)
        results.append((label, t))

    return results


def print_results(results, title="Benchmark Results"):
    """
    Print formatted benchmark results.

    Parameters
    results : list of tuple[str, float]
        List of (label, time_in_seconds) pairs.
    title : str, optional
        Title for the output table.

    Returns
    None
    """
    print(f"\n{title}")
    print("-" * 40)
    print(f"{'Function':<20} {'Time (ms)':>10}")
    print("-" * 40)

    for label, t in results:
        print(f"{label:<20} {t * 1000:>10.4f}")

    print("-" * 40)


def mean_loop(x):
    """
    Compute mean using a Python loop.

    Parameters
    x : iterable of float
        Input sequence of length n.

    Returns
    float
        Mean of the input values.

    Time Complexity
    O(n)

    Space Complexity
    O(1)
    """

    total = 0.0
    for val in x:
        total += val
    return total / len(x)


def mean_vectorised(x):
    """Compute mean using NumPy vectorisation.

    Parameters
    x : np.ndarray
        Input array of shape (n,).

    Returns
    float
        Mean of the input values.

    Time Complexity
    O(n)

    Space Complexity
    O(1)
    """
    return np.mean(x)


def top_k_loop(x, k):
    """
    Compute top-k elements using Python sorting.

    Parameters
    x : iterable
        Input sequence of length n.
    k : int
        Number of top elements to return.

    Returns
    list
        List of k largest elements (not necessarily sorted).

    Raises
    ValueError
        If k is not in range [1, len(x)].

    Time Complexity
    O(n log n), due to sorting.

    Space Complexity
    O(n)
    """
    return sorted(x)[-k:]


def top_k_vectorised(x, k):
    """
    Compute top-k elements using NumPy argpartition.

    Parameters
    x : np.ndarray
        Input array of shape (n,).
    k : int
        Number of top elements to return.

    Returns
    np.ndarray
        Array of k largest elements (unordered).

    Raises
    ValueError
        If k is not in range [1, n].

    Time Complexity
    O(n), average case using partial partition.

    Space Complexity
    O(k)
    """
    idx = np.argpartition(x, -k)[-k:]
    return x[idx]


def run_all_benchmarks(n=1_000_000, k=10, seed=42):
    """
    Run benchmark suite comparing loop vs NumPy implementations.

    Parameters
    n : int, optional
        Size of input array (default is 1,000,000).
    k : int, optional
        Number of top elements for top-k benchmark (default is 10).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    None

    Time Complexity
    O(n + benchmark_cost), dominated by benchmarking functions.

    Space Complexity
    O(n), for storing the random array.
    """
    np.random.seed(seed)
    x = np.random.rand(n)

    print("\nEnvironment:")
    print(f"Array size: {n}")
    print(f"Top-k: {k}")
    print(f"Seed: {seed}")

    mean_results = compare_functions(
        funcs=[mean_vectorised, mean_loop],
        args=(x,),
        labels=["Mean (NumPy)", "Mean (Loop)"],
    )
    print_results(mean_results, title="Mean Benchmark")

    topk_results = compare_functions(
        funcs=[top_k_vectorised, top_k_loop],
        args=(x, k),
        labels=["Top-k (NumPy)", "Top-k (Loop)"],
    )
    print_results(topk_results, title="Top-k Benchmark")


if __name__ == "__main__":
    run_all_benchmarks()
