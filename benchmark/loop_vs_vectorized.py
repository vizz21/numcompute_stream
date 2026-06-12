import numpy as np
import time
from numcompute_stream.stats import mean, variance

def loop_mean(x):
    total = 0
    count = 0
    for val in x:
        if not np.isnan(val):
            total += val
            count += 1
    return total / count if count > 0 else np.nan

def benchmark_vectorization():
    print("Benchmarking Loop vs. Vectorized Operations")
    print("-" * 40)
    
    # Large dataset for benchmarking
    n_samples = 1_000_000
    data = np.random.randn(n_samples)
    data[np.random.choice(n_samples, 1000, replace=False)] = np.nan
    
    # Loop-based mean
    start = time.time()
    res_loop = loop_mean(data)
    time_loop = time.time() - start
    
    # Vectorized mean (using our package's function)
    start = time.time()
    res_vec = mean(data, ignore_nan=True)
    time_vec = time.time() - start
    
    print(f"Loop Mean:       {res_loop:.6f} | Time: {time_loop:.6f}s")
    print(f"Vectorized Mean: {res_vec:.6f} | Time: {time_vec:.6f}s")
    print(f"Speedup:         {time_loop / time_vec:.2f}x")
    print("-" * 40)

if __name__ == "__main__":
    benchmark_vectorization()
