import numpy as np
from typing import Optional, Union



def mean(x, axis=None, ignore_nan: bool = True):
    """
    Compute the mean of an array
    """
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("Empty array passed to mean().")
    return np.nanmean(x, axis=axis) if ignore_nan else np.mean(x, axis=axis)


def variance(x, axis=None, ddof: int = 0, ignore_nan: bool = True):
    """
    Compute the variance of an array
    """
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("Empty array passed to variance().")
    return (np.nanvar(x, axis=axis, ddof=ddof)
            if ignore_nan else np.var(x, axis=axis, ddof=ddof))


def welford(x):
    """
    Compute the sample mean and *sample* variance (ddof=1) via Welford's
    online algorithm
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        raise ValueError("Empty input (or all-NaN) passed to welford().")
    if n == 1:
        return float(x[0]), 0.0
    return float(np.mean(x)), float(np.var(x, ddof=1))


def histogram(x, bins: int = 10, range=None):
    """
    Compute a histogram
    """
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("Empty array passed to histogram().")
    return np.histogram(x, bins=bins, range=range)


def quantile(x, q, axis=None):
    """
    Compute quantiles
    """
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("Empty array passed to quantile().")
    q_arr = np.asarray(q)
    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be in [0, 1].")
    return np.quantile(x, q, axis=axis)



class StreamingMeanVariance:
    """
    Incrementally compute per-feature mean and variance using Welford algorithm.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean_val = None 
        self._m2 = None

    def update_stats(self, X_chunk: np.ndarray,
                     axis: int = 0) -> "StreamingMeanVariance":

        X_chunk = np.asarray(X_chunk, dtype=float)
        if X_chunk.size == 0:
            return self

        if X_chunk.ndim == 1:
            valid = X_chunk[~np.isnan(X_chunk)]
            n_new = valid.size
            if n_new == 0:
                return self
            mean_new = float(np.mean(valid))
            var_new = float(np.var(valid)) if n_new > 1 else 0.0
            m2_new = var_new * n_new
        else:
            n_new_per_col = np.sum(~np.isnan(X_chunk), axis=0)
            if np.all(n_new_per_col == 0):
                return self
            n_new = int(np.min(n_new_per_col[n_new_per_col > 0]))
            mean_new = np.nanmean(X_chunk, axis=0)
            var_new = np.nanvar(X_chunk, axis=0)
            m2_new = var_new * n_new_per_col

        if self.n == 0:
            self.mean_val = mean_new.copy() if isinstance(mean_new, np.ndarray) else mean_new
            self._m2 = m2_new.copy() if isinstance(m2_new, np.ndarray) else m2_new
            self.n = n_new
        else:
            n_old = self.n
            mean_old = self.mean_val
            n_total = n_old + n_new
            delta = mean_new - mean_old

            self.mean_val = (n_old * mean_old + n_new * mean_new) / n_total
            self._m2 = (self._m2 + m2_new
                        + delta ** 2 * n_old * n_new / n_total)
            self.n = n_total

        return self

    def update(self, X_chunk: np.ndarray,
               axis: int = 0) -> "StreamingMeanVariance":
        return self.update_stats(X_chunk, axis=axis)

    def mean(self) -> Union[float, np.ndarray]:
        if self.n == 0:
            return np.nan
        return self.mean_val

    def variance(self, ddof: int = 0) -> Union[float, np.ndarray]:
        if self.n <= ddof:
            return np.nan
        return self._m2 / (self.n - ddof)

    def std(self, ddof: int = 0) -> Union[float, np.ndarray]:
        v = self.variance(ddof=ddof)
        if isinstance(v, float) and np.isnan(v):
            return np.nan
        return np.sqrt(v)


def update_stats(X_chunk: np.ndarray) -> StreamingMeanVariance:
    return StreamingMeanVariance().update_stats(X_chunk)


__all__ = [
    "mean", "variance", "welford", "histogram", "quantile",
    "StreamingMeanVariance", "update_stats",
]
