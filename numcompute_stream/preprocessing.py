import numpy as np
from typing import Optional, Tuple



def _validate_array(X: np.ndarray, name: str = "X",
                    check_nan: bool = False) -> np.ndarray:
    """
    Convert input to a 2D float NumPy array and validate it.

    Parameters
 
    X : array-like
    name : str
    check_nan : bool
        If True, raise ValueError when NaNs are present.

    Returns
    np.ndarray, shape (n_samples, n_features)

    Raises
    ValueError
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if check_nan and np.any(np.isnan(arr)):
        raise ValueError(
            f"{name} contains NaN values. Apply SimpleImputer first."
        )
    return arr


def _validate_1d(X: np.ndarray, name: str = "X") -> np.ndarray:
    """Convert to 1D array and check it is non-empty."""
    arr = np.asarray(X)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    return arr



class StandardScaler:
    """
    Standardise features to zero mean and unit variance.

    Formula: z = (X - mean) / std
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_samples_seen_: int = 0

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = 0
        arr = _validate_array(X, check_nan=True)
        return self.partial_fit(arr)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise ValueError("StandardScaler is not fitted. Call fit() first.")
        arr = _validate_array(X, check_nan=True)
        if arr.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"Expected {self.mean_.shape[0]} features, got {arr.shape[1]}."
            )
        return (arr - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def partial_fit(self, X: np.ndarray) -> "StandardScaler":
        arr = _validate_array(X, check_nan=False)

        n_new = arr.shape[0]
        mean_new = np.nanmean(arr, axis=0)
        var_new = np.nanvar(arr, axis=0)

        if self.n_samples_seen_ == 0:
            self.mean_ = mean_new
            self.var_ = var_new
            self.n_samples_seen_ = n_new
        else:
            n_old = self.n_samples_seen_
            mean_old = self.mean_
            var_old = self.var_
            n_total = n_old + n_new

            delta = mean_new - mean_old
            self.mean_ = (n_old * mean_old + n_new * mean_new) / n_total
            m2_old = var_old * n_old
            m2_new = var_new * n_new
            self.var_ = (m2_old + m2_new + delta ** 2 * n_old * n_new / n_total) / n_total
            self.n_samples_seen_ = n_total

        self.std_ = np.where(self.var_ == 0.0, 1.0, np.sqrt(self.var_))
        return self




class MinMaxScaler:
    """
    Scale features to [a, b]

    Formula: z = (X - min) / (max - min) * (b - a) + a
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        a, b = feature_range
        if b <= a:
            raise ValueError(
                f"feature_range max ({b}) must be greater than min ({a})."
            )
        self.feature_range = (float(a), float(b))
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.range_: Optional[np.ndarray] = None
        self.n_samples_seen_: int = 0

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.min_ = None
        self.max_ = None
        self.n_samples_seen_ = 0
        return self.partial_fit(_validate_array(X, check_nan=True))

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise ValueError("MinMaxScaler is not fitted. Call fit() first.")
        arr = _validate_array(X, check_nan=True)
        if arr.shape[1] != self.min_.shape[0]:
            raise ValueError(
                f"Expected {self.min_.shape[0]} features, got {arr.shape[1]}."
            )
        a, b = self.feature_range
        return (arr - self.min_) / self.range_ * (b - a) + a

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def partial_fit(self, X: np.ndarray) -> "MinMaxScaler":
        arr = _validate_array(X, check_nan=True)
        chunk_min = np.nanmin(arr, axis=0)
        chunk_max = np.nanmax(arr, axis=0)

        if self.n_samples_seen_ == 0:
            self.min_ = chunk_min
            self.max_ = chunk_max
        else:
            self.min_ = np.minimum(self.min_, chunk_min)
            self.max_ = np.maximum(self.max_, chunk_max)

        self.range_ = np.where(self.max_ == self.min_, 1.0, self.max_ - self.min_)
        self.n_samples_seen_ += arr.shape[0]
        return self


class OneHotEncoder:
    def __init__(self) -> None:
        self.categories_: Optional[np.ndarray] = None
        self.fitted_: bool = False

    def fit(self, X: np.ndarray) -> "OneHotEncoder":
        self.categories_ = None
        self.fitted_ = False
        return self.partial_fit(_validate_1d(X))

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise ValueError(
                "OneHotEncoder is not fitted. Call fit() or partial_fit() first."
            )
        arr = _validate_1d(X)
        return (arr[:, None] == self.categories_).astype(int)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def partial_fit(self, X: np.ndarray) -> "OneHotEncoder":
        arr = np.asarray(X)
        if arr.size == 0:
            return self
        new_cats = np.unique(arr)
        if self.categories_ is None:
            self.categories_ = new_cats
        else:
            self.categories_ = np.unique(
                np.concatenate([self.categories_, new_cats])
            )
        self.fitted_ = True
        return self


class SimpleImputer:
    """
    Imputer for completing missing values with simple strategies like  mean, median
  along each column, or using a constant value.
    """

    def __init__(self, strategy: str = "mean", fill_value: float = 0) -> None:
        allowed = {"mean", "median", "constant"}
        if strategy not in allowed:
            raise ValueError(
                f"Invalid strategy {strategy!r}. Choose from {allowed}."
            )
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_: Optional[np.ndarray] = None
        self.n_samples_seen_: int = 0

        self._sum_: Optional[np.ndarray] = None
        self._count_: Optional[np.ndarray] = None

        self._all_chunks_: list = []

    def fit(self, X: np.ndarray) -> "SimpleImputer":
        self.statistics_ = None
        self.n_samples_seen_ = 0
        self._sum_ = None
        self._count_ = None
        self._all_chunks_ = []
        return self.partial_fit(_validate_array(X))

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.statistics_ is None:
            raise ValueError(
                "SimpleImputer is not fitted. Call fit() or partial_fit() first."
            )
        arr = _validate_array(X, check_nan=False)
        return np.where(np.isnan(arr), self.statistics_, arr)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def partial_fit(self, X: np.ndarray) -> "SimpleImputer":
        arr = _validate_array(X, check_nan=False)

        if self.strategy == "mean":
            chunk_sum = np.nansum(arr, axis=0)
            chunk_count = np.sum(~np.isnan(arr), axis=0)
            if self.n_samples_seen_ == 0:
                self._sum_ = chunk_sum
                self._count_ = chunk_count
            else:
                self._sum_ += chunk_sum
                self._count_ += chunk_count
            self.statistics_ = np.where(
                self._count_ > 0, self._sum_ / self._count_, np.nan
            )

        elif self.strategy == "median":
            self._all_chunks_.append(arr)
            combined = np.vstack(self._all_chunks_)
            self.statistics_ = np.nanmedian(combined, axis=0)

        elif self.strategy == "constant":
            n_features = arr.shape[1]
            self.statistics_ = np.full(n_features, float(self.fill_value))

        self.n_samples_seen_ += arr.shape[0]
        return self


__all__ = ["StandardScaler", "MinMaxScaler", "OneHotEncoder", "SimpleImputer"]