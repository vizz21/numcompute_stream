from pathlib import Path
import numpy as np
import pytest
from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer

DATA_DIR = Path(__file__).resolve().parent / "data" / "preprocessing"

#sTANDARDScaler
def test_standard_scaler_mean_zero_std_pne() -> None:
        X = np.load(DATA_DIR / "preprocessing_normal.npy")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.allclose(np.mean(X_scaled, axis=0), 0.0)
        assert np.allclose(np.std(X_scaled, axis=0), 1.0)


def test_standard_scaler_all_equal_values_no_crash() -> None:
    X = np.load(DATA_DIR / "preprocessing_equal.npy")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert not np.any(np.isnan(X_scaled))
    assert not np.any(np.isinf(X_scaled))


def test_standard_scaler_not_fitted_raises() -> None:
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    scaler = StandardScaler()
    with pytest.raises(ValueError, match="not fitted"):
        scaler.transform(X)


def test_standard_scaler_wrong_features_raises() -> None:
    X_train = np.array([[1.0, 2.0],
                        [3.0, 4.0]])
    X_test = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
    scaler = StandardScaler()
    scaler.fit(X_train)
    with pytest.raises(ValueError, match="features"):
        scaler.transform(X_test)

#MinMax
def test_minmax_scaler_default_range() -> None:
    X = np.load(DATA_DIR / "preprocessing_normal.npy")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(np.min(X_scaled, axis=0), 0.0)
    assert np.allclose(np.max(X_scaled, axis=0), 1.0)

def test_minmax_scaler_custom_feature_range() -> None:
    X = np.load(DATA_DIR / "preprocessing_normal.npy")
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(np.min(X_scaled, axis=0), -1.0)
    assert np.allclose(np.max(X_scaled, axis=0), 1.0)

def test_minmax_scaler_all_equal_values_no_crash() -> None:
    X = np.load(DATA_DIR / "preprocessing_equal.npy")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    assert not np.any(np.isnan(X_scaled))
    assert not np.any(np.isinf(X_scaled))

def test_minmax_scaler_invalid_range_raises() -> None:
    with pytest.raises(ValueError, match="must be greater than min"):
        MinMaxScaler(feature_range=(1.0, 0.0))


#OneHotEncoder
def test_onehotencoder_correct_output() -> None:
    X = np.load(DATA_DIR / "preprocessing_categories.npy")
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)
    assert X_encoded.shape == (4, 3)
    assert np.array_equal(X_encoded[0], X_encoded[3]) 
    assert np.all(X_encoded.sum(axis=1) == 1)

def test_onehotencoder_not_fitted_raises() -> None:
    X = np.array(["cat", "dog"])
    encoder = OneHotEncoder()
    with pytest.raises(ValueError, match="not fitted"):
        encoder.transform(X)

#SimpleImputer
def test_simple_imputer_mean_replaces_nan() -> None:
    X = np.load(DATA_DIR / "preprocessing_nan.npy")
    imputer = SimpleImputer(strategy="mean")
    X_out = imputer.fit_transform(X)
    assert not np.any(np.isnan(X_out))
    assert np.isclose(X_out[1, 0], np.nanmean(X[:, 0]))

def test_simple_imputer_constant_replaces_nan() -> None:
    X = np.load(DATA_DIR / "preprocessing_nan.npy")
    imputer = SimpleImputer(strategy="constant", fill_value=-1.0)
    X_out = imputer.fit_transform(X)
    assert not np.any(np.isnan(X_out))
    assert X_out[1, 0] == -1.0
    assert X_out[2, 1] == -1.0

def test_simple_imputer_invalid_strategy_raises() -> None:
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    with pytest.raises(ValueError, match="Invalid strategy"):
        SimpleImputer(strategy="mode").fit(X)

def test_simple_imputer_no_nan_unchanged() -> None:
    X = np.load(DATA_DIR / "preprocessing_normal.npy")
    imputer = SimpleImputer(strategy="mean")
    X_out = imputer.fit_transform(X)
    assert np.array_equal(X, X_out)