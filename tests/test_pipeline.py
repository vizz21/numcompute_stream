import numpy as np
import pytest
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent/ "data"/ "pipeline"


def test_pipeline_imputer_then_scaler()-> None:
     X = np.array([[1.0,    2.0],
                  [np.nan, 4.0],
                  [5.0,    6.0]])
     
     pipe = Pipeline([
          ("imputer", SimpleImputer(strategy="mean")),
          ("scaler", StandardScaler())
     ])

     X_out = pipe.fit_transform(X)
     

     assert not np.any(np.isnan(X_out))
     assert np.allclose(np.mean(X_out, axis=0), 0.0)
     assert np.allclose(np.std(X_out, axis=0), 1.0)


def test_pipeline_imputer_then_minmax() -> None:
     X = np.array([[1.0,    2.0],
                  [np.nan, 4.0],
                  [5.0,    6.0]])
    
     pipe = Pipeline([
          ("imputer", SimpleImputer(strategy="mean")),
          ("scaler", MinMaxScaler())
    ])

     X_out = pipe.fit_transform(X)

     assert not np.any(np.isnan(X_out))
     assert np.allclose(np.min(X_out, axis=0), 0.0)
     assert np.allclose(np.max(X_out, axis=0), 1.0)


def test_pipeline_fit_transform_same_as_fit_then_transform() -> None:
     X = np.load(DATA_DIR/ "pipeline_normal.npy")
     pipe1 = Pipeline([("scaler", StandardScaler())])
     pipe2 = Pipeline([("scaler", StandardScaler())])
     X_out1 = pipe1.fit_transform(X)
     pipe2.fit(X)
     X_out2 = pipe2.transform(X)

     assert np.allclose(X_out1,X_out2)



def test_pipeline_empty_step_raises() -> None:
     with pytest.raises(ValueError, match="steps"):
          Pipeline([])

def test_pipeline_duplicate_name_raises() -> None:
     with pytest.raises(ValueError, match="Duplicate"):
          Pipeline([
               ("scaler", StandardScaler()),
               ("scaler", MinMaxScaler())])



def test_pipeline_step_missing_fit_raises() -> None:
    class NoFit:
        def transform(self, X): return X
    with pytest.raises(ValueError, match="fit"):
        Pipeline([("bad", NoFit())])


def test_pipeline_nonfinal_missing_transform_raises() -> None:
    class NoTransform:
        def fit(self, X): return self
    with pytest.raises(ValueError, match="Non-final"):
        Pipeline([("bad", NoTransform()),
                  ("scaler", StandardScaler())])

def test_pipeline_final_missing_both_raises() -> None:
    class NoTransformNoPredict:
        def fit(self, X): return self
    with pytest.raises(ValueError, match="Final step"):
        Pipeline([("bad", NoTransformNoPredict())])


def test_pipeline_predict_no_predict_method_raises() -> None:
    X = np.load(DATA_DIR / "pipeline_normal.npy")
    pipe = Pipeline([("scaler", StandardScaler())])
    pipe.fit(X)
    with pytest.raises(ValueError, match="predict"):
        pipe.predict(X)
