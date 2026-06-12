from pathlib import Path
import numpy as np
import pytest

from numcompute_stream.utils import (
    euchlidean_distance,
    pairwise_euclidean_distances,
    manhattan_distance,
    cosine_similarity,
    jaccard_similarity,
    hamming_distance,
    relu,
    sigmoid,
    softmax,
    logsumexp,
    topk_indices,
    batch_slices
)

DATA_DIR = Path(__file__).resolve().parent / "data"/ "utils"

def test_euclidean_distance_known_values() -> None:
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert euchlidean_distance(a, b) == 5.0

def test_euclidean_distance_identical_arrays() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert euchlidean_distance(a, a) == 0.0

def test_pairwise_euclidean_distances_shape() -> None:
    X = np.load(DATA_DIR / "utils_X.npy")
    Y = np.load(DATA_DIR / "utils_Y.npy")
    out = pairwise_euclidean_distances(X, Y)
    assert out.shape == (X.shape[0], Y.shape[0])

def test_pairwise_euclidean_distances_invalid_inputs_raise() -> None:
    X = np.array([[1.0, 2.0]])
    Y = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="feature dimension"):
        pairwise_euclidean_distances(X, Y)

def test_manhattan_distance_known_values() -> None:
    a = np.array([1.0, 2.0])
    b = np.array([4.0, 6.0])
    assert manhattan_distance(a, b) == 7.0

def test_manhattan_distance_identical_arrays() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert manhattan_distance(a, a) == 0.0

def test_cosine_similarity_identical_vectors() -> None:
    a = np.array([1.0, 0.0])
    assert np.isclose(cosine_similarity(a, a), 1.0)

def test_cosine_similarity_orthogonal_vectors() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert np.isclose(cosine_similarity(a, b), 0.0)

def test_cosine_similarity_zero_vector() -> None:
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])
    assert cosine_similarity(a, b) == 0.0


def test_jaccard_similarity_known_values() -> None:
    a = np.array([1, 1, 0, 1])
    b = np.array([1, 0, 1, 1])
    assert np.isclose(jaccard_similarity(a, b), 2/4)


def test_jaccard_similarity_identical_arrays() -> None:
    a = np.array([1, 1, 0])
    assert np.isclose(jaccard_similarity(a, a), 1.0)

def test_hamming_distance_known_values() -> None:
    a = np.array([1, 0, 1, 1])
    b = np.array([1, 1, 0, 1])
    assert hamming_distance(a, b) == 2

def test_hamming_distance_identical_arrays() -> None:
    a = np.array([1, 0, 1])
    assert hamming_distance(a, a) == 0

def test_relu_positive_values_unchanged() -> None:
    x = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(relu(x), x)

def test_relu_negative_values_become_zero() -> None:
    x = np.array([-1.0, -2.0, 3.0])
    expected = np.array([0.0, 0.0, 3.0])
    assert np.array_equal(relu(x), expected)

def test_sigmoid_zero_input() -> None:
    assert np.isclose(sigmoid(np.array([0.0])), 0.5)

def test_sigmoid_output_range() -> None:
    x = np.array([-100.0, 0.0, 100.0])
    out = sigmoid(x)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)

def test_softmax_sums_to_one() -> None:
    x = np.array([1.0, 2.0, 3.0])
    assert np.isclose(np.sum(softmax(x)), 1.0)


def test_softmax_largest_value_has_highest_probability() -> None:
    x = np.array([1.0, 2.0, 3.0])
    out = softmax(x)
    assert np.argmax(out) == 2


def test_logsumexp_known_values() -> None:
    x = np.array([0.0, 0.0, 0.0])
    assert np.isclose(logsumexp(x), np.log(3.0))


def test_topk_indices_largest() -> None:
    values = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    out = topk_indices(values, k=2, largest=True)
    assert set(out) == {4, 2}

def test_topk_indices_invalid_inputs_raise() -> None:
    values = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="1D"):
        topk_indices(values.reshape(3, 1), k=1)
    with pytest.raises(ValueError, match="k must"):
        topk_indices(values, k=0)

def test_batch_slices_normal_case() -> None:
    slices = list(batch_slices(10, 3))
    assert slices == [(0, 3), (3, 6), (6, 9), (9, 10)]

def test_batch_slices_drop_last() -> None:
    slices = list(batch_slices(10, 3, drop_last=True))
    assert slices == [(0, 3), (3, 6), (6, 9)]

def test_batch_slices_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        list(batch_slices(-1, 3))
    with pytest.raises(ValueError, match="positive"):
        list(batch_slices(10, 0))