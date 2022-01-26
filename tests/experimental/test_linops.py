"""Tests for linear operators and sparse matrices."""


import jax.numpy as jnp
import jax.scipy.linalg
import pytest

import tornadox

# Tests for blockdiagonals


@pytest.fixture
def A():
    return jnp.arange(0, 9).reshape((3, 3))


@pytest.fixture
def B():
    return jnp.arange(10, 19).reshape((3, 3))


@pytest.fixture
def sparse_dense_blockdiag(A, B):
    dense = jax.scipy.linalg.block_diag(A, B)

    # todense() works correctly
    sparse = tornadox.experimental.linops.BlockDiagonal.from_arrays(A, B)
    return sparse, dense


def test_todense(sparse_dense_blockdiag):
    sparse, dense = sparse_dense_blockdiag
    assert jnp.allclose(sparse.todense(), dense)


def test_matmul_blockdiag_blockdiag(sparse_dense_blockdiag, A, B):
    sparse, dense = sparse_dense_blockdiag
    expected = jax.scipy.linalg.block_diag(A @ A, B @ B)
    new = sparse @ sparse
    assert isinstance(new, tornadox.experimental.linops.BlockDiagonal)
    assert jnp.allclose(new.todense(), expected)


def test_matvec_blodiag_array(sparse_dense_blockdiag):
    sparse, dense = sparse_dense_blockdiag
    arr = jnp.arange(0, dense.shape[0])
    new = sparse @ arr
    expected = dense @ arr
    assert isinstance(new, jnp.ndarray)
    assert new.shape == arr.shape
    assert jnp.allclose(new, expected)


def test_sum_block_diagonals(A, B):
    B1 = tornadox.experimental.linops.BlockDiagonal.from_arrays(A, B)
    B2 = tornadox.experimental.linops.BlockDiagonal.from_arrays(B, A)
    new = B1 + B2
    expected = B1.todense() + B2.todense()
    assert isinstance(new, tornadox.experimental.linops.BlockDiagonal)
    assert jnp.allclose(new.todense(), expected)


def test_diff_block_diagonals(A, B):
    B1 = tornadox.experimental.linops.BlockDiagonal.from_arrays(A, B)
    B2 = tornadox.experimental.linops.BlockDiagonal.from_arrays(B, A)
    new = B1 - B2
    expected = B1.todense() - B2.todense()
    assert isinstance(new, tornadox.experimental.linops.BlockDiagonal)
    assert jnp.allclose(new.todense(), expected)


def test_transpose_block_diagonals(A, B):
    BD = tornadox.experimental.linops.BlockDiagonal.from_arrays(A, B)
    new = BD.T
    expected = BD.todense().T
    assert isinstance(new, tornadox.experimental.linops.BlockDiagonal)
    assert jnp.allclose(new.todense(), expected)


# Tests for derivative selection


@pytest.fixture
def P0():
    return tornadox.experimental.linops.DerivativeSelection(derivative=0)


def test_is_operator(P0):
    assert isinstance(P0, tornadox.experimental.linops.DerivativeSelection)


def test_array(P0):
    array = jnp.array([1, 11, 111])
    assert jnp.allclose(P0 @ array, jnp.array(1))


def test_matrix(P0):
    matrix = jnp.array(
        [
            [11, 12, 13],
            [21, 22, 23],
            [31, 32, 33],
        ]
    )
    assert jnp.allclose(P0 @ matrix, jnp.array([11, 12, 13]))


def test_batched_matrix(P0):
    batched_matrix = jnp.array(
        [
            [
                [111, 112, 113],
                [121, 122, 123],
                [131, 132, 133],
            ],
            [
                [211, 212, 213],
                [221, 222, 223],
                [231, 232, 233],
            ],
        ]
    )
    expected = jnp.array(
        [
            [111, 112, 113],
            [211, 212, 213],
        ]
    )
    assert jnp.allclose(P0 @ batched_matrix, expected)


# Test for blockdiagonal truncation


@pytest.mark.parametrize("d,n1,n2", [(5, 3, 2), (2, 3, 1)])
def test_truncate_block_diagonal_array(d, n1, n2):
    array_stack = jnp.arange(d * n1 * n2).reshape((d, n1, n2))
    dense_matrix = tornadox.experimental.linops.BlockDiagonal(array_stack).todense()
    dense_as_array_stack = tornadox.experimental.linops.truncate_block_diagonal(
        dense_matrix, num_blocks=d, block_shape=(n1, n2)
    )
    assert dense_as_array_stack.shape == (d, n1, n2)
    assert jnp.allclose(
        tornadox.experimental.linops.BlockDiagonal(dense_as_array_stack).todense(),
        dense_matrix,
    )
    assert jnp.allclose(dense_as_array_stack, array_stack)
