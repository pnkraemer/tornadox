"""Tests for linear operators and sparse matrices."""


import jax.numpy as jnp
import jax.scipy.linalg
import pytest

import tornado


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
    sparse = tornado.linops.BlockDiagonal.from_arrays(A, B)
    return sparse, dense


def test_todense(sparse_dense_blockdiag):
    sparse, dense = sparse_dense_blockdiag
    assert jnp.allclose(sparse.todense(), dense)


def test_matmul_blockdiag_blockdiag(sparse_dense_blockdiag, A, B):
    sparse, dense = sparse_dense_blockdiag
    expected = jax.scipy.linalg.block_diag(A @ A, B @ B)
    new = sparse @ sparse
    assert isinstance(new, tornado.linops.BlockDiagonal)
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
    B1 = tornado.linops.BlockDiagonal.from_arrays(A, B)
    B2 = tornado.linops.BlockDiagonal.from_arrays(B, A)
    new = B1 + B2
    expected = B1.todense() + B2.todense()
    assert isinstance(new, tornado.linops.BlockDiagonal)
    assert jnp.allclose(new.todense(), expected)
