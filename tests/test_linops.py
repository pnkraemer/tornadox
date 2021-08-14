"""Tests for linear operators and sparse matrices."""


import tornado
import jax.numpy as jnp
import jax.scipy.linalg

def test_block_diagonal():

    A = jnp.arange(0,9).reshape((3, 3))
    B = jnp.arange(10, 19).reshape((3, 3))
    dense = jax.scipy.linalg.block_diag(A, B)

    # todense() works correctly
    sparse = tornado.linops.BlockDiagonal(A, B)
    assert jnp.allclose(sparse.todense(), dense)

    # matmul() works correctly with other linops
    new = sparse @ sparse
    expected = jax.scipy.linalg.block_diag(A @ A, B @ B)
    assert jnp.allclose(new.todense(), expected)
