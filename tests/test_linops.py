"""Tests for linear operators and sparse matrices."""


import jax.numpy as jnp
import jax.scipy.linalg

import tornado


def test_block_diagonal():

    A = jnp.arange(0, 9).reshape((3, 3))
    B = jnp.arange(10, 19).reshape((3, 3))
    dense = jax.scipy.linalg.block_diag(A, B)

    # todense() works correctly
    sparse = tornado.linops.BlockDiagonal.from_arrays(A, B)
    assert jnp.allclose(sparse.todense(), dense)

    # matmul() works correctly with other linops
    expected = jax.scipy.linalg.block_diag(A @ A, B @ B)
    new = sparse @ sparse
    assert isinstance(new, tornado.linops.BlockDiagonal)
    assert jnp.allclose(new.todense(), expected)

    # matvec() works correctly with jax.numpy.array
    arr = jnp.arange(0, dense.shape[0])
    new = sparse @ arr
    expected = dense @ arr
    assert isinstance(new, jnp.ndarray)
    assert new.shape == arr.shape
    assert jnp.allclose(new, expected)

    # Sum of two block diagonals works
    B1 = tornado.linops.BlockDiagonal.from_arrays(A, B)
    B2 = tornado.linops.BlockDiagonal.from_arrays(B, A)
    new = B1 + B2
    expected = B1.todense() + B2.todense()
    assert isinstance(new, tornado.linops.BlockDiagonal)
    assert jnp.allclose(new.todense(), expected)
