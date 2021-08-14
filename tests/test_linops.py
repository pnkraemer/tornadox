"""Tests for linear operators and sparse matrices."""


import tornado
import jax.numpy as jnp
import jax.scipy.linalg

def test_block_diagonal():

    A = jnp.arange(0,9).reshape((3, 3))
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
