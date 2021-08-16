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


def test_projection_matrix():

    P0 = tornado.linops.DerivativeSelection(derivative=0)

    # Test array
    test_array = jnp.array([1, 11, 111])
    assert jnp.allclose(P0 @ test_array, jnp.array(1))

    # Test matrix
    test_matrix = jnp.array(
        [
            [11, 12, 13],
            [21, 22, 23],
            [31, 32, 33],
        ]
    )
    assert jnp.allclose(P0 @ test_matrix, jnp.array([11, 12, 13]))

    # Test batched matrix
    test_batched_matrix = jnp.array(
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
    assert jnp.allclose(P0 @ test_batched_matrix, expected)
