"""Linear operators and sparse matrices."""


import jax.numpy as jnp
import jax.scipy.linalg


class BlockDiagonal:
    """Block-diagonal matrices where the blocks have all equal shape.

    Parameters
    ----------
    array_stack
        Stack of blocks (which are jax.ndarray objects).
        Shape (N, K, L) which implies that there are N blocks, each of which has shape (K, L).
    """

    def __init__(self, array_stack):
        self._array_stack = array_stack

    @property
    def array_stack(self):
        return self._array_stack

    @property
    def num_blocks(self):
        return self._array_stack.shape[0]

    @classmethod
    def from_arrays(cls, *arrays):
        """Same interface as jax.scipy.linalg.block_diag(). Can be used for tests."""
        return cls(jnp.stack(arrays))

    def todense(self):
        return jax.scipy.linalg.block_diag(*self.array_stack)

    def __matmul__(self, other):
        if isinstance(other, BlockDiagonal):
            array_stack = self.array_stack @ other.array_stack
            return BlockDiagonal(array_stack)

        if isinstance(other, jnp.ndarray) and other.ndim == 1:

            reshaped_array = other.reshape((self.num_blocks, -1))
            block_matmul = jnp.einsum("ijk,ik->ij", self.array_stack, reshaped_array)
            return block_matmul.reshape((-1,))
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, BlockDiagonal):
            array_stack = self.array_stack + other.array_stack
            return BlockDiagonal(array_stack)
        return NotImplemented


# Todo: make faster?
def truncate_block_diagonal(dense_array, num_blocks, block_shape):
    n1, n2 = block_shape
    return jnp.stack(
        [
            dense_array[i * n1 : (i + 1) * n1, i * n2 : (i + 1) * n2]
            for i in range(num_blocks)
        ]
    )
