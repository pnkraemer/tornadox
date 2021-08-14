"""Linear operators and sparse matrices."""


import jax.scipy.linalg
import jax.numpy as jnp


class BlockDiagonal:
    """Block-diagonal matrices where the blocks have all equal shape."""
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
        """Same interface as jax.scipy.linalg.block_diagonal(). Can be used for tests."""
        return cls(jnp.stack(arrays))

    def todense(self):
        return jax.scipy.linalg.block_diag(*self.array_stack)

    def __matmul__(self, other):
        if isinstance(other, BlockDiagonal):
            array_stack = self.array_stack @ other.array_stack
            return BlockDiagonal(array_stack)

        # Only matvec works now!
        assert isinstance(other, jnp.ndarray)
        assert other.ndim == 1

        reshaped_array = other.reshape((self.num_blocks, -1))
        block_matmul = jnp.einsum("ijk,ik->ij", self.array_stack, reshaped_array)
        return block_matmul.reshape((-1,))


