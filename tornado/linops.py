"""Linear operators and sparse matrices."""


import jax.scipy.linalg
import jax.numpy as jnp


class BlockDiagonal:
    """Block-diagonal matrices where the blocks have all equal shape."""
    def __init__(self, array_stack):

        self.array_stack = array_stack

    @classmethod
    def from_arrays(cls, *arrays):
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

        num_elements = self.array_stack.shape[0]
        reshaped = other.reshape((num_elements, -1))
        multiplied = jnp.einsum("ijk,ik->ij", self.array_stack, reshaped)

        return multiplied.reshape((-1,))


