"""Linear operators and sparse matrices."""


import jax.scipy.linalg


class BlockDiagonal:

    def __init__(self, *arrs):
        self.arrs = arrs

    def todense(self):
        return jax.scipy.linalg.block_diag(*self.arrs)

    def __matmul__(self, other):
        return BlockDiagonal(*(a @ b for (a, b) in zip(self.arrs, other.arrs)))