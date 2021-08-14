"""Linear operators and sparse matrices."""


import jax.scipy.linalg

class BlockDiagonal:

    def __init__(self, *arrs):
        self.arrs = arrs

    def todense(self):
        return jax.scipy.linalg.block_diag(*self.arrs)