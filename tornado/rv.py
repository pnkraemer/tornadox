"""Random variables."""


import dataclasses

import jax.numpy

@dataclasses.dataclass
class MultivariateNormal:
    mean: jax.numpy.ndarray
    cov_cholesky: jax.numpy.ndarray

    def cov(self):
        return self.cov_cholesky @ self.cov_cholesky.T
