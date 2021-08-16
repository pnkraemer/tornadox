"""Random variables."""


import dataclasses

import jax.numpy


@dataclasses.dataclass
class MultivariateNormal:
    """Multivariate normal distributions."""

    mean: jax.numpy.ndarray
    cov_sqrtm: jax.numpy.ndarray

    @property
    def cov(self):
        return self.cov_sqrtm @ self.cov_sqrtm.T
