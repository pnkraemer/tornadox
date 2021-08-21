"""Random variables."""


import dataclasses

import jax.numpy as jnp


@dataclasses.dataclass
class MultivariateNormal:
    """Multivariate normal distributions."""

    mean: jnp.ndarray
    cov_sqrtm: jnp.ndarray

    @property
    def cov(self):
        return self.cov_sqrtm @ self.cov_sqrtm.T


@dataclasses.dataclass
class BatchedMultivariateNormal:
    """Multivariate normal distributions."""

    mean: jnp.ndarray
    cov_sqrtm: jnp.ndarray

    @property
    def cov(self):
        return self.cov_sqrtm @ jnp.transpose(self.cov_sqrtm, axes=(0, 2, 1))
