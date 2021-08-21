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
    """Batched multivariate normal distributions."""

    mean: jnp.ndarray
    cov_sqrtm: jnp.ndarray

    @property
    def cov(self):
        return self.cov_sqrtm @ jnp.transpose(self.cov_sqrtm, axes=(0, 2, 1))


@dataclasses.dataclass
class MatrixNormal:
    """Matrixvariate normal distributions."""

    mean: jnp.ndarray
    cov_sqrtm_1: jnp.ndarray
    cov_sqrtm_2: jnp.ndarray

    @property
    def cov_1(self):
        return self.cov_sqrtm_1 @ self.cov_sqrtm_1.T

    @property
    def cov_2(self):
        return self.cov_sqrtm_2 @ self.cov_sqrtm_2.T

    def dense_cov(self):
        return jnp.kron(self.cov_1, self.cov_2)
