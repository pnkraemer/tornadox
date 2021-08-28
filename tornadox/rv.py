"""Random variables."""


import dataclasses
from collections import namedtuple

import jax.numpy as jnp


class MultivariateNormal(namedtuple("MatrixNormal", "mean cov_sqrtm")):
    """Multivariate normal distributions."""

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


class MatrixNormal(namedtuple("MatrixNormal", "mean cov_sqrtm_1 cov_sqrtm_2")):
    """Matrixvariate normal distributions."""

    @property
    def cov_1(self):
        return self.cov_sqrtm_1 @ self.cov_sqrtm_1.T

    @property
    def cov_2(self):
        return self.cov_sqrtm_2 @ self.cov_sqrtm_2.T

    def dense_cov(self):
        return jnp.kron(self.cov_1, self.cov_2)

    def dense_cov_sqrtm(self):
        return jnp.kron(self.cov_sqrtm_1, self.cov_sqrtm_2)
