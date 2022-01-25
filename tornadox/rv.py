"""Random variables."""

from collections import namedtuple

import jax.numpy as jnp


class MultivariateNormal(namedtuple("_MultivariateNormal", "mean cov_sqrtm")):
    """Multivariate normal distributions."""

    @property
    def cov(self):
        return self.cov_sqrtm @ self.cov_sqrtm.T


class BatchedMultivariateNormal(
    namedtuple("_BatchedMultivariateNormal", "mean cov_sqrtm")
):
    """Batched multivariate normal distributions."""

    @property
    def cov(self):
        return self.cov_sqrtm @ jnp.transpose(self.cov_sqrtm, axes=(0, 2, 1))


class MatrixNormal(namedtuple("_MatrixNormal", "mean cov_sqrtm_1 cov_sqrtm_2")):
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


class LeftIsotropicMatrixNormal(
    namedtuple("_LeftIsotropicMatrixNormal", "mean d cov_sqrtm_2")
):
    """Matrixvariate normal distributions as they appear in the EK0

    The left Kronecker factor is the identity anyways. Instead of storing that one as a
    full matrix, just store its dimension.
    """

    @property
    def cov_sqrtm_1(self):
        return jnp.eye(self.d)

    @property
    def cov_1(self):
        return jnp.eye(self.d)

    @property
    def cov_2(self):
        return self.cov_sqrtm_2 @ self.cov_sqrtm_2.T

    @property
    def cov(self):
        return jnp.stack([self.cov_2 for _ in range(self.d)])

    def dense_cov(self):
        return jnp.kron(self.cov_1, self.cov_2)

    def dense_cov_sqrtm(self):
        return jnp.kron(self.cov_sqrtm_1, self.cov_sqrtm_2)
