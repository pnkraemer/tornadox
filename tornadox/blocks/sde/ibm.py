"""Integrated Brownian motion."""

from functools import partial

import jax.lax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("num_derivatives",))
def preconditioner_diagonal(*, dt, num_derivatives):
    powers = jnp.arange(num_derivatives, -1, -1)

    scales = _factorial(powers)
    powers = powers + 0.5

    scaling_vector = (jnp.abs(dt) ** powers) / scales
    scaling_vector_inv = (jnp.abs(dt) ** (-powers)) * scales

    return scaling_vector, scaling_vector_inv


@partial(jax.jit, static_argnames=("num_derivatives",))
def system_matrices_1d(*, num_derivatives):
    x = jnp.arange(num_derivatives + 1)

    A_1d = jnp.flip(
        _arange_to_pascal_matrix(x)[0]
    )  # no idea why the [0] is necessary...
    Q_1d = jnp.flip(_arange_to_hilbert_matrix(x))
    return A_1d, jnp.linalg.cholesky(Q_1d)


@jax.jit
def _arange_to_hilbert_matrix(a):
    return 1 / (a[:, None] + a[None, :] + 1)


@jax.jit
def _arange_to_pascal_matrix(a):
    return _batch_gram(_binom)(a[:, None], a[None, :])


def _batch_gram(k):
    r"""Batch a function.

    A function :math:`f: R^k \times R^k \rightarrow R` becomes a function

        .. math: g: R^{n \times k} \times R^{k \times m} \rightarrow R^{n \times m}.

    The batching follows broadcasting rules:
    think `jnp.exp(x[:, None], y[None, :])` for 1d inputs.
    """
    k_vmapped_x = jax.vmap(k, in_axes=(0, None), out_axes=-1)
    k_vmapped_xy = jax.vmap(k_vmapped_x, in_axes=(None, 1), out_axes=-1)
    return jax.jit(k_vmapped_xy)


@jax.jit
def _binom(n, k):
    a = _factorial(n)
    b = _factorial(n - k)
    c = _factorial(k)
    return a / (b * c)


@jax.jit
def _factorial(n):
    return jax.lax.exp(jax.lax.lgamma(n + 1.0))
