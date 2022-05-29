"""Tests for the Kronecker EK0."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from tornadox.blocks.sde import ibm
from tornadox.blocks.step_impl import ek0_kronecker


@partial(jax.jit, static_argnames=("n", "d"))
def _setup(*, n, d):
    """Setup for derivative-dimension ordering."""

    # Preconditioners
    p, p_inv = ibm.preconditioner_diagonal(dt=0.1, num_derivatives=n)

    # System matrices
    a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=n)

    # Some mean and covariance
    m = jnp.arange(d * (n + 1)).reshape((n + 1, d))
    c_sqrtm = jnp.arange((n + 1) ** 2).reshape(((n + 1), (n + 1)))

    return (p, p_inv), (a, q_sqrtm), (m, c_sqrtm)


@pytest.mark.parametrize("n", (3,))
@pytest.mark.parametrize("d", (2,))
def test_ek0_attempt_step_forward_only(n, d):
    (p, p_inv), (a, q_sqrtm), (m, c_sqrtm) = _setup(n=n, d=d)
    out = ek0_kronecker.attempt_step_forward_only(
        f=lambda x: x * (1 - x),
        m=m,
        c_sqrtm=c_sqrtm,
        p=p,
        p_inv=p_inv,
        a=a,
        q_sqrtm=q_sqrtm,
    )
    cor, obs, err = out

    m, c_sqrtm = cor
    assert m.shape == (n + 1, d)
    assert c_sqrtm.shape == ((n + 1), (n + 1))

    m, c_sqrtm = obs
    assert m.shape == (d,)
    assert c_sqrtm.shape == ()

    assert err.shape == ()


@pytest.mark.parametrize("n", (3,))
@pytest.mark.parametrize("d", (2,))
def test_ek0_attempt_step(n, d):
    (p, p_inv), (a, q_sqrtm), (m, c_sqrtm) = _setup(n=n, d=d)
    out = ek0_kronecker.attempt_step(
        f=lambda x: x * (1 - x),
        m=m,
        c_sqrtm=c_sqrtm,
        p=p,
        p_inv=p_inv,
        a=a,
        q_sqrtm=q_sqrtm,
    )
    cor, obs, ext, bw, err = out

    m, c_sqrtm = cor
    assert m.shape == (n + 1, d)
    assert c_sqrtm.shape == ((n + 1), (n + 1))

    m, c_sqrtm = obs
    assert m.shape == (d,)
    assert c_sqrtm.shape == ()

    m, c_sqrtm = ext
    assert m.shape == (n + 1, d)
    assert c_sqrtm.shape == ((n + 1), (n + 1))

    g, (m, c_sqrtm) = bw
    assert m.shape == (n + 1, d)
    assert c_sqrtm.shape == ((n + 1), (n + 1))
    assert g.shape == ((n + 1), (n + 1))

    assert err.shape == ()
