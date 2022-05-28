"""Tests for traditional EK1 implementations."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from tornadox.blocks.sde import ibm
from tornadox.blocks.step_impl.first_order import ek1_projmatfree_d_nu


@partial(jax.jit, static_argnames=("n", "d"))
def _setup(*, n, d):
    """Setup for derivative-dimension ordering."""

    # Preconditioners
    p, p_inv = ibm.preconditioner_diagonal(dt=0.1, num_derivatives=n)
    p, p_inv = jnp.tile(p, reps=d), jnp.tile(p_inv, reps=d)

    # System matrices
    eye_d = jnp.eye(d)
    a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=n)
    a, q_sqrtm = jnp.kron(eye_d, a), jnp.kron(eye_d, a)

    # Projection matrices
    e0, e1, *_ = jnp.eye(n + 1)[:, None, :]
    e0, e1 = jnp.kron(eye_d, e0), jnp.kron(eye_d, e1)

    # Some mean and covariance
    m = jnp.arange(1, 1 + d * (n + 1))
    c_sqrtm = jnp.arange(1, 1 + d**2 * (n + 1) ** 2).reshape(
        (d * (n + 1), d * (n + 1))
    )

    return (p, p_inv), (a, q_sqrtm), (e0, e1), (m, c_sqrtm)


@pytest.mark.parametrize("n", (3,))
@pytest.mark.parametrize("d", (2,))
def test_ek1_attempt_step_projmatfree_d_nu_forward_only(n, d):
    (p, p_inv), (a, q_sqrtm), _, (m, c_sqrtm) = _setup(n=n, d=d)
    out = ek1_projmatfree_d_nu.attempt_step_forward_only(
        f=lambda x: jnp.flip(x) * (1 - x),
        df=lambda x: 1 - 2 * x,
        m=m,
        c_sqrtm=c_sqrtm,
        p=p,
        p_inv=p_inv,
        a=a,
        q_sqrtm=q_sqrtm,
        num_derivatives=n,
    )

    cor, obs, err = out

    m, c_sqrtm = cor
    assert m.shape == (d * (n + 1),)
    assert c_sqrtm.shape == (d * (n + 1), d * (n + 1))

    m, c_sqrtm = obs
    assert m.shape == (d,)
    assert c_sqrtm.shape == (d, d)

    assert err.shape == (d,)


@pytest.mark.parametrize("n", (3,))
@pytest.mark.parametrize("d", (2,))
def test_ek1_attempt_step_projmatfree_d_nu(n, d):
    (p, p_inv), (a, q_sqrtm), _, (m, c_sqrtm) = _setup(n=n, d=d)
    out = ek1_projmatfree_d_nu.attempt_step(
        f=lambda x: jnp.flip(x) * (1 - x),
        df=lambda x: 1 - 2 * x,
        m=m,
        c_sqrtm=c_sqrtm,
        p=p,
        p_inv=p_inv,
        a=a,
        q_sqrtm=q_sqrtm,
        num_derivatives=n,
    )

    cor, obs, ext, bw, err = out

    m, c_sqrtm = cor
    assert m.shape == (d * (n + 1),)
    assert c_sqrtm.shape == (d * (n + 1), d * (n + 1))

    m, c_sqrtm = ext
    assert m.shape == (d * (n + 1),)
    assert c_sqrtm.shape == (d * (n + 1), d * (n + 1))

    m, c_sqrtm = obs
    assert m.shape == (d,)
    assert c_sqrtm.shape == (d, d)

    g, (m, c_sqrtm) = bw
    assert m.shape == (d * (n + 1),)
    assert c_sqrtm.shape == (d * (n + 1), d * (n + 1))
    assert g.shape == (d * (n + 1), d * (n + 1))

    assert err.shape == (d,)
