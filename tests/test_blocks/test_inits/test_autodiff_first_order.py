"""Tests for exact initialisations."""
import jax
import jax.numpy as jnp
import pytest
import pytest_cases

from tornadox.blocks.inits import autodiff_first_order


def case_taylormode():
    return autodiff_first_order.taylormode


def case_forwardmode_jvp():
    return autodiff_first_order.forwardmode_jvp


def case_reversemode():
    return autodiff_first_order.reversemode


@pytest_cases.parametrize_with_cases("init", cases=".")
@pytest.mark.parametrize("n", (0, 1, 3))
def test_init(init, n):

    d = 2

    u0 = jnp.arange(3, 3 + d, dtype=float)

    def f(u):
        return u * (1.0 - u) * 0.1 - 4

    init = init(f=f, u0=u0, num_derivatives=n)
    assert init.shape == (n + 1, d)

    assert jnp.allclose(init[0, :], u0)
    if n >= 1:
        assert jnp.allclose(init[1, :], f(u0))
    if n >= 2:
        assert jnp.allclose(init[2, :], jax.jacfwd(f)(u0) @ f(u0))
