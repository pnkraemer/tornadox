"""Tests for exact initialisations of second order ODEs."""

import jax.numpy as jnp
import pytest
import pytest_cases

from tornadox import ivp_examples
from tornadox.blocks.inits import autodiff_first_order, autodiff_second_order


def case_reversemode():
    return autodiff_second_order.reversemode


def case_forwardmode_jvp():
    return autodiff_second_order.forwardmode_jvp


def case_taylormode():
    return autodiff_second_order.taylormode


@pytest_cases.parametrize_with_cases("init", cases=".")
@pytest.mark.parametrize("n", (0, 1, 4))
def test_init(init, n):

    f, _, u0 = ivp_examples.vanderpol()
    f_2nd, _, (u0_2nd, du0_2nd) = ivp_examples.vanderpol_second_order()

    init_reference = autodiff_first_order.forwardmode_jvp(f=f, u0=u0, num_derivatives=n)
    init_2nd = init(f=f_2nd, u0=u0_2nd, du0=du0_2nd, num_derivatives=n)

    assert jnp.allclose(init_2nd, init_reference[:, : u0_2nd.shape[0]])
