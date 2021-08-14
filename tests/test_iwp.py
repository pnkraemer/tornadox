import jax.numpy as jnp
import pytest

import tornado


@pytest.fixture
def iwp():
    return tornado.iwp.IntegratedWienerTransition(
        wiener_process_dimension=1, num_derivatives=2
    )


@pytest.fixture
def dt():
    return 0.1


@pytest.fixture
def ah_22_ibm(dt):
    return jnp.array(
        [
            [1.0, dt, dt ** 2 / 2.0],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def qh_22_ibm(dt):
    return jnp.array(
        [
            [dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
            [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
            [dt ** 3 / 6.0, dt ** 2 / 2.0, dt],
        ]
    )


def test_non_preconditioned_system_matrices(dt, iwp, ah_22_ibm, qh_22_ibm):
    state_transition_matrix, process_noise_cov_chol = iwp.non_preconditioned_discretize(
        dt
    )

    assert jnp.allclose(state_transition_matrix, ah_22_ibm)
    assert jnp.allclose(process_noise_cov_chol @ process_noise_cov_chol.T, qh_22_ibm)


def test_preconditioned_system_matrices(dt, iwp):
    precond_state_trans_mat, precond_proc_noice_chol = iwp.preconditioned_discretize

    (
        non_precond_state_trans_mat,
        non_precond_proc_noice_chol,
    ) = iwp.non_preconditioned_discretize(dt)

    precond, precond_inv = iwp.nordsieck_preconditioner(dt)
    assert jnp.allclose(
        precond @ precond_state_trans_mat @ precond_inv, non_precond_state_trans_mat
    )
    assert jnp.allclose(precond @ precond_proc_noice_chol, non_precond_proc_noice_chol)


def test_projection_matrices(iwp):
    P = iwp.projection_matrix(0)
    assert isinstance(P, jnp.ndarray)
    d, q = iwp.wiener_process_dimension, iwp.num_derivatives
    assert P.shape == (d, q + 1)
    assert (P == 1).sum() == d


def test_reorder_states():
    # Transition handles reordering
    iwp = tornado.iwp.IntegratedWienerTransition(
        num_derivatives=1, wiener_process_dimension=3
    )

    # 11 ~ derivative of 1, 22 ~ derivative of 2
    arr = jnp.array([1, 2, 3, 11, 22, 33])

    new_arr = iwp.reorder_state_from_derivative_to_coordinate(arr)
    expected = jnp.array([1, 11, 2, 22, 3, 33])
    for r, e in zip(new_arr, expected):
        assert r == e

    old_arr = iwp.reorder_state_from_coordinate_to_derivative(new_arr)
    for r, e in zip(old_arr, arr):
        assert r == e
