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
