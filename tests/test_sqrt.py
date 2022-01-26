"""Tests for square-root utilities."""

import jax.numpy as jnp
import pytest

import tornadox


@pytest.fixture
def iwp():
    """Steal system matrices from an IWP transition."""
    return tornadox.iwp.IntegratedWienerTransition(
        wiener_process_dimension=1, num_derivatives=1
    )


@pytest.fixture
def H_and_SQ(iwp, measurement_style):
    """Measurement model via IWP system matrices."""
    H, SQ = iwp.preconditioned_discretize_1d

    if measurement_style == "full":
        return H, SQ
    return H[:1], SQ[:1, :1]


@pytest.fixture
def SC(iwp):
    """Initial covariance via IWP process noise."""
    return iwp.preconditioned_discretize_1d[1]


@pytest.fixture
def batch_size():
    """Batch size > 1. Test batched transitions."""
    return 5


@pytest.mark.parametrize("measurement_style", ["full", "partial"])
def test_propagate_cholesky_factor(H_and_SQ, SC, measurement_style):
    """Assert that sqrt propagation coincides with non-sqrt propagation."""
    H, SQ = H_and_SQ

    # First test: Non-optional S2
    chol = tornadox.sqrt.propagate_cholesky_factor(S1=(H @ SC), S2=SQ)
    cov = H @ SC @ SC.T @ H.T + SQ @ SQ.T
    assert jnp.allclose(chol @ chol.T, cov)
    assert jnp.allclose(jnp.tril(chol), chol)


@pytest.mark.parametrize("measurement_style", ["full", "partial"])
def test_batched_propagate_cholesky_factors(
    H_and_SQ, SC, measurement_style, batch_size
):
    """Batched propagation coincides with non-batched propagation."""

    H, SQ = H_and_SQ
    H = tornadox.experimental.linops.BlockDiagonal(jnp.stack([H] * batch_size))
    SQ = tornadox.experimental.linops.BlockDiagonal(jnp.stack([SQ] * batch_size))
    SC = tornadox.experimental.linops.BlockDiagonal(jnp.stack([SC] * batch_size))

    chol = tornadox.sqrt.batched_propagate_cholesky_factor(
        (H @ SC).array_stack, SQ.array_stack
    )
    chol_as_bd = tornadox.experimental.linops.BlockDiagonal(chol)
    reference = tornadox.sqrt.propagate_cholesky_factor(
        (H @ SC).todense(), SQ.todense()
    )
    assert jnp.allclose(chol_as_bd.todense(), reference)


@pytest.mark.parametrize("measurement_style", ["full", "partial"])
def test_batched_sqrtm_to_cholesky(H_and_SQ, SC, measurement_style, batch_size):
    """Sqrtm-to-cholesky is the same for batched and non-batched."""
    H, SQ = H_and_SQ
    d = H.shape[0]
    H = tornadox.experimental.linops.BlockDiagonal(jnp.stack([H] * batch_size))
    SC = tornadox.experimental.linops.BlockDiagonal(jnp.stack([SC] * batch_size))

    chol = tornadox.sqrt.batched_sqrtm_to_cholesky((H @ SC).T.array_stack)
    chol_as_bd = tornadox.experimental.linops.BlockDiagonal(chol)

    reference = tornadox.sqrt.sqrtm_to_cholesky((H @ SC).T.todense())
    assert jnp.allclose(chol_as_bd.todense(), reference)
    assert chol_as_bd.array_stack.shape == (batch_size, d, d)


@pytest.mark.parametrize("measurement_style", ["full", "partial"])
def test_update_sqrt(H_and_SQ, SC, measurement_style):
    """Sqrt-update coincides with non-square-root update."""

    H, _ = H_and_SQ

    SC_new, kalman_gain, innov_chol = tornadox.sqrt.update_sqrt(H, SC)
    assert isinstance(SC_new, jnp.ndarray)
    assert isinstance(kalman_gain, jnp.ndarray)
    assert isinstance(innov_chol, jnp.ndarray)
    assert SC_new.shape == SC.shape
    assert kalman_gain.shape == (H.shape[1], H.shape[0])
    assert innov_chol.shape == (H.shape[0], H.shape[0])

    # expected:
    S = H @ SC @ SC.T @ H.T
    K = SC @ SC.T @ H.T @ jnp.linalg.inv(S)
    C = SC @ SC.T - K @ S @ K.T

    # Test SC
    assert jnp.allclose(SC_new @ SC_new.T, C)
    assert jnp.allclose(SC_new, jnp.tril(SC_new))

    # Test K
    assert jnp.allclose(K, kalman_gain)

    # Test S
    assert jnp.allclose(innov_chol @ innov_chol.T, S)
    assert jnp.allclose(innov_chol, jnp.tril(innov_chol))


@pytest.mark.parametrize("measurement_style", ["full", "partial"])
def test_batched_update_sqrt(H_and_SQ, SC, measurement_style, batch_size):
    """Batched updated coincides with non-batched update."""
    H, _ = H_and_SQ
    d_out, d_in = H.shape
    H = tornadox.experimental.linops.BlockDiagonal(jnp.stack([H] * batch_size))
    SC = tornadox.experimental.linops.BlockDiagonal(jnp.stack([SC] * batch_size))

    chol, K, S = tornadox.sqrt.batched_update_sqrt(
        H.array_stack,
        SC.array_stack,
    )
    assert isinstance(chol, jnp.ndarray)
    assert isinstance(K, jnp.ndarray)
    assert isinstance(S, jnp.ndarray)
    assert K.shape == (batch_size, d_in, d_out)
    assert chol.shape == (batch_size, d_in, d_in)
    assert S.shape == (batch_size, d_out, d_out)

    ref_chol, ref_K, ref_S = tornadox.sqrt.update_sqrt(H.todense(), SC.todense())
    chol_as_bd = tornadox.experimental.linops.BlockDiagonal(chol)
    K_as_bd = tornadox.experimental.linops.BlockDiagonal(K)
    S_as_bd = tornadox.experimental.linops.BlockDiagonal(S)

    # K can be compared elementwise, S and chol not (see below).
    assert jnp.allclose(K_as_bd.todense(), ref_K)

    # The Cholesky-factor of positive semi-definite matrices is only unique
    # up to column operations (e.g. column reordering), i.e. there could be slightly
    # different Cholesky factors in batched and non-batched versions.
    # Therefore, we only check that the results are valid Cholesky factors themselves
    assert jnp.allclose((S_as_bd @ S_as_bd.T).todense(), ref_S @ ref_S.T)
    assert jnp.allclose((chol_as_bd @ chol_as_bd.T).todense(), ref_chol @ ref_chol.T)
