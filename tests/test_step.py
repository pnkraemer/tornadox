"""Tests for stepsize selection."""

import jax.numpy as jnp
import tornado

def test_propose_first_dt():

    ivp = tornado.ivp.vanderpol()

    dt = tornado.step.propose_first_dt(ivp)
    assert dt > 0


def test_constant_steps():
    dt = 0.1
    steprule = tornado.step.ConstantSteps(dt)

    proposed = steprule.suggest(previous_dt=jnp.nan, scaled_error_estimate=0.1)
    assert proposed == 0.1
    assert steprule.is_accepted(scaled_error_estimate=0.1)

def test_adaptive_steps():
    abstol = 0.1
    reltol = 0.01
    steprule = tornado.step.AdaptiveSteps(first_dt=0.1, abstol=abstol, reltol=reltol)
    assert isinstance(steprule, tornado.step.AdaptiveSteps)

    # < 1 accepted, > 1 rejected
    assert steprule.is_accepted(scaled_error_estimate=0.99)
    assert not steprule.is_accepted(scaled_error_estimate=1.01)

    # Accepting a step makes the next one larger
    assert steprule.suggest(previous_dt=0.3, scaled_error_estimate=0.5, local_convergence_rate=2) > 0.3
    assert steprule.suggest(previous_dt=0.3, scaled_error_estimate=2.0, local_convergence_rate=2) < 0.3

    # Error estimation to scaled norm 1d
    unscaled_error_estimate = jnp.array([0.5])
    reference_state = jnp.array([2.0])
    E = steprule.errorest_to_norm(unscaled_error_estimate=unscaled_error_estimate, reference_state=reference_state)
    scaled_error = unscaled_error_estimate / (abstol + reltol * reference_state)
    assert jnp.allclose(E, scaled_error)

    # Error estimation to scaled norm 2d
    unscaled_error_estimate = jnp.array([0.5, 0.6])
    reference_state = jnp.array([2.0, 3.0])
    E = steprule.errorest_to_norm(unscaled_error_estimate=unscaled_error_estimate, reference_state=reference_state)
    scaled_error = jnp.linalg.norm(unscaled_error_estimate / (abstol + reltol * reference_state)) / jnp.sqrt(2)
    assert jnp.allclose(E, scaled_error)

#
# # since Adaptive Steps are different now...
# class TestAdaptiveStep(unittest.TestCase):
#     """We pretend that we have a solver of local error rate three and see if steps are
#     proposed accordingly."""
#
#     def test_errorest_to_norm_1d(self):
#         errorest = 0.5
#         reference_state = np.array(2.0)
#         expected = errorest / (self.atol + self.rtol * reference_state)
#         received = self.asr.errorest_to_norm(errorest, reference_state)
#         self.assertAlmostEqual(expected, received)
#
#     def test_errorest_to_norm_2d(self):
#         errorest = np.array([0.1, 0.2])
#         reference_state = np.array([2.0, 3.0])
#         expected = np.linalg.norm(
#             errorest / (self.atol + self.rtol * reference_state)
#         ) / np.sqrt(2)
#         received = self.asr.errorest_to_norm(errorest, reference_state)
#         self.assertAlmostEqual(expected, received)
#
#     def test_minstep_maxstep(self):
#         adaptive_steps = diffeq.stepsize.AdaptiveSteps(
#             firststep=1.0,
#             limitchange=(0.0, 1e10),
#             minstep=0.1,
#             maxstep=10,
#             atol=1,
#             rtol=1,
#         )
#
#         with self.assertRaises(RuntimeError):
#             adaptive_steps.suggest(
#                 laststep=1.0, scaled_error=100_000.0, localconvrate=1
#             )
#         with self.assertRaises(RuntimeError):
#             adaptive_steps.suggest(
#                 laststep=1.0, scaled_error=1.0 / 100_000.0, localconvrate=1
#             )
#
#
