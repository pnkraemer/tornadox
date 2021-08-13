"""Stepsize selection strategies."""

import jax.numpy as jnp
import abc

def propose_first_dt(ivp):
    norm_y0 = jnp.linalg.norm(ivp.y0)
    norm_dy0 = jnp.linalg.norm(ivp.f(ivp.t0, ivp.y0))
    return 0.01 * norm_y0 / norm_dy0


class StepRule(abc.ABC):
    """Step-size selection rules for ODE solvers."""

    def __init__(self, first_dt):
        self.first_dt = first_dt

    @abc.abstractmethod
    def suggest(
        self,  previous_dt, scaled_error_estimate, local_convergence_rate=None
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def is_accepted(self, scaled_error_estimate):
        raise NotImplementedError

    def errorest_to_norm(
        self, scaled_error_estimate, reference_state
    ):
        raise NotImplementedError


class ConstantSteps(StepRule):
    """Constant step-sizes."""

    def __init__(self, dt):
        self.dt = dt
        super().__init__(first_dt=dt)

    def suggest(
        self,  previous_dt, scaled_error_estimate, local_convergence_rate=None
    ):
        return self.dt

    def is_accepted(self, scaled_error_estimate):
        return True
    #
    # def errorest_to_norm(
    #     self, errorest: ToleranceDiffusionType, reference_state: np.ndarray
    # ):
    #     pass





#
# # since Adaptive Steps are different now...
# class TestAdaptiveStep(unittest.TestCase):
#     """We pretend that we have a solver of local error rate three and see if steps are
#     proposed accordingly."""
#
#     def setUp(self):
#         """Set up imaginative solver of convergence rate 3."""
#         self.atol = 0.1
#         self.rtol = 0.01
#         self.asr = diffeq.stepsize.AdaptiveSteps(
#             firststep=1.0, atol=self.atol, rtol=self.rtol
#         )
#
#     def test_is_accepted(self):
#         errorest = 0.5  # < 1, should be accepted
#         self.assertTrue(self.asr.is_accepted(errorest))
#
#     def test_suggest(self):
#         """If errorest <1, the next step should be larger."""
#         step = 0.55 * random_state.rand()
#         errorest = 0.75
#         sugg = self.asr.suggest(step, errorest, localconvrate=3)
#         self.assertGreater(sugg, step)
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
