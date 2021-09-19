"""This end-to-end test verifies that the ODE filters can be used for large PDEs."""

import tornadox

# 0.04 -> 20k dimensions
fhn = tornadox.ivp.fhn_2d(bbox=[[-1.0, -1.0], [1.0, 1.0]], dx=0.02)
steprule = tornadox.step.AdaptiveSteps(abstol=1e-3, reltol=1e-2)
solver = tornadox.ek1.DiagonalEK1(
    num_derivatives=4,
    steprule=steprule,
    initialization=tornadox.init.RungeKutta(use_df=False),
)

dt = tornadox.step.propose_first_dt(f=fhn.f, t0=fhn.t0, y0=fhn.y0)
init = solver.initialize(fhn)
solver.post_initialize(fhn)
step_attempted, info_attempted = solver.attempt_step(init, dt, *fhn)
step_performed, _, info_performed = solver.perform_full_step(init, dt, *fhn)

# %timeit solver.initialize(fhn)
# %timeit solver.attempt_step(init, dt, *fhn)
# %timeit solver.perform_full_step(init, dt, *fhn)

print(info_attempted, info_performed)
