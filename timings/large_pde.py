"""Does it make sense to jit for large problems?


This script evaluates (and benchmarks) whether the jitted perform step makes sense for LARGE problems.
We can use it to check the efficiency of the implementation scheme in the future.
"""

import tornadox

# 0.04 -> 20k dimensions
fhn = tornadox.ivp.fhn_2d(bbox=[[-1.0, -1.0], [1.0, 1.0]], dx=0.5)
steprule = tornadox.step.AdaptiveSteps(abstol=1e-3, reltol=1e-2)
solver = tornadox.ek1.DiagonalEK1(
    num_derivatives=4,
    steprule=steprule,
    initialization=tornadox.init.CompiledRungeKutta(use_df=False),
)

dt = tornadox.step.propose_first_dt(f=fhn.f, t0=fhn.t0, y0=fhn.y0)
init = solver.initialize(*fhn)
step_attempted, info_attempted = solver.attempt_step(init, dt, *fhn)
step_performed, _, info_performed = solver.perform_full_step(init, dt, *fhn)
step_performed_compiled, _, info_performed_compiled = solver.perform_full_step_compiled(
    init, dt, *fhn
)

# %timeit solver.initialize(*fhn)
# %timeit solver.attempt_step(init, dt, *fhn)  # 46 ms
# %timeit solver.perform_full_step(init, dt, *fhn)  # 769 ms
# %timeit solver.perform_full_step_compiled(init, dt, *fhn)  # 772 ms

print(info_attempted)
print(info_performed)
print(info_performed_compiled)
