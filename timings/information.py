import jax.numpy as jnp
import matplotlib.pyplot as plt

import tornadox

vdp = tornadox.ivp.vanderpol(stiffness_constant=0.1)
tol = 1e-4
ek1 = tornadox.ek1.InformationEK1(
    steprule=tornadox.step.AdaptiveSteps(abstol=tol, reltol=tol),
    initialization=tornadox.init.RungeKutta(method="Radau"),
    num_derivatives=2,
)

ms = []
ts = []
for state, _ in ek1.solution_generator(vdp):
    ms.append(state.y.mean()[0])
    ts.append(state.t)

means = jnp.stack(ms)
ts = jnp.stack(ts)

plt.subplots(dpi=200)
plt.plot(ts, means)
plt.show()
