import celluloid
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

import tornadox

bruss = tornadox.ivp.brusselator(N=100)
tol = 1e-8
ek1 = tornadox.ek1.ReferenceEK1(
    steprule=tornadox.step.AdaptiveSteps(abstol=1e-2 * tol, reltol=tol),
    initialization=tornadox.init.RungeKutta(method="Radau"),
    num_derivatives=4,
)

ms = []
ts = []

fig, _ = plt.subplots()

camera = celluloid.Camera(fig)

for state, _ in ek1.solution_generator(bruss):
    ms.append(state.y.mean[0])
    ts.append(state.t)

    t = state.t

    plt.title("LogScale(vmin=1e-10, vmax=1e0, clip=True)")
    plt.imshow(
        state.y.cov, norm=LogNorm(vmin=1e-10, vmax=1e0, clip="True"), cmap=cm.Greys
    )
    camera.snap()

animation = camera.animate()
animation.save("animation.mp4")


means = jnp.stack(ms)
ts = jnp.stack(ts)

plt.subplots(dpi=200)
plt.title(len(ts))
plt.plot(ts, means)
# plt.show()
#
# fig, ax = plt.subplots(dpi=200)
# stride = 2
# for t, cov in zip(ts[::stride], covariances[::stride]):
#     plt.title(t)
#     ax.imshow(cov)
#     plt.pause(0.001)
# plt.show()
