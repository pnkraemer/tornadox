"""Can efficient JVPs use sparsity of Jacobians automatically?"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import tornadox

# Set up problem
for N in [5, 500, 5_000, 50_000]:
    bruss = tornadox.ivp.brusselator(N=N)
    t0, y0 = bruss.t0, bruss.y0

    f = lambda x: bruss.f(t0, x)
    df = jax.jacfwd(f)

    f = jax.jit(f)
    df = jax.jit(df)

    # Check naive JVP implementation against efficient JVP implementation

    def jvp_naive(f, df, y, v):
        dfy = df(y)
        return dfy @ v

    def jvp(f, y, v):
        return jax.jvp(f, (y,), (v,))[1]

    fy = f(y0)
    print(f"N={N}")
    if N <= 1000:

        # Plot Jacobian sparsity
        dfy = df(y0)
        plt.title(f"N={N}")
        plt.imshow(dfy)
        plt.colorbar()
        plt.show()

        assert jnp.allclose(jvp_naive(f, df, y0, fy), jvp(f, y0, fy))

        print("Naive:")
        # %timeit jvp_naive(f, df, y=y0, v=fy).block_until_ready()

    print("JVP:")
    # %timeit jvp(f, y=y0, v=fy).block_until_ready()
    print()
