"""This is an ipython-ready script that benchmarks what the fastest way of projecting E0 @ m is.


The results are (perhaps somewhat surprising):

1. It is fastest (in jitted and non-jitted implementation) to store the means in the EK0 as (n,d) matrices instead of (d*n,) vectors
2. Without jit, matmul with a dense (1,n) projection matrix is faster than slicing!
3. With jit, slicing is faster. The slicing operator (which uses jnp.take) and calling mean[0] directly are equally fast,
   and the decision between those are probably better off when made based on readability, not performance.

On a side note:
It seems to me (NK) that the code is better off optimised for jitting not for non-jitting,
because this is the setting where the things that should be faster (slicing)
are actually faster than the naive approaches (projmatmul).
"""

# Regarding the ipython magic commands:
# Uncomment %timeit ...
# before copying the script into your ipython terminal.

import jax
import jax.numpy as jnp

import tornadox

print()


print("Store mean as a (n*d,) vector")

for d in [10, 100, 1_000]:
    n = 8  # nu + 1
    fake_mean = jnp.arange(1, 1 + d * n)

    # Projection styles
    iwp = tornadox.iwp.IntegratedWienerTransition(
        wiener_process_dimension=d, num_derivatives=n - 1
    )
    E0 = iwp.projection_matrix(0)
    e0_matrix = iwp.projection_matrix_1d(0)

    def full_matrix(arr):
        return E0 @ arr

    def vec_trick_matrix(arr):
        return tornadox.ek0.vec_trick_mul_right(e0_matrix, fake_mean)

    assert jnp.allclose(full_matrix(fake_mean), vec_trick_matrix(fake_mean))

    print(f"d={d}")

    # %timeit full_matrix(fake_mean).block_until_ready()
    # %timeit vec_trick_matrix(fake_mean).block_until_ready()
print()


print("Store mean as a (n*d,) vector and jit")

for d in [10, 100, 1_000]:
    n = 8  # nu + 1
    fake_mean = jnp.arange(1, 1 + d * n)

    # Projection styles
    iwp = tornadox.iwp.IntegratedWienerTransition(
        wiener_process_dimension=d, num_derivatives=n - 1
    )
    E0 = iwp.projection_matrix(0)
    e0_matrix = iwp.projection_matrix_1d(0)

    @jax.jit
    def full_matrix(arr):
        return E0 @ arr

    @jax.jit
    def vec_trick_matrix(arr):
        return tornadox.ek0.vec_trick_mul_right(e0_matrix, fake_mean)

    assert jnp.allclose(full_matrix(fake_mean), vec_trick_matrix(fake_mean))

    print(f"d={d}")

    # %timeit full_matrix(fake_mean).block_until_ready()
    # %timeit vec_trick_matrix(fake_mean).block_until_ready()
print()


# For mean as a matrix of things
print("Store mean as a (n,d) matrix")

for d in [10, 100, 1_000, 10_000]:
    n = 8  # nu + 1
    fake_mean_as_matrix = jnp.arange(1, 1 + d * n).reshape((n, d))

    # Projection styles
    iwp = tornadox.iwp.IntegratedWienerTransition(
        wiener_process_dimension=d, num_derivatives=n - 1
    )
    e0_matrix = iwp.projection_matrix_1d(0)
    e0_operator = iwp.projection_operator_1d(0)

    def matrix(arr):
        return e0_matrix @ arr

    def operator(arr):
        return e0_operator @ arr

    def slicing(arr):
        return arr[0]

    assert jnp.allclose(matrix(fake_mean_as_matrix), operator(fake_mean_as_matrix))
    assert jnp.allclose(operator(fake_mean_as_matrix), slicing(fake_mean_as_matrix))

    print(f"d={d}")

    # %timeit matrix(fake_mean_as_matrix).block_until_ready()
    # %timeit operator(fake_mean_as_matrix).block_until_ready()
    # %timeit slicing(fake_mean_as_matrix).block_until_ready()
print()


# For mean as a matrix of things
print("Store mean as a (n,d) matrix and jit")

for d in [10, 100, 1_000, 10_000]:
    n = 8  # nu + 1
    fake_mean_as_matrix = jnp.arange(1, 1 + d * n).reshape((n, d))

    # Projection styles
    iwp = tornadox.iwp.IntegratedWienerTransition(
        wiener_process_dimension=d, num_derivatives=n - 1
    )
    e0_matrix = iwp.projection_matrix_1d(0)
    e0_operator = iwp.projection_operator_1d(0)

    @jax.jit
    def matrix(arr):
        return e0_matrix @ arr

    @jax.jit
    def operator(arr):
        return e0_operator @ arr

    @jax.jit
    def slicing(arr):
        return arr[0]

    assert jnp.allclose(matrix(fake_mean_as_matrix), operator(fake_mean_as_matrix))
    assert jnp.allclose(operator(fake_mean_as_matrix), slicing(fake_mean_as_matrix))

    print(f"d={d}")

    # %timeit matrix(fake_mean_as_matrix).block_until_ready()
    # %timeit operator(fake_mean_as_matrix).block_until_ready()
    # %timeit slicing(fake_mean_as_matrix).block_until_ready()
print()


# Returns the following on a generic, small laptop:
#
# Store mean as a (n*d,) vector
# d=10
# 152 µs ± 3.96 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
# 422 µs ± 46.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# d=100
# 165 µs ± 4.82 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
# 344 µs ± 18 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# d=1000
# 5.14 ms ± 441 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# 404 µs ± 27.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
#
# Store mean as a (n*d,) vector and jit
# d=10
# 3.88 µs ± 66.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 3.15 µs ± 104 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# d=100
# 16.5 µs ± 196 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 3.18 µs ± 47.8 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# d=1000
# 4.79 ms ± 183 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# 3.68 µs ± 78.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#
# Store mean as a (n,d) matrix
# d=10
# 152 µs ± 3.9 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
# 519 µs ± 76.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# 644 µs ± 1.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# d=100
# 168 µs ± 4.32 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
# 538 µs ± 10.4 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# 709 µs ± 7.47 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# d=1000
# 179 µs ± 5.22 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
# 541 µs ± 32.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# 698 µs ± 55.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# d=10000
# 238 µs ± 1.69 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# 492 µs ± 4.59 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# 660 µs ± 4.61 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
#
# Store mean as a (n,d) matrix and jit
# d=10
# 3.2 µs ± 8.92 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 3.34 µs ± 204 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 3.74 µs ± 296 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# d=100
# 3.95 µs ± 243 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 3.4 µs ± 85.8 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 3.44 µs ± 82.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# d=1000
# 9.16 µs ± 192 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 4.06 µs ± 99.3 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 4.07 µs ± 86.8 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# d=10000
# 54.2 µs ± 1.13 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
# 6.66 µs ± 264 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# 6.59 µs ± 52.4 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#
