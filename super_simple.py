import jax
import optax
from jax import numpy as jnp


def main():
    BATCH = 8
    DEPTH = 1024

    k = jax.random.PRNGKey(0)
    inp = jax.random.normal(k, (BATCH, DEPTH))
    true = jax.random.normal(k, (BATCH, DEPTH))
    w = jax.random.normal(k, (DEPTH, DEPTH))
    b = jax.random.normal(k, (1, DEPTH))

    y = jnp.dot(inp, w)
    y = y + b
    print(y.shape)
    z = optax.l2_loss(y, true)


if __name__ == "__main__":
    main()
