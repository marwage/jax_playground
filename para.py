import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding


def lay(inp):
    w = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
    b = jax.random.normal(jax.random.PRNGKey(0), (8192, ))
    y = jnp.matmul(inp, w) + b
    return jax.nn.relu(y)


def sharding():
    sharding = PositionalSharding(mesh_utils.create_device_mesh((4, 1)))
    x = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
    print(type(x))
    y = jax.device_put(x, sharding)
    print(type(y))
    jax.debug.visualize_array_sharding(y)
    z = jnp.sin(y)
    print(type(z))
    jax.debug.visualize_array_sharding(z)


def main():
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 8192))
    p_lay = jax.pmap(lay)
    y = p_lay(x)
    print(y.shape)


if __name__ == "__main__":
    main()
