import functools

import jax
import optax
from flax import linen as nn
from flax.training import checkpoints, train_state
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit, with_sharding_constraint
from jax.sharding import Mesh, PartitionSpec


class DotRelu(nn.Module):
    depth: int

    @nn.compact
    def __call__(self, x):
        w = self.param('w', nn.initializers.xavier_normal(),
                       (x.shape[-1], self.depth))
        b = self.param('b', nn.initializers.xavier_normal(), (self.depth, 1))

        y = jnp.dot(x, w)
        y = y + jnp.reshape(b, (1, ) * (y.ndim - 1) + (-1, ))
        y = jax.nn.relu(y)

        return y


def main():
    BATCH = 8
    DEPTH = 1024
    k = random.PRNGKey(0)
    x = jax.random.normal(k, (BATCH, DEPTH))
    y = jax.random.normal(k, (BATCH, DEPTH))

    optimizer = optax.adam(learning_rate=0.001)
    model = DotRelu(DEPTH)

    def init_fn(k, x, model, optimizer):
        variables = model.init(k, x)
        state = train_state.TrainState.create(apply_fn=model.apply,
                                              params=variables['params'],
                                              tx=optimizer)
        return state

    state = init_fn(k, x, model, optimizer)

    def train_step(state, x, y):

        def loss(params):
            z = model.apply({'params': params}, x)
            return optax.l2_loss(z, y)

        grad_fn = jax.grad(loss)
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state

    new_state = train_step(state, x, y)
    return

    @functools.partial(pjit,
                       in_axis_resources=(state_spec, x_spec),
                       out_axis_resources=x_spec)
    def pjit_apply_fn(state, x):
        return state.apply_fn({'params': state.params}, x)

    with mesh:
        y = pjit_apply_fn(new_state, x)
    print(type(y))
    print(y.dtype)
    print(y.shape)


if __name__ == "__main__":
    main()
