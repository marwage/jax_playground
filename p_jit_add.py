import argparse
import functools

import flax
import jax
import optax
from flax import linen as nn
from flax.training import checkpoints, train_state
from jax import lax
from jax import numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit, with_sharding_constraint
from jax.sharding import Mesh, PartitionSpec


class DotRelu(nn.Module):
    depth: int

    @nn.compact
    def __call__(self, x):
        w = self.param(
            'w',
            nn.with_partitioning(nn.initializers.xavier_normal(),
                                 (None, 'model')), (x.shape[-1], self.depth))

        y = jnp.dot(x, w)
        y = jax.nn.relu(y)
        y = with_sharding_constraint(y, PartitionSpec('data', None))

        return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int)
    parser.add_argument('--num-proc', type=int)
    args = parser.parse_args()

    jax.distributed.initialize(coordinator_address="worker-0:1234",
                               num_processes=args.num_proc,
                               process_id=args.id)

    device_count = jax.device_count()
    print(f"device_count {device_count}")
    device_mesh = mesh_utils.create_device_mesh((device_count // 2, 2))
    mesh = Mesh(devices=device_mesh, axis_names=("data", "model"))

    inp_spec = PartitionSpec('data', None)
    true_spec = PartitionSpec('data', None)

    BATCH = 8
    #  LAYERS = 4
    DEPTH = 1024
    k = jax.random.PRNGKey(args.id)
    inp = jax.random.normal(k, (BATCH, DEPTH))
    true = jax.random.normal(k, (BATCH, DEPTH))

    optimizer = optax.adam(learning_rate=0.001)
    model = DotRelu(DEPTH)

    def init_fn(k, x, model, optimizer):
        variables = model.init(k, x)  # Initialize the model.
        state = train_state.TrainState.create(  # Create a `TrainState`.
            apply_fn=model.apply,
            params=variables['params'],
            tx=optimizer)
        return state

    with mesh:
        abstract_variables = jax.eval_shape(
            functools.partial(init_fn, model=model, optimizer=optimizer), k,
            inp)

    state_spec = nn.get_partition_spec(abstract_variables)
    #  print(state_spec)

    pjit_init_fn = pjit(
        init_fn,
        static_argnums=(2, 3),
        in_axis_resources=(PartitionSpec(None), inp_spec),  # PRNG key and x
        out_axis_resources=state_spec)
    with mesh:
        initialized_state = pjit_init_fn(k, inp, model, optimizer)
    #  print(jax.tree_map(jnp.shape, initialized_state))

    def train_step(state, x, y):
        # A fake loss function.
        def loss(params):
            z = model.apply({'params': params}, x)
            return optax.l2_loss(z, y).sum()

        grad_fn = jax.grad(loss)
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state

    pjit_step_fn = pjit(
        train_step,
        in_axis_resources=(state_spec, inp_spec,
                           true_spec),  # input annotations
        out_axis_resources=state_spec,  # output annotations
    )
    with mesh:
        new_state = pjit_step_fn(initialized_state, inp, true)

    state_bytes = flax.serialization.to_bytes(new_state)
    with open(f"/data/state_{args.id}.msgpack", "wb") as fi:
        fi.write(state_bytes)


if __name__ == "__main__":
    main()
