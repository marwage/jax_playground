import jax


def main():
    jax.distributed.initialize(coordinator_address="127.0.0.1:1234",
                               num_processes=1,
                               process_id=0)

    xs = jax.numpy.ones(16)
    p_sum = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')
    y = p_sum(xs)
    print(y)


if __name__ == "__main__":
    main()
