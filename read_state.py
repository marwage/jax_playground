import flax


def read(i):
    with open(f"data/state_{i}.msgpack", "rb") as fi:
        byt = fi.read()
    return flax.serialization.msgpack_restore(byt)


def prin(state):
    print(state["params"]["w"]["value"].shape)
    #  print(state["params"]["b"]["value"].shape)


def comp(a, b):
    c = a["params"]["w"]["value"]
    d = b["params"]["w"]["value"]
    if (c == d).all():
        print("equal")
    else:
        print("not equal")


def main():
    a = read(0)
    b = read(1)

    prin(a)
    prin(b)

    comp(a, b)


if __name__ == "__main__":
    main()
