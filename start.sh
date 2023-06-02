#!/bin/bash

docker run \
    --gpus all \
    --shm-size 10.24gb \
    -v $(pwd)/data:/data \
    --rm \
    -i \
    -t \
    kungfu.azurecr.io/mw-jax:latest \
    /bin/bash
