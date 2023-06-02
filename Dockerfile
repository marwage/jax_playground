#!/usr/bin/env -S sh -c 'docker build --rm -t kungfu.azurecr.io/mw-jax:latest -f $0 .'

FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

WORKDIR /workspace

RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip
RUN apt-get update
RUN apt-get install -y python3-pip

RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install rich \
    flax

ADD . /workspace
