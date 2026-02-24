#!/bin/bash

# Set environment variables for megakernel build
MEGAKERNELS_ROOT="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
THUNDERKITTENS_ROOT=$MEGAKERNELS_ROOT/ThunderKittens
PYTHON_VERSION=3.12
GPU=B200
