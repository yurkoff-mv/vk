#!/bin/bash
DOCKER_TAG="yurkoff/vk-llm:0.1-gpu"
BASE_IMAGE="nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu18.04"
PYTHON_VERSION=3.8
DOCKER_BUILDKIT=1 docker build --file Dockerfile --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg PYTHON_VERSION=$PYTHON_VERSION -t $DOCKER_TAG .
