# syntax = docker/dockerfile:experimental
ARG BASE_IMAGE=ubuntu:rolling

# Note:
# Define here the default python version to be used in all later build-stages as default.
# ARG and ENV variables do not persist across stages (they're build-stage scoped).
# That is crucial for ARG PYTHON_VERSION, which otherwise becomes "" leading to nasty bugs,
# that don't let the build fail, but break current version handling logic and result
# in images with wrong python version. To fix that, we will restate the ARG PYTHON_VERSION
# on each build-stage.
ARG PYTHON_VERSION=3.8

FROM ${BASE_IMAGE} AS compile-image
ARG BASE_IMAGE=ubuntu:rolling
ARG PYTHON_VERSION
ENV PYTHONUNBUFFERED TRUE

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install software-properties-common -y && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt remove python-pip  python3-pip && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        ca-certificates \
        g++ \
        python3-distutils \
        python$PYTHON_VERSION \
        python$PYTHON_VERSION-dev \
        python$PYTHON_VERSION-venv \
        openjdk-11-jdk-headless \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Make the virtual environment and "activating" it by adding it first to the path.
# From here on the python$PYTHON_VERSION interpreter is used and the packages
# are installed in /home/venv which is what we need for the "runtime-image"
RUN python$PYTHON_VERSION -m venv /home/venv
ENV PATH="/home/venv/bin:$PATH"

# This is only useful for cuda env
RUN export USE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

ARG CUDA_VERSION="118"
COPY ./app/requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN python -m pip install -U pip setuptools
RUN python -m pip install --no-cache-dir torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt



# Final image for production
FROM ${BASE_IMAGE} AS runtime-image
# Re-state ARG PYTHON_VERSION to make it active in this build-stage (uses default define at the top)
ARG PYTHON_VERSION
ARG USERNAME=model-server
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ENV PYTHONUNBUFFERED TRUE
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}



RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install software-properties-common -y && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt remove python-pip  python3-pip && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    sudo \
    nano \
    openssh-server \
    openssh-client \
    git \
    git-lfs \
    python$PYTHON_VERSION \
    python3-distutils \
    python$PYTHON_VERSION-dev \
    python$PYTHON_VERSION-venv \
    # using openjdk-17-jdk due to circular dependency(ca-certificates) bug in openjdk-17-jre-headless debian package
    # https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=1009905
    openjdk-11-jdk-headless \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp

#https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -G sudo -m $USERNAME \
    && mkdir -p /home/$USERNAME/tmp \
    && mkdir -p /home/$USERNAME/models \
#    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
#RUN adduser $USERNAME sudo

COPY --chown=model-server --from=compile-image /home/venv /home/venv
COPY --chown=model-server app/ /home/model-server/
COPY --chown=model-server models/ /home/model-server/models

ENV PATH="/home/venv/bin:$PATH"

#USER $USERNAME
WORKDIR /home/model-server

EXPOSE 8080

CMD ["python", "./vk_GPTQ.py"]
