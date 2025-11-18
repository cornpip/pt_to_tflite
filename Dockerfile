FROM tensorflow/tensorflow:2.13.0-gpu

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && python -m pip install --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir onnx onnx-tf tensorflow-addons tensorflow-probability \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
