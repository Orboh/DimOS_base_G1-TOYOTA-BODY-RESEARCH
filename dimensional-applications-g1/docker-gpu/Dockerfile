# G1 x DimOS agentic blueprint -- GPU portable container.
#
# Notes:
#   1) Base = nvidia/cuda + cuDNN runtime, so onnxruntime-gpu's CUDAExecutionProvider
#      actually loads. Required for the perception / embedding pipeline to leave
#      the CPU.
#   2) requirements.txt is the frozen state of the app venv as of 2026-05-19 after
#      removing the CPU `onnxruntime` package so only `onnxruntime-gpu==1.26.0` is
#      installed. Side-effect: numpy=2.4.6, protobuf=7.34.1.
#   3) The image tag keeps its legacy `go2-agentic-gpu` name so already-built
#      images and the shipped tar stay usable; the contents are robot-agnostic.
#
# Build:  docker build -t go2-agentic-gpu -f docker-gpu/Dockerfile .
# Run:    ./docker-gpu/run.sh
# Send:   ./docker-gpu/say.sh "前に進んで"      # or: dimos humancli inside the container

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:/usr/local/bin:/usr/bin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential pkg-config \
        iproute2 iputils-ping netcat-openbsd \
        ca-certificates curl tini \
        libgl1 libglib2.0-0 libgomp1 \
        libturbojpeg0-dev portaudio19-dev \
        libxext6 libxrender1 libxrandr2 libxcursor1 libxinerama1 libxi6 libxfixes3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv

WORKDIR /app

# 1) Python deps -- frozen from the working local venv (GPU variant).
COPY docker-gpu/requirements.txt /app/requirements.txt
RUN uv venv --python python3.12 /app/.venv \
 && uv pip install --python /app/.venv/bin/python --no-cache -r /app/requirements.txt

# 2) Drop the dimos_interface stub into site-packages (upstream packaging gap).
COPY docker-gpu/dimos_interface_stub /tmp/dimos_interface
RUN cp -r /tmp/dimos_interface /app/.venv/lib/python3.12/site-packages/dimos/web/dimos_interface \
 && rm -rf /tmp/dimos_interface

# 3) Application scripts (run_robot.sh, run_sim.sh, say.sh).
COPY run_robot.sh run_sim.sh say.sh /app/dimensional-applications-g1/
RUN chmod +x /app/dimensional-applications-g1/*.sh \
 && ln -s /app/.venv /app/dimensional-applications-g1/.venv

# 4) Entrypoint -- LCM multicast on (host-shared) loopback, then exec.
COPY docker-gpu/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
CMD ["dimos", "run", "unitree-g1-agentic"]
