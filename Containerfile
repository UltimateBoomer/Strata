ARG ROCM_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.0.2
FROM $ROCM_IMAGE

# Our fork of Niko1221/Strata: upstream v0.2.7 plus the disk-backed conversation cache (park files updated in place, read back by streaming, borrowed from), the turn slice (a conversation keeps the engine between its tool calls) and request priorities (fair share in bands, preemption) and model variants (one server switches between quantizations of one model) and load images (a variant's expert load kept in load order on disk, read sequentially) and sessions (a request names its conversation; only sessions' turns are parked, a client yields and drops by request) and parked conversations kept across restarts and variant switches (one budget for every variant's files) and a web console at / in place of the chat page (the turn scheduler, the variants and each parked conversation on the Monitor).
ARG STRATA_REPOSITORY=https://github.com/UltimateBoomer/Strata.git
ARG STRATA_REVISION=03f4c6796b13890375cef835fba22da9e635b4cd
ARG HIP_ARCHITECTURES=gfx1101
ARG BUILD_JOBS=8
# Use the llama.cpp revision pinned by the selected Strata checkout.
# The BLAS development packages the base image lacks; ROCm 10's -full images already carry them (as amdrocm-*).
ARG ROCM_DEV_PACKAGES="hipblas-dev rocblas-dev hipblaslt-dev"
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates git python3 python3-venv libatomic1 libgomp1 \
    $ROCM_DEV_PACKAGES \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/strata
RUN git init . && git remote add origin "$STRATA_REPOSITORY" \
    && git fetch --depth 1 origin "$STRATA_REVISION" && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$STRATA_REVISION" \
    && LLAMA_REVISION="$(awk 'NR == 1 { print $1; exit }' third_party/ggml/VERSION.txt)" \
    && test "${#LLAMA_REVISION}" -eq 40 \
    && git init /opt/llama.cpp \
    && git -C /opt/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git -C /opt/llama.cpp fetch --depth 1 --progress origin "$LLAMA_REVISION" \
    && git -C /opt/llama.cpp checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/llama.cpp rev-parse HEAD)" = "$LLAMA_REVISION"
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt -r requirements-build.txt
ENV PATH=/opt/venv/bin:/opt/rocm/bin:$PATH \
    STRATA_GGUF_PY=/opt/llama.cpp/gguf-py
RUN cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
    -DSTRATA_PREFILL_MMQ=ON \
    -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=ON \
    -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCHITECTURES" \
    -DSTRATA_GGML_DIR=/opt/llama.cpp \
    && cmake --build build-hip --target strata tune_hipblaslt -j "$BUILD_JOBS" \
    && mkdir engine && cp build-hip/strata build-hip/tune_hipblaslt engine/ \
    && rm -rf build-hip .git /opt/llama.cpp/.git
# The DeepSeek-V4.1-Flash engine (config/server-deepseek.json), in its own tree with upstream's release flags
# (packaging/build_hip_linux.sh). Upstream ships it for gfx12 only; its kernel tests pass on gfx1101.
RUN cmake -S . -B build-dsv41 -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
    -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=ON \
    -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON \
    -DCMAKE_HIP_ARCHITECTURES="$HIP_ARCHITECTURES" \
    -DSTRATA_GGML_DIR=/opt/llama.cpp -DSTRATA_BUILD_ID="$STRATA_REVISION" \
    && cmake --build build-dsv41 --target strata-dsv41 -j "$BUILD_JOBS" \
    && cp build-dsv41/strata-dsv41 engine/ \
    && rm -rf build-dsv41
RUN cmake -S tools/vision -B build-vision -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_DIR=/opt/llama.cpp -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=ON \
    && cmake --build build-vision --target strata-vision -j "$BUILD_JOBS" \
    && cp build-vision/bin/strata-vision engine/ \
    && rm -rf build-vision
# The engine has no RUNPATH; ROCm 10's images, unlike ROCm 7's, do not register /opt/rocm/lib with the loader.
RUN echo /opt/rocm/lib > /etc/ld.so.conf.d/zz-strata-rocm.conf && ldconfig \
    && ldd engine/strata engine/strata-dsv41 | (! grep 'not found')
COPY scripts/model-init.py scripts/serve.py scripts/calibrate.py scripts/tune-hipblaslt.sh /usr/local/lib/strata-local/
ENTRYPOINT ["python", "/usr/local/lib/strata-local/serve.py"]
