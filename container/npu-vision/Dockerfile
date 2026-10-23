# The NPU picture reader built from public sources only, so anyone can rebuild it:
# Intel's published driver packages for Ubuntu 24.04, Ubuntu's own Level Zero loader,
# and one pinned OpenVINO wheel.  The pinned versions are the ones the reader was
# measured with (docs/INTEL_NPU.md, 2026-10-23).  The kernel side stays on the host.
#
# A machine whose host driver differs a lot from this pin may need the version-matched
# build instead: Dockerfile.host-libs (README says when to prefer it).
FROM ubuntu:24.04

ARG NPU_DRIVER_TAG=v1.10.1
ARG NPU_DRIVER_VER=1.10.1.20241220-12430270326

# libze1 carries the Level Zero loader, which the NPU plugin dlopens: without it the
# NPU silently vanishes from the device list (docs/INTEL_NPU.md).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libze1 python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Intel's own NPU userspace, exactly as published for Ubuntu 24.04: the Level Zero
# driver and the compiler.  No firmware here: the host's kernel loads it for the
# device, and the container only receives the /dev/accel node.
RUN for p in intel-level-zero-npu intel-driver-compiler-npu; do \
        curl -fsSL -o "/tmp/$p.deb" \
            "https://github.com/intel/linux-npu-driver/releases/download/$NPU_DRIVER_TAG/${p}_${NPU_DRIVER_VER}_ubuntu24.04_amd64.deb"; \
    done \
    && apt-get update \
    && apt-get install -y /tmp/intel-level-zero-npu.deb /tmp/intel-driver-compiler-npu.deb \
    && rm -rf /tmp/*.deb /var/lib/apt/lists/*

# One venv carries the pinned OpenVINO (runtime + NPU plugin), numpy and pillow.
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir openvino==2026.4.1 numpy pillow

# The Strata reader, exporter and health check (staged by fetch_tools.sh), and the
# entrypoint that answers the server's READY / ENC / OK protocol on stdin/stdout.
COPY vendor/ /opt/strata/
COPY entrypoint.sh /usr/local/bin/strata-vision-npu
ENTRYPOINT ["/usr/local/bin/strata-vision-npu"]
