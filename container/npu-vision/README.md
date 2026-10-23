# NPU picture reader in a container

The picture reader from [docs/INTEL_NPU.md](../../docs/INTEL_NPU.md) runs inside an OCI
container with the real NPU. The kernel side (the `intel_vpu` module and the firmware) stays
on the host; the container gets the device node plus the userspace driver.

Measured on this PC (Lenovo with the Arrow-Lake NPU, kernel `intel_vpu` 1.0, userspace
1.10.1, OpenVINO 2026.4.1, podman 4 under Ubuntu 24.04): the chain of two IRs inside a rootful
container gives mean cosine 0.998390, min 0.967598 against the engine's own
`strata-vision` embeddings - the same numbers as on bare metal - warm encode 2.3 s,
`OK 576 24 24 2231` for a 768x768 picture.  Those numbers are from the version-matched
build; the public build (same driver version from Intel's packages, Ubuntu 24.04 base)
passed the same check on this machine on 2026-10-24.

## Published image

```sh
podman pull ghcr.io/sibnick/strata-npu-vision:latest     # or: docker pull
```

The published image is built by GitHub Actions from the public recipe below, from the
`npu-vision-reader` branch of this repository; tags carry the driver and OpenVINO versions
(`driver1.10.1-ov2026.4.1`).  It is an experimental artifact for trying the NPU reader,
versioned by what it contains, not a release component of Strata.

## Build from public sources (the published recipe)

Intel's published driver packages for Ubuntu 24.04, Ubuntu's Level Zero loader, one pinned
OpenVINO wheel.  Needs internet, nothing else:

```sh
container/npu-vision/fetch_tools.sh                      # stage the reader, exporter, health check
podman build -t strata-npu-vision container/npu-vision
```

The driver and OpenVINO versions are `ARG`s (`NPU_DRIVER_TAG`, `NPU_DRIVER_VER`); the pin
is what this reader was measured with.  Intel changed the release packaging after this
era, so newer driver versions need the build below unless you adapt the download.

## Build version-matched to your host

When the host driver is far from the pin (or your distro ships the driver as RPM), put
your own host's userspace libraries into the image, so the container matches the running
kernel module exactly - the ioctl ABI between them is versioned:

```sh
container/npu-vision/fetch_host_libs.sh                  # collect host driver libs + the reader
podman build -t strata-npu-vision -f container/npu-vision/Dockerfile.host-libs container/npu-vision
```

Re-run both after a host NPU driver update: matching is the point.

## Health check, then use

The device node `/dev/accel/accel0` is `root:render`. **Rootful** containers open it
directly; **rootless** ones see `nobody:nogroup` and fail with a permission error - measured.
Two ways to use the NPU:

```sh
# 1) rootful podman (what Intel's own docker steps effectively are)
sudo podman run --rm --device /dev/accel/accel0 --entrypoint python \
    strata-npu-vision /opt/strata/npu_health.py
# expect: devices: ['CPU', 'NPU'] and "NPU inference inside the container OK"

# 2) serve pictures the way the server serves them to a vision exe (stdin/stdout protocol)
sudo podman run -i --rm \
    -v "$HOME/models/npu-vision:/ir:ro" \
    -v /path/to/pictures:/work \
    strata-npu-vision
# then type:  ENC /work/pic.png /work/pic.sve   and read the OK line;  QUIT ends.
# /work must be writable: the .sve output is written into it.
```

The IR files come from `tools/vision/export_npu_encoder.py` (docs/INTEL_NPU.md says how);
the entrypoint defaults to `/ir/encoder-1.xml:/ir/encoder-2.xml`, so mount the directory
that holds them at `/ir`.

Rootless podman can work with a one-time root setup (`echo "$USER:109:1" | sudo tee -a
/etc/subgid`, then `--gidmapping 109:109:1 --group-add keep-groups`) - that route was
proposed from the podman documentation and is **not yet validated** on this machine.

Without `--device` (or rootless without the mapping) the reader still works on the CPU:
add `--device CPU` as the reader's own argument (`strata-npu-vision --device CPU`), which is
the fallback the server would see as a slow but correct reader.  Real docker was not tested
on this machine; docker's `--device` route follows Intel's own published steps.

## Cache volume

The first start compiles both graphs on the host CPU: measured 170 s + 190 s of one core at
100%.  Two caches exist (docs/INTEL_NPU.md, "Startup compile and its caches"): the driver's
own cache under the home directory holds exactly one of the two graphs (a volume on it
helps one graph only), while OpenVINO's `CACHE_DIR` keeps both and brings the next start to
1.2 s + 1.3 s.  Point it at the cache volume:

```sh
sudo podman run -i --rm --device /dev/accel/accel0 \
    -e STRATA_VISION_NPU_CACHE_DIR=/root/.cache/openvino \
    -v strata-npu-cache:/root/.cache \
    -v "$HOME/models/npu-vision:/ir:ro" -v /path/to/pictures:/work \
    strata-npu-vision
```

## As the server's vision.exe

The reader speaks READY / ENC / OK on stdin/stdout, which `podman run -i` passes through
unchanged.  Two details decide the mounts.  The server starts the reader with its working
directory set to the server's vision work directory and names picture and output files
*relative to it* (`serve/server.py`, `Vision._start_reader` and the `ENC` line, issue #480) -
so the shim must mount that directory, read-write (the `.sve` output lands there), and set
the container's working directory to the mount.  The shim is started by the server itself,
so inside the shim `$PWD` is exactly that directory.  A shim script with

```sh
#!/bin/sh
exec sudo podman run -i --rm --device /dev/accel/accel0 \
    -v strata-npu-cache:/root/.cache \
    -v "$HOME/models/npu-vision:/ir:ro" \
    -v "$PWD:/work" -w /work \
    strata-npu-vision "$@"
```

is a drop-in for the `vision.exe` named in the server config: wherever that names
`tools/vision/strata_vision_npu.py` (a runner script, for example), name the shim instead
and pictures are served from the container.
If the NPU reader does not start, the server falls back to the CPU reader when the config
names one (`vision.cpu_exe`); a runner can add that automatically when a `strata-vision`
binary was built next to the engine.
Steady-state CPU while the NPU encodes is near zero; the startup compile is the only load.

## Files

- `Dockerfile` - the public recipe: ubuntu:24.04, Intel's published driver debs (versions
  are `ARG`s), `libze1`, a venv with pinned `openvino==2026.4.1 numpy pillow`, the reader.
- `Dockerfile.host-libs` - the version-matched recipe: python:3.12-slim plus your host's
  driver libraries.
- `fetch_tools.sh` - stages `vendor/` (reader, exporter, health check) for the public build.
- `fetch_host_libs.sh` - stages `lib/` (Level Zero loader + layers, the VPU driver, the NPU
  compiler) and `vendor/` for the version-matched build.
- `npu_health.py` - device check, exits non-zero when the NPU is missing.
- `entrypoint.sh` - defaults `STRATA_VISION_NPU_MODEL_IR` to `/ir/encoder-1.xml:/ir/encoder-2.xml`.
- `lib/`, `vendor/` - build staging, regenerated by the scripts, not committed.
