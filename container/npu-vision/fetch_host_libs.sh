#!/bin/sh
# Collect everything the image needs from this machine into the build directory:
# the Intel NPU userspace libraries (version-matched to the running kernel
# module) and the two Strata Python tools.  Run from anywhere, then build:
#
#   container/npu-vision/fetch_host_libs.sh
#   podman build -t strata-npu-vision container/npu-vision
set -e
cd "$(dirname "$0")"

LIBDIR=/usr/lib/x86_64-linux-gnu
NEEDED="libze_loader.so.1 libze_tracing_layer.so.1 libze_validation_layer.so.1
libze_intel_vpu.so.1 libnpu_driver_compiler.so"

mkdir -p lib vendor
rm -f lib/*
for so in $NEEDED; do
    [ -e "$LIBDIR/$so" ] || { echo "missing $LIBDIR/$so - install the Intel NPU driver packages first" >&2; exit 1; }
    # copy the file and its symlink siblings so ldconfig can resolve the soname
    cp -a "$LIBDIR"/"$so"* lib/
done

REPO=$(cd ../.. && pwd)
cp -a "$REPO/tools/vision/strata_vision_npu.py" vendor/
cp -a "$REPO/tools/vision/export_npu_encoder.py" vendor/
cp -a npu_health.py vendor/

echo "staged:"
du -sh lib vendor
