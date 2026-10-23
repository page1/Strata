#!/bin/sh
# Stage what the public build (Dockerfile) needs into vendor/: the two Strata tools
# and the health check.  Run from anywhere, then build:
#
#   container/npu-vision/fetch_tools.sh
#   podman build -t strata-npu-vision container/npu-vision
#
# The version-matched build stages more (your host's driver libraries):
# see fetch_host_libs.sh and Dockerfile.host-libs.
set -e
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
mkdir -p vendor
cp -a "$REPO/tools/vision/strata_vision_npu.py" vendor/
cp -a "$REPO/tools/vision/export_npu_encoder.py" vendor/
cp -a npu_health.py vendor/
echo "staged:"
du -sh vendor
