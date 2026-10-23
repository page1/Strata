#!/bin/sh
# The reader speaks the server's stdin/stdout protocol (READY / ENC / OK), so
# `podman run -i` is exactly what the server expects from a vision exe.
# Extra arguments pass through to the reader untouched.
if [ -z "$STRATA_VISION_NPU_MODEL_IR" ]; then
    STRATA_VISION_NPU_MODEL_IR=/ir/encoder-1.xml:/ir/encoder-2.xml
    export STRATA_VISION_NPU_MODEL_IR
fi
exec python /opt/strata/strata_vision_npu.py "$@"
