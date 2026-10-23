"""Container health check: enumerate devices, compile and run a tiny model on
the NPU.  Prints one OK line or a clear failure.  Exits 0 only on success."""

import sys

import numpy as np
import openvino as ov

core = ov.Core()
devices = core.available_devices
print("openvino", ov.get_version(), "devices:", devices)
if "NPU" not in devices:
    print(
        "FAIL: no NPU device. Pass --device /dev/accel/accel0 and check the"
        " Level Zero loader and the driver libraries inside the container."
    )
    sys.exit(1)
a = ov.opset13.parameter(shape=[1, 64], dtype=np.float32, name="a")
w = ov.opset13.constant(
    np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
)
m = ov.Model([ov.opset13.matmul(a, w, False, False)], [a], "tiny")
comp = core.compile_model(m, "NPU")
got = np.asarray(comp(np.ones((1, 64), np.float32))[comp.output(0)])
print(
    f"NPU inference inside the container OK: shape {got.shape} mean {float(got.mean()):.4f}"
)
