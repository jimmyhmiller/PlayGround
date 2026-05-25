"""Dump Mojo's actual gaussian_noise_fill output to disk (1D)."""
from std.sys import has_accelerator
from std.testing import TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr

from fixture import save_fp32_1d
from cfm_estimator_new import gaussian_noise_fill


def test_dump() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var B = 1
    var MEL = 80
    var T = 602
    var n = B * MEL * T

    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    gaussian_noise_fill(ctx, buf, n, UInt64(0xC0FFEE), Float32(1.0))
    ctx.synchronize()

    var data = List[Float32](capacity=n)
    with buf.map_to_host() as h:
        for i in range(n):
            data.append(h[i])
    save_fp32_1d("weights/s3gen_prompt/lcg_diag/mojo_lcg_noise.bin", data)
    print("[dump] wrote mojo_lcg_noise.bin (1D,", n, "fp32)")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
