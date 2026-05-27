"""Dump our gaussian_noise_fill output for inspection."""
from std.gpu.host import DeviceContext
from cfm_estimator_new import gaussian_noise_fill
from std.python import Python

def main() raises:
    var ctx = DeviceContext()
    var N = 80 * 1132
    var buf = ctx.enqueue_create_buffer[DType.float32](N)
    gaussian_noise_fill(ctx, buf, N, UInt64(0xC0FFEE), Float32(1.0))
    ctx.synchronize()

    var host = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()

    # Export to numpy via Python.
    var py_np = Python.import_module("numpy")
    var arr = py_np.empty(N, dtype="float32")
    var arr_addr = Int(py=arr.ctypes.data)
    var dst = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=arr_addr)
    var src = host.unsafe_ptr()
    for i in range(N):
        dst[i] = src[i]
    py_np.save("/tmp/cfm_diag/mojo_noise_sample.npy", arr.reshape(1, 80, 1132))
    print("saved /tmp/cfm_diag/mojo_noise_sample.npy")
    # Print quick stats.
    var s_min: Float32 = 1e30
    var s_max: Float32 = -1e30
    var s_sum: Float32 = 0.0
    var s_sq: Float32 = 0.0
    for i in range(N):
        var v = src[i]
        if v < s_min: s_min = v
        if v > s_max: s_max = v
        s_sum += v
        s_sq += v * v
    var mean = s_sum / Float32(N)
    var var_ = s_sq / Float32(N) - mean * mean
    print("noise N=", N, " min=", s_min, " max=", s_max, " mean=", mean, " var=", var_)
