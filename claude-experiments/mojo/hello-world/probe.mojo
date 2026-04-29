from std.gpu.host import DeviceContext

def main() raises:
    var ctx = DeviceContext()
    print("device:", ctx.name())
    print("api:", ctx.api())
    var buf = ctx.enqueue_create_buffer[DType.float32](16)
    print("allocated 64 bytes ok")
    buf.enqueue_fill(1.0)
    ctx.synchronize()
    print("fill+sync ok")
