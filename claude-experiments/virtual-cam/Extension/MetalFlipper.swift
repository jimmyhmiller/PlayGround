import Metal
import CoreVideo

/// Zero-copy, GPU-only 180° flip.
///
/// The source and destination CVPixelBuffers are IOSurface-backed. We wrap each
/// one's IOSurface directly as an MTLTexture (no copy, unified memory), then a
/// single compute dispatch reads each source pixel and writes it to the
/// 180°-rotated position. The CPU never touches a pixel — it only encodes the
/// command; every byte of pixel movement happens on the GPU. This is the minimal
/// work a 180° rotation can be: exactly one read + one write per pixel.
final class MetalFlipper {

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    // BGRA, matching the capture/output pixel format.
    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void flip180(texture2d<half, access::read>  src [[texture(0)]],
                        texture2d<half, access::write> dst [[texture(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
        uint w = src.get_width();
        uint h = src.get_height();
        if (gid.x >= w || gid.y >= h) { return; }
        // Read the mirrored source pixel, write straight out. One read, one write.
        half4 c = src.read(uint2(w - 1 - gid.x, h - 1 - gid.y));
        dst.write(c, gid);
    }
    """

    init?() {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let q = dev.makeCommandQueue() else { return nil }
        device = dev
        queue = q
        do {
            let lib = try dev.makeLibrary(source: Self.shaderSource, options: nil)
            guard let fn = lib.makeFunction(name: "flip180") else { return nil }
            pipeline = try dev.makeComputePipelineState(function: fn)
        } catch {
            return nil
        }
    }

    private func makeTexture(_ pb: CVPixelBuffer, usage: MTLTextureUsage) -> MTLTexture? {
        guard let surface = CVPixelBufferGetIOSurface(pb)?.takeUnretainedValue() else { return nil }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: CVPixelBufferGetWidth(pb),
            height: CVPixelBufferGetHeight(pb),
            mipmapped: false)
        desc.usage = usage
        desc.storageMode = .shared   // Apple Silicon unified memory, IOSurface-backed.
        return device.makeTexture(descriptor: desc, iosurface: surface, plane: 0)
    }

    /// Flip `src` into `dst` entirely on the GPU. `completion(true)` fires once
    /// the GPU has finished writing `dst` (so it's safe to publish). No CPU pixel
    /// work happens anywhere in here.
    func flip(src: CVPixelBuffer, dst: CVPixelBuffer, completion: @escaping (Bool) -> Void) {
        guard let srcTex = makeTexture(src, usage: .shaderRead),
              let dstTex = makeTexture(dst, usage: .shaderWrite),
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            completion(false); return
        }

        enc.setComputePipelineState(pipeline)
        enc.setTexture(srcTex, index: 0)
        enc.setTexture(dstTex, index: 1)

        let tew = pipeline.threadExecutionWidth
        let teh = max(1, pipeline.maxTotalThreadsPerThreadgroup / tew)
        let tgSize = MTLSize(width: tew, height: teh, depth: 1)
        let grid = MTLSize(width: srcTex.width, height: srcTex.height, depth: 1)
        // Apple GPUs support non-uniform threadgroups; the kernel also bounds-checks.
        enc.dispatchThreads(grid, threadsPerThreadgroup: tgSize)
        enc.endEncoding()

        cb.addCompletedHandler { _ in
            withExtendedLifetime((srcTex, dstTex)) {}
            completion(true)
        }
        cb.commit()
    }
}
