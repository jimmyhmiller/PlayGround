import Cocoa
import Metal
import MetalKit

class FlameGraphView: MTKView {
    private var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var renderPipelineState: MTLRenderPipelineState!
    private var vertexBuffer: MTLBuffer!
    private var indexBuffer: MTLBuffer!
    
    private var stackTrace: StackTrace?
    private var viewMatrix = matrix_identity_float4x4
    private var projectionMatrix = matrix_identity_float4x4
    
    private var panGesture: NSPanGestureRecognizer!
    private var magnificationGesture: NSMagnificationGestureRecognizer!
    
    private var zoomScale: Float = 1.0
    private var panOffset = SIMD2<Float>(0, 0)
    
    struct Vertex {
        let position: SIMD3<Float>
        let color: SIMD4<Float>
    }
    
    struct Uniforms {
        var modelViewProjectionMatrix: matrix_float4x4
    }
    
    private var uniformBuffer: MTLBuffer!
    
    override init(frame frameRect: NSRect, device: MTLDevice?) {
        super.init(frame: frameRect, device: device)
        setupMetal()
        setupGestures()
    }
    
    required init(coder: NSCoder) {
        super.init(coder: coder)
        setupMetal()
        setupGestures()
    }
    
    private func setupMetal() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        
        self.device = device
        self.device = device
        commandQueue = device.makeCommandQueue()
        
        colorPixelFormat = .bgra8Unorm
        
        setupShaders()
        updateProjectionMatrix()
        
        uniformBuffer = device.makeBuffer(length: MemoryLayout<Uniforms>.size, options: [])
    }
    
    private func setupShaders() {
        let vertexFunction = """
        #include <metal_stdlib>
        using namespace metal;
        
        struct Vertex {
            float3 position [[attribute(0)]];
            float4 color [[attribute(1)]];
        };
        
        struct VertexOut {
            float4 position [[position]];
            float4 color;
        };
        
        struct Uniforms {
            float4x4 modelViewProjectionMatrix;
        };
        
        vertex VertexOut vertex_main(Vertex in [[stage_in]],
                                   constant Uniforms& uniforms [[buffer(1)]]) {
            VertexOut out;
            out.position = uniforms.modelViewProjectionMatrix * float4(in.position, 1.0);
            out.color = in.color;
            return out;
        }
        """
        
        let fragmentFunction = """
        #include <metal_stdlib>
        using namespace metal;
        
        struct VertexOut {
            float4 position [[position]];
            float4 color;
        };
        
        fragment float4 fragment_main(VertexOut in [[stage_in]]) {
            return in.color;
        }
        """
        
        let library = try! device.makeLibrary(source: vertexFunction + fragmentFunction, options: nil)
        let vertexFunc = library.makeFunction(name: "vertex_main")!
        let fragmentFunc = library.makeFunction(name: "fragment_main")!
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunc
        pipelineDescriptor.fragmentFunction = fragmentFunc
        pipelineDescriptor.colorAttachments[0].pixelFormat = colorPixelFormat
        
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float3
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        
        vertexDescriptor.attributes[1].format = .float4
        vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.size
        vertexDescriptor.attributes[1].bufferIndex = 0
        
        vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex>.size
        vertexDescriptor.layouts[0].stepFunction = .perVertex
        
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        renderPipelineState = try! device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    private func setupGestures() {
        panGesture = NSPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        magnificationGesture = NSMagnificationGestureRecognizer(target: self, action: #selector(handleMagnification(_:)))
        
        addGestureRecognizer(panGesture)
        addGestureRecognizer(magnificationGesture)
    }
    
    @objc private func handlePan(_ gesture: NSPanGestureRecognizer) {
        let translation = gesture.translation(in: self)
        panOffset.x += Float(translation.x) * 0.01 / zoomScale
        panOffset.y -= Float(translation.y) * 0.01 / zoomScale
        gesture.setTranslation(.zero, in: self)
        updateViewMatrix()
    }
    
    @objc private func handleMagnification(_ gesture: NSMagnificationGestureRecognizer) {
        if gesture.state == .changed {
            zoomScale *= Float(1.0 + gesture.magnification)
            zoomScale = max(0.1, min(zoomScale, 20.0))
            gesture.magnification = 0
            updateViewMatrix()
        }
    }
    
    private func updateProjectionMatrix() {
        let aspect = Float(bounds.width / bounds.height)
        projectionMatrix = matrix_perspective_left_hand(fovyRadians: Float.pi / 4,
                                                       aspectRatio: aspect,
                                                       nearZ: 0.1,
                                                       farZ: 1000.0)
    }
    
    private func updateViewMatrix() {
        var transform = matrix_identity_float4x4
        transform = matrix_multiply(transform, matrix_scale(zoomScale, zoomScale, 1.0))
        transform = matrix_multiply(transform, matrix_translate(panOffset.x, panOffset.y, 0))
        viewMatrix = transform
    }
    
    func loadStackTrace(_ stackTrace: StackTrace) {
        self.stackTrace = stackTrace
        generateFlameGraphGeometry()
    }
    
    private func generateFlameGraphGeometry() {
        guard let stackTrace = stackTrace else { return }
        
        var vertices: [Vertex] = []
        var indices: [UInt16] = []
        
        let timelineWidth: Float = 2.0
        let frameHeight: Float = 0.05
        let captureWidth = timelineWidth / Float(stackTrace.totalCaptures)
        
        for (captureIndex, capture) in stackTrace.captures.enumerated() {
            let x = Float(captureIndex) * captureWidth - 1.0
            
            for frame in capture.frames {
                let y = Float(frame.depth) * frameHeight - 1.0
                let color = colorForFunction(frame.function)
                
                let topLeft = Vertex(position: SIMD3<Float>(x, y + frameHeight, 0), color: color)
                let topRight = Vertex(position: SIMD3<Float>(x + captureWidth, y + frameHeight, 0), color: color)
                let bottomLeft = Vertex(position: SIMD3<Float>(x, y, 0), color: color)
                let bottomRight = Vertex(position: SIMD3<Float>(x + captureWidth, y, 0), color: color)
                
                let baseIndex = UInt16(vertices.count)
                vertices.append(contentsOf: [topLeft, topRight, bottomLeft, bottomRight])
                
                indices.append(contentsOf: [
                    baseIndex, baseIndex + 1, baseIndex + 2,
                    baseIndex + 1, baseIndex + 3, baseIndex + 2
                ])
            }
        }
        
        vertexBuffer = device.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<Vertex>.size, options: [])
        indexBuffer = device.makeBuffer(bytes: indices, length: indices.count * MemoryLayout<UInt16>.size, options: [])
    }
    
    private func colorForFunction(_ functionName: String) -> SIMD4<Float> {
        let hash = abs(functionName.hashValue)
        let r = Float((hash >> 16) & 0xFF) / 255.0
        let g = Float((hash >> 8) & 0xFF) / 255.0
        let b = Float(hash & 0xFF) / 255.0
        return SIMD4<Float>(r, g, b, 0.8)
    }
    
    override func draw(_ rect: NSRect) {
        guard let drawable = currentDrawable,
              let renderPassDescriptor = currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)
        
        var uniforms = Uniforms(modelViewProjectionMatrix: matrix_multiply(projectionMatrix, viewMatrix))
        let uniformBufferPointer = uniformBuffer.contents().bindMemory(to: Uniforms.self, capacity: 1)
        uniformBufferPointer.pointee = uniforms
        
        renderEncoder.setRenderPipelineState(renderPipelineState)
        
        if let vertexBuffer = vertexBuffer, let indexBuffer = indexBuffer {
            renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            renderEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
            
            let indexCount = indexBuffer.length / MemoryLayout<UInt16>.size
            renderEncoder.drawIndexedPrimitives(type: .triangle,
                                              indexCount: indexCount,
                                              indexType: .uint16,
                                              indexBuffer: indexBuffer,
                                              indexBufferOffset: 0)
        }
        
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    override func viewDidEndLiveResize() {
        super.viewDidEndLiveResize()
        updateProjectionMatrix()
    }
}

func matrix_perspective_left_hand(fovyRadians fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (farZ - nearZ)
    
    return matrix_float4x4(columns: (
        SIMD4<Float>(xs, 0, 0, 0),
        SIMD4<Float>(0, ys, 0, 0),
        SIMD4<Float>(0, 0, zs, 1),
        SIMD4<Float>(0, 0, -nearZ * zs, 0)
    ))
}

func matrix_scale(_ x: Float, _ y: Float, _ z: Float) -> matrix_float4x4 {
    return matrix_float4x4(columns: (
        SIMD4<Float>(x, 0, 0, 0),
        SIMD4<Float>(0, y, 0, 0),
        SIMD4<Float>(0, 0, z, 0),
        SIMD4<Float>(0, 0, 0, 1)
    ))
}

func matrix_translate(_ x: Float, _ y: Float, _ z: Float) -> matrix_float4x4 {
    return matrix_float4x4(columns: (
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(0, 0, 1, 0),
        SIMD4<Float>(x, y, z, 1)
    ))
}