import Foundation
import Metal
import simd

/// High-level text renderer that simplifies Metal text rendering
public class TextRenderer {
    /// Font atlas manager
    public let fontManager: FontAtlasManager
    
    /// Metal device
    private let device: MTLDevice
    
    /// Render pipeline state
    private var renderPipeline: MTLRenderPipelineState?
    
    /// Vertex buffer for text geometry
    private var vertexBuffer: MTLBuffer?
    
    /// Maximum number of characters that can be rendered in one batch
    private let maxCharacters: Int
    
    /// Current atlas texture
    private var atlasTexture: MTLTexture?
    
    /// Last known modification count
    private var lastModificationCount: UInt64 = 0
    
    /// Current vertex buffer offset for multiple draw calls within a frame
    private var currentVertexOffset: Int = 0
    
    /// Vertex structure for text rendering
    public struct TextVertex {
        public var position: simd_float2
        public var texCoord: simd_float2
        public var color: simd_float4
        
        public init(position: simd_float2, texCoord: simd_float2, color: simd_float4 = simd_float4(1, 1, 1, 1)) {
            self.position = position
            self.texCoord = texCoord
            self.color = color
        }
    }
    
    /// Text alignment options
    public enum TextAlignment {
        case left
        case center
        case right
    }
    
    /// Initialize a new text renderer
    /// - Parameters:
    ///   - device: Metal device
    ///   - fontName: Font name
    ///   - fontSize: Font size in points
    ///   - maxCharacters: Maximum characters to render in one batch (default: 10000)
    ///   - coordinateOrigin: Coordinate system origin (default: .topLeft for Metal)
    public init(
        device: MTLDevice,
        fontName: String,
        fontSize: Float,
        maxCharacters: Int = 10000,
        coordinateOrigin: CoordinateOrigin = .topLeft
    ) throws {
        self.device = device
        self.maxCharacters = maxCharacters
        
        // Create font manager with specified coordinate system
        self.fontManager = try FontAtlasManager(
            fontName: fontName,
            fontSize: fontSize,
            coordinateOrigin: coordinateOrigin
        )
        
        // Create vertex buffer
        let bufferSize = maxCharacters * 6 * MemoryLayout<TextVertex>.stride
        self.vertexBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
        
        // Setup render pipeline
        try setupRenderPipeline()
        
        // Create initial texture
        updateAtlasTexture()
    }
    
    /// Draw text at the specified position
    /// - Parameters:
    ///   - text: Text to draw
    ///   - position: Position in screen coordinates
    ///   - color: Text color (default: white)
    ///   - renderEncoder: Metal render command encoder
    public func drawText(
        _ text: String,
        at position: CGPoint,
        color: simd_float4 = simd_float4(1, 1, 1, 1),
        using renderEncoder: MTLRenderCommandEncoder
    ) {
        // Update texture if needed
        updateAtlasTexture()
        
        // Split text by newlines and render each line
        let lines = text.components(separatedBy: .newlines)
        let lineHeight = fontManager.metrics.cellHeight
        
        var allVertices: [TextVertex] = []
        
        for (lineIndex, line) in lines.enumerated() {
            let linePosition = CGPoint(
                x: position.x,
                y: position.y + CGFloat(lineIndex) * lineHeight
            )
            
            let lineVertices = generateVertices(for: line, at: linePosition, color: color)
            allVertices.append(contentsOf: lineVertices)
        }
        
        guard !allVertices.isEmpty else { return }
        
        // Check if vertices exceed remaining buffer capacity
        let maxVertices = maxCharacters * 6
        let remainingCapacity = maxVertices - currentVertexOffset
        
        if allVertices.count > remainingCapacity {
            print("⚠️ Warning: Not enough buffer space. Vertices: \(allVertices.count), remaining: \(remainingCapacity). Resetting buffer.")
            // Reset buffer for this frame if we exceed capacity
            currentVertexOffset = 0
        }
        
        let vertexCount = min(allVertices.count, maxVertices - currentVertexOffset)
        guard vertexCount > 0 else { return }
        
        // Update vertex buffer at current offset
        guard let vertexBuffer = vertexBuffer else { return }
        let bufferPointer = vertexBuffer.contents().bindMemory(to: TextVertex.self, capacity: maxVertices)
        
        for index in 0..<vertexCount {
            bufferPointer[currentVertexOffset + index] = allVertices[index]
        }
        
        // Set render state
        renderEncoder.setRenderPipelineState(renderPipeline!)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: currentVertexOffset * MemoryLayout<TextVertex>.stride, index: 0)
        renderEncoder.setFragmentTexture(atlasTexture, index: 0)
        
        // Draw
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: vertexCount)
        
        // Advance offset for next draw call
        currentVertexOffset += vertexCount
    }
    
    /// Draw multiline text with optional alignment
    /// - Parameters:
    ///   - text: Text to draw (can contain newlines)
    ///   - rect: Rectangle to draw text within
    ///   - alignment: Text alignment
    ///   - lineSpacing: Additional spacing between lines
    ///   - color: Text color
    ///   - renderEncoder: Metal render command encoder
    public func drawText(
        _ text: String,
        in rect: CGRect,
        alignment: TextAlignment = .left,
        lineSpacing: CGFloat = 0,
        color: simd_float4 = simd_float4(1, 1, 1, 1),
        using renderEncoder: MTLRenderCommandEncoder
    ) {
        let lines = text.components(separatedBy: .newlines)
        let lineHeight = fontManager.metrics.cellHeight + lineSpacing
        var currentY = rect.minY + fontManager.metrics.ascent
        
        for line in lines {
            let lineWidth = fontManager.lineWidth(line)
            let x: CGFloat
            
            switch alignment {
            case .left:
                x = rect.minX
            case .center:
                x = rect.minX + (rect.width - lineWidth) / 2
            case .right:
                x = rect.minX + rect.width - lineWidth
            }
            
            drawText(line, at: CGPoint(x: x, y: currentY), color: color, using: renderEncoder)
            currentY += lineHeight
        }
    }
    
    /// Set projection matrix for the renderer
    /// - Parameters:
    ///   - renderEncoder: Metal render command encoder
    ///   - viewportSize: Size of the viewport
    public func setProjectionMatrix(using renderEncoder: MTLRenderCommandEncoder, viewportSize: CGSize) {
        // Reset vertex buffer offset at the start of each frame
        currentVertexOffset = 0
        
        var projection = orthographicProjection(
            left: 0,
            right: Float(viewportSize.width),
            bottom: Float(viewportSize.height),
            top: 0,
            near: -1,
            far: 1
        )
        
        renderEncoder.setVertexBytes(&projection, length: MemoryLayout<simd_float4x4>.size, index: 1)
    }
    
    // MARK: - Private Methods
    
    private func setupRenderPipeline() throws {
        // Create shader library from source
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;
        
        struct TextVertex {
            float2 position [[attribute(0)]];
            float2 texCoord [[attribute(1)]];
            float4 color [[attribute(2)]];
        };
        
        struct TextVertexOut {
            float4 position [[position]];
            float2 texCoord;
            float4 color;
        };
        
        vertex TextVertexOut text_vertex(TextVertex in [[stage_in]],
                                       constant float4x4& projection [[buffer(1)]]) {
            TextVertexOut out;
            out.position = projection * float4(in.position, 0.0, 1.0);
            out.texCoord = in.texCoord;
            out.color = in.color;
            return out;
        }
        
        fragment float4 text_fragment(TextVertexOut in [[stage_in]],
                                    texture2d<float> atlas [[texture(0)]]) {
            constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
            float alpha = atlas.sample(textureSampler, in.texCoord).r;
            return float4(in.color.rgb, in.color.a * alpha);
        }
        """
        
        let library = try device.makeLibrary(source: shaderSource, options: nil)
        let vertexFunction = library.makeFunction(name: "text_vertex")
        let fragmentFunction = library.makeFunction(name: "text_fragment")
        
        // Create vertex descriptor
        let vertexDescriptor = MTLVertexDescriptor()
        
        // Position
        vertexDescriptor.attributes[0].format = .float2
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        
        // Texture coordinates
        vertexDescriptor.attributes[1].format = .float2
        vertexDescriptor.attributes[1].offset = MemoryLayout<simd_float2>.size
        vertexDescriptor.attributes[1].bufferIndex = 0
        
        // Color
        vertexDescriptor.attributes[2].format = .float4
        vertexDescriptor.attributes[2].offset = MemoryLayout<simd_float2>.size * 2
        vertexDescriptor.attributes[2].bufferIndex = 0
        
        // Layout
        vertexDescriptor.layouts[0].stride = MemoryLayout<TextVertex>.stride
        vertexDescriptor.layouts[0].stepRate = 1
        vertexDescriptor.layouts[0].stepFunction = .perVertex
        
        // Create pipeline descriptor
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        // Configure blending for transparency
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        renderPipeline = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    private func updateAtlasTexture() {
        // Check if texture needs update
        let currentModCount = fontManager.modificationCount
        guard atlasTexture == nil || currentModCount > lastModificationCount else { return }
        
        // Create or recreate texture
        let managedTexture = fontManager.createManagedTexture(device: device)
        atlasTexture = managedTexture.metalTexture
        lastModificationCount = currentModCount
    }
    
    private func generateVertices(for text: String, at position: CGPoint, color: simd_float4) -> [TextVertex] {
        var vertices: [TextVertex] = []
        var currentX = Float(position.x)
        let baselineY = Float(position.y)
        
        let atlasSize = Float(fontManager.withAtlas { $0.size })
        
        for character in text {
            guard let glyph = fontManager.renderCharacter(character) else { continue }
            
            // // Skip empty glyphs
            // if glyph.width == 0 || glyph.height == 0 {
            //     currentX += glyph.advanceX
            //     continue
            // }
            
            // Calculate glyph position using simple approach that works
            let left = currentX + Float(glyph.offsetX)
            let top = baselineY - Float(glyph.height) - Float(glyph.offsetY)
            let right = left + Float(glyph.width)
            let bottom = top + Float(glyph.height)
            
            // Calculate texture coordinates
            let u1 = Float(glyph.atlasX) / atlasSize
            let v1 = Float(glyph.atlasY) / atlasSize
            let u2 = Float(glyph.atlasX + glyph.width) / atlasSize
            let v2 = Float(glyph.atlasY + glyph.height) / atlasSize
            
            // Create quad (two triangles)
            vertices.append(contentsOf: [
                // First triangle
                TextVertex(position: simd_float2(left, top), texCoord: simd_float2(u1, v1), color: color),
                TextVertex(position: simd_float2(right, top), texCoord: simd_float2(u2, v1), color: color),
                TextVertex(position: simd_float2(left, bottom), texCoord: simd_float2(u1, v2), color: color),
                
                // Second triangle
                TextVertex(position: simd_float2(right, top), texCoord: simd_float2(u2, v1), color: color),
                TextVertex(position: simd_float2(right, bottom), texCoord: simd_float2(u2, v2), color: color),
                TextVertex(position: simd_float2(left, bottom), texCoord: simd_float2(u1, v2), color: color)
            ])
            
            currentX += glyph.advanceX
        }
        
        return vertices
    }
    
    private func orthographicProjection(left: Float, right: Float, bottom: Float, top: Float, near: Float, far: Float) -> simd_float4x4 {
        let sx = 2 / (right - left)
        let sy = 2 / (top - bottom)
        let sz = 1 / (far - near)
        let tx = (right + left) / (left - right)
        let ty = (top + bottom) / (bottom - top)
        let tz = near / (near - far)
        
        return simd_float4x4(
            simd_float4(sx, 0, 0, 0),
            simd_float4(0, sy, 0, 0),
            simd_float4(0, 0, sz, 0),
            simd_float4(tx, ty, tz, 1)
        )
    }
}