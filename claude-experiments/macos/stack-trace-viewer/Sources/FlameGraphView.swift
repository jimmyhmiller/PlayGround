import Cocoa
import Metal
import MetalKit
import CoreText

class Camera {
    var position = SIMD2<Float>(0, 0)
    var zoom: Float = 0.4
    var timeScale: Float = 1.0  // Controls the width of rectangles
    
    private let groundLevel: Float = -1.0  // Bottom of screen
    private var floorPosition: Float = 0
    private var leftBoundary: Float = 0
    private var rightBoundary: Float = 0
    
    func setupBounds(dataBounds: (minX: Float, maxX: Float, minY: Float, maxY: Float), screenAspect: Float) {
        // Calculate the floor constraint: lowest Y position camera can reach
        // When camera is at floor, data bottom (dataBounds.minY) should be at screen bottom (groundLevel)
        // Camera position = screen position - (data position * zoom)
        floorPosition = groundLevel - (dataBounds.minY * zoom)
        
        // Calculate horizontal boundaries to keep the figure always visible
        // Use timeScale for horizontal scaling, zoom for vertical
        // Left boundary: prevent data from going off the right side of screen
        let dataLeft = dataBounds.minX * timeScale
        let screenRight = screenAspect
        leftBoundary = screenRight - dataLeft - 0.2  // Small margin to keep some data visible
        
        // Right boundary: prevent data from going off the left side of screen
        let dataRight = dataBounds.maxX * timeScale
        let screenLeft = -screenAspect
        rightBoundary = screenLeft - dataRight + 0.2  // Small margin to keep some data visible
    }
    
    func moveBy(deltaX: Float, deltaY: Float) {
        // Move camera by delta amounts
        position.x += deltaX
        position.y += deltaY
        
        // Apply constraints
        applyConstraints()
    }
    
    func setPosition(x: Float, y: Float) {
        position.x = x
        position.y = y
        applyConstraints()
    }
    
    private func applyConstraints() {
        // Floor constraint: can't scroll DOWN past the floor (prevent camera from going below floor)
        position.y = min(floorPosition, position.y)
        
        // No horizontal constraints - allow free horizontal scrolling
        // This is needed because when zoomed in, the data can be much wider than the screen
    }
    
    func getViewMatrix() -> matrix_float4x4 {
        var transform = matrix_identity_float4x4
        // Don't apply timeScale here - it's already baked into the geometry
        transform = matrix_multiply(transform, matrix_scale(1.0, zoom, 1.0))
        transform = matrix_multiply(transform, matrix_translate(position.x, position.y, 0))
        return transform
    }
}

class FlameGraphView: MTKView {
    private var metalDevice: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var renderPipelineState: MTLRenderPipelineState!
    private var textRenderPipelineState: MTLRenderPipelineState!
    private var vertexBuffer: MTLBuffer?
    private var indexBuffer: MTLBuffer?
    private var textVertexBuffer: MTLBuffer?
    private var textIndexBuffer: MTLBuffer?
    
    private var stackTrace: StackTrace?
    private var viewMatrix = matrix_identity_float4x4
    private var projectionMatrix = matrix_identity_float4x4
    
    private var magnificationGesture: NSMagnificationGestureRecognizer!
    
    private var camera: Camera!
    private var dataBounds = (minX: Float(-1.0), maxX: Float(1.0), minY: Float(-1.0), maxY: Float(1.0))
    private var frameRects: [(rect: NSRect, functionName: String)] = []
    private var debugLabel: NSTextField!
    
    private var fontAtlasTexture: MTLTexture!
    private var fontAtlasGlyphs: [Character: (x: Float, y: Float, width: Float, height: Float)] = [:]
    private var samplerState: MTLSamplerState!
    
    private func createTextQuad(character: Character, position: SIMD3<Float>, size: SIMD2<Float>, color: SIMD4<Float>) -> [TextVertex]? {
        guard let glyph = fontAtlasGlyphs[character] else { return nil }
        
        // UV coordinates - no flipping needed since the test grid works correctly
        let uvLeft = glyph.x
        let uvRight = glyph.x + glyph.width
        let uvTop = glyph.y
        let uvBottom = glyph.y + glyph.height
        
        return [
            TextVertex(
                position: SIMD3<Float>(position.x, position.y + size.y, position.z),
                texCoord: SIMD2<Float>(uvLeft, uvTop),
                color: color
            ),
            TextVertex(
                position: SIMD3<Float>(position.x + size.x, position.y + size.y, position.z),
                texCoord: SIMD2<Float>(uvRight, uvTop),
                color: color
            ),
            TextVertex(
                position: SIMD3<Float>(position.x, position.y, position.z),
                texCoord: SIMD2<Float>(uvLeft, uvBottom),
                color: color
            ),
            TextVertex(
                position: SIMD3<Float>(position.x + size.x, position.y, position.z),
                texCoord: SIMD2<Float>(uvRight, uvBottom),
                color: color
            )
        ]
    }
    
    struct Vertex {
        let position: SIMD3<Float>
        let color: SIMD4<Float>
    }
    
    struct TextVertex {
        let position: SIMD3<Float>
        let texCoord: SIMD2<Float>
        let color: SIMD4<Float>
    }
    
    struct Uniforms {
        var modelViewProjectionMatrix: matrix_float4x4
    }
    
    private var uniformBuffer: MTLBuffer!
    private var textUniformBuffer: MTLBuffer!
    
    override init(frame frameRect: NSRect, device: MTLDevice? = nil) {
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
        
        self.metalDevice = device
        self.device = device
        commandQueue = device.makeCommandQueue()
        
        colorPixelFormat = .bgra8Unorm
        isPaused = false
        enableSetNeedsDisplay = false
        
        setupShaders()
        updateProjectionMatrix()
        
        uniformBuffer = device.makeBuffer(length: MemoryLayout<Uniforms>.size, options: [])
        textUniformBuffer = device.makeBuffer(length: MemoryLayout<Uniforms>.size, options: [])
        
        // Initialize camera
        camera = Camera()
        
        // Create font atlas
        createFontAtlas()
        
        // Create initial test geometry so something shows up
        createTestGeometry()
        updateViewMatrix()
        setupDebugView()
    }
    
    override var acceptsFirstResponder: Bool {
        return true
    }
    
    private func createFontAtlas() {
        let fontSize: CGFloat = 20
        let font = NSFont(name: "Ubuntu Mono", size: fontSize) ?? NSFont(name: "Menlo", size: fontSize) ?? NSFont.monospacedSystemFont(ofSize: fontSize, weight: .medium)
        
        // Characters to include in atlas - start with A in top-left
        let characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:()[]<>/"
        print("Creating font atlas with \(characters.count) characters")
        
        // Fixed atlas size
        let atlasWidth = 512
        let atlasHeight = 384
        let charWidth = 32  // Standard monospace width
        let charHeight = 32  // Square cells for better proportions
        let charsPerRow = 16
        
        // Create CGContext for proper text rendering
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: nil,
            width: atlasWidth,
            height: atlasHeight,
            bitsPerComponent: 8,
            bytesPerRow: atlasWidth * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else { 
            print("Failed to create CGContext")
            return 
        }
        
        // Clear to black background 
        context.setFillColor(CGColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0))
        context.fill(CGRect(x: 0, y: 0, width: atlasWidth, height: atlasHeight))
        
        // Set text rendering properties  
        context.setFillColor(CGColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0))
        context.setTextDrawingMode(.fill)
        
        // Draw each character
        for (index, char) in characters.enumerated() {
            let col = index % charsPerRow
            let row = index / charsPerRow
            
            let startX = col * charWidth
            // Flip Y coordinate so A appears at visual top-left
            let startY = (atlasHeight - (row + 1) * charHeight)
            
            // Store glyph info with corrected UV coordinates
            // The texture coordinates should match exactly how we drew them
            fontAtlasGlyphs[char] = (
                x: Float(startX) / Float(atlasWidth),
                y: Float(row * charHeight) / Float(atlasHeight),  // Use row-based Y for UV
                width: Float(charWidth) / Float(atlasWidth),
                height: Float(charHeight) / Float(atlasHeight)
            )
            
            // Create attributed string
            let string = String(char)
            let attributes: [NSAttributedString.Key: Any] = [
                .font: font,
                .foregroundColor: NSColor.white
            ]
            let attributedString = NSAttributedString(string: string, attributes: attributes)
            
            // Calculate draw position (Core Graphics origin is bottom-left)
            let drawX = CGFloat(startX + 4)
            let drawY = CGFloat(atlasHeight - startY - charHeight + 8)
            let drawRect = CGRect(x: drawX, y: drawY, width: CGFloat(charWidth - 8), height: CGFloat(charHeight - 8))
            
            
            // Draw the character using NSGraphicsContext
            NSGraphicsContext.saveGraphicsState()
            
            // Create graphics context from CGContext
            let nsContext = NSGraphicsContext(cgContext: context, flipped: false)
            NSGraphicsContext.current = nsContext
            
            // Calculate text position in the atlas - center in cell
            let textRect = CGRect(
                x: CGFloat(startX + 2), 
                y: CGFloat(startY + 6), 
                width: CGFloat(charWidth - 4), 
                height: CGFloat(charHeight - 12)
            )
            
            attributedString.draw(in: textRect)
            
            NSGraphicsContext.restoreGraphicsState()
        }
        
        // Get pixel data from context
        guard let pixelData = context.data else { 
            print("Failed to get pixel data from context")
            return 
        }
        
        // Save atlas as PNG file for debugging
        if let cgImage = context.makeImage() {
            let bitmapRep = NSBitmapImageRep(cgImage: cgImage)
            if let pngData = bitmapRep.representation(using: .png, properties: [:]) {
                let fileURL = URL(fileURLWithPath: "/tmp/font_atlas_debug.png")
                try? pngData.write(to: fileURL)
                print("Font atlas saved to: \(fileURL.path)")
            }
        }
        
        // Create Metal texture
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: atlasWidth,
            height: atlasHeight,
            mipmapped: false
        )
        
        fontAtlasTexture = metalDevice.makeTexture(descriptor: textureDescriptor)
        fontAtlasTexture?.replace(
            region: MTLRegionMake2D(0, 0, atlasWidth, atlasHeight),
            mipmapLevel: 0,
            withBytes: pixelData,
            bytesPerRow: atlasWidth * 4
        )
        
        print("Font atlas created: \(atlasWidth)x\(atlasHeight), \(fontAtlasGlyphs.count) glyphs")
        
        // Create sampler state
        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerState = metalDevice.makeSamplerState(descriptor: samplerDescriptor)
    }
    
    private func setupShaders() {
        let shaderSource = """
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
        
        fragment float4 fragment_main(VertexOut in [[stage_in]]) {
            return in.color;
        }
        """
        
        let library = try! metalDevice.makeLibrary(source: shaderSource, options: nil)
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
        
        renderPipelineState = try! metalDevice.makeRenderPipelineState(descriptor: pipelineDescriptor)
        
        // Create text rendering pipeline
        let textShaderSource = """
        #include <metal_stdlib>
        using namespace metal;
        
        struct TextVertex {
            float3 position [[attribute(0)]];
            float2 texCoord [[attribute(1)]];
            float4 color [[attribute(2)]];
        };
        
        struct TextVertexOut {
            float4 position [[position]];
            float2 texCoord;
            float4 color;
        };
        
        struct Uniforms {
            float4x4 modelViewProjectionMatrix;
        };
        
        vertex TextVertexOut text_vertex_main(TextVertex in [[stage_in]],
                                             constant Uniforms& uniforms [[buffer(1)]]) {
            TextVertexOut out;
            out.position = uniforms.modelViewProjectionMatrix * float4(in.position, 1.0);
            out.texCoord = in.texCoord;
            out.color = in.color;
            return out;
        }
        
        fragment float4 text_fragment_main(TextVertexOut in [[stage_in]],
                                         texture2d<float> fontTexture [[texture(0)]],
                                         sampler fontSampler [[sampler(0)]]) {
            // Sample the font texture
            float4 texColor = fontTexture.sample(fontSampler, in.texCoord);
            
            // Since we drew white text on dark background, use the brightness as alpha
            float brightness = (texColor.r + texColor.g + texColor.b) / 3.0;
            
            // Return white text with brightness-based alpha
            return float4(1.0, 1.0, 1.0, brightness);
        }
        """
        
        let textLibrary = try! metalDevice.makeLibrary(source: textShaderSource, options: nil)
        let textVertexFunc = textLibrary.makeFunction(name: "text_vertex_main")!
        let textFragmentFunc = textLibrary.makeFunction(name: "text_fragment_main")!
        
        let textPipelineDescriptor = MTLRenderPipelineDescriptor()
        textPipelineDescriptor.vertexFunction = textVertexFunc
        textPipelineDescriptor.fragmentFunction = textFragmentFunc
        textPipelineDescriptor.colorAttachments[0].pixelFormat = colorPixelFormat
        textPipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        textPipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        textPipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        textPipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        textPipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        textPipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        textPipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        let textVertexDescriptor = MTLVertexDescriptor()
        textVertexDescriptor.attributes[0].format = .float3
        textVertexDescriptor.attributes[0].offset = 0
        textVertexDescriptor.attributes[0].bufferIndex = 0
        
        textVertexDescriptor.attributes[1].format = .float2
        textVertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.size
        textVertexDescriptor.attributes[1].bufferIndex = 0
        
        textVertexDescriptor.attributes[2].format = .float4
        textVertexDescriptor.attributes[2].offset = MemoryLayout<SIMD3<Float>>.size + MemoryLayout<SIMD2<Float>>.size
        textVertexDescriptor.attributes[2].bufferIndex = 0
        
        textVertexDescriptor.layouts[0].stride = MemoryLayout<TextVertex>.size
        textVertexDescriptor.layouts[0].stepFunction = .perVertex
        
        textPipelineDescriptor.vertexDescriptor = textVertexDescriptor
        
        textRenderPipelineState = try! metalDevice.makeRenderPipelineState(descriptor: textPipelineDescriptor)
    }
    
    private func setupGestures() {
        // Re-enable zoom
        magnificationGesture = NSMagnificationGestureRecognizer(target: self, action: #selector(handleMagnification(_:)))
        addGestureRecognizer(magnificationGesture)
    }
    
    override func scrollWheel(with event: NSEvent) {
        let deltaX = Float(event.scrollingDeltaX)
        let deltaY = Float(event.scrollingDeltaY)
        
        // Use base scale factor for movement
        let baseScaleFactor: Float = 0.01
        
        // Move camera based on scroll input
        let cameraDeltaX = deltaX * baseScaleFactor
        let cameraDeltaY = -deltaY * baseScaleFactor  // Invert Y for natural scrolling
        
        camera.moveBy(deltaX: cameraDeltaX, deltaY: cameraDeltaY)
        updateViewMatrix()
        updateDebugInfo()
    }
    
    // This method is no longer needed - camera handles its own constraints
    private func constrainPanOffset() {
        // Camera handles its own constraints now
    }
    
    private func setupInitialView() {
        let aspect = Float(bounds.width / bounds.height)
        
        // Setup camera bounds
        camera.setupBounds(dataBounds: dataBounds, screenAspect: aspect)
        
        // Calculate initial position to show data bottom-left at screen bottom-left
        // Working backwards from the user's good values:
        // Data bottom-left (-1.0, -1.0) should appear at screen bottom-left (-aspect, -1.0)
        let dataBottomLeft = SIMD2<Float>(dataBounds.minX, dataBounds.minY)  // (-1.0, -1.0)
        let screenBottomLeft = SIMD2<Float>(-aspect, -1.0)
        
        // Camera position = screen position - (data position * zoom)
        let initialX = screenBottomLeft.x - (dataBottomLeft.x * camera.zoom)
        let initialY = screenBottomLeft.y - (dataBottomLeft.y * camera.zoom)
        
        camera.setPosition(x: initialX, y: initialY)
        
        updateViewMatrix()
        updateDebugInfo()
        
        // Test calculation
        testPositioningMath()
    }
    
    private func testPositioningMath() {
        let aspect = Float(bounds.width / bounds.height)  // 1200/800 = 1.5
        print("Testing positioning math:")
        print("Aspect ratio: \(aspect)")
        print("Your working values: Pan(-2.500, -1.325), Zoom: 0.400")
        print("My calculated values: Pan(\(camera.position.x), \(camera.position.y)), Zoom: \(camera.zoom)")
        
        // Let me work backwards from your good values
        let workingPanX: Float = -2.500
        let workingPanY: Float = -1.325
        let workingZoom: Float = 0.400
        
        // What position does the bottom-left data point end up at with your values?
        let dataBottomLeft = SIMD2<Float>(dataBounds.minX, dataBounds.minY)  // (-1.0, -1.0)
        let screenPos = dataBottomLeft * workingZoom + SIMD2<Float>(workingPanX, workingPanY)
        print("With your values, data bottom-left (-1,-1) appears at screen position: \(screenPos)")
        print("Screen left edge is at: \(-aspect), screen bottom is at: -1.0")
    }
    
    private func setupDebugView() {
        debugLabel = NSTextField(frame: NSRect(x: 10, y: 10, width: 300, height: 80))
        debugLabel.isEditable = false
        debugLabel.isSelectable = false
        debugLabel.backgroundColor = NSColor.black.withAlphaComponent(0.7)
        debugLabel.textColor = NSColor.white
        debugLabel.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        debugLabel.isBordered = false
        debugLabel.maximumNumberOfLines = 0
        debugLabel.cell?.usesSingleLineMode = false
        debugLabel.cell?.wraps = true
        
        // Add click gesture
        let clickGesture = NSClickGestureRecognizer(target: self, action: #selector(debugLabelClicked))
        debugLabel.addGestureRecognizer(clickGesture)
        
        addSubview(debugLabel)
        updateDebugInfo()
    }
    
    @objc private func debugLabelClicked() {
        let debugInfo = getDebugInfo()
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(debugInfo, forType: .string)
    }
    
    @objc private func handleMagnification(_ gesture: NSMagnificationGestureRecognizer) {
        if gesture.state == .changed || gesture.state == .ended {
            let oldTimeScale = camera.timeScale
            // Adjust time scale (rectangle width) based on magnification
            camera.timeScale *= Float(1.0 + gesture.magnification)
            camera.timeScale = max(0.1, min(100.0, camera.timeScale))  // Allow much larger scale
            gesture.magnification = 0  // Reset magnification
            
            print("TimeScale changed from \(oldTimeScale) to \(camera.timeScale)")
            
            // Regenerate geometry with new time scale
            if stackTrace != nil {
                generateFlameGraphGeometry()
            }
            
            // Update camera bounds after geometry regeneration
            let aspect = Float(bounds.width / bounds.height)
            camera.setupBounds(dataBounds: dataBounds, screenAspect: aspect)
            
            print("After regeneration - dataBounds: \(dataBounds)")
            print("Camera position: \(camera.position)")
            print("View matrix: \(camera.getViewMatrix())")
            
            updateViewMatrix()
            updateDebugInfo()
        }
    }
    
    private func getDebugInfo() -> String {
        return """
        Camera: (\(String(format: "%.3f", camera.position.x)), \(String(format: "%.3f", camera.position.y)))
        Zoom: \(String(format: "%.3f", camera.zoom))
        TimeScale: \(String(format: "%.3f", camera.timeScale))
        DataBounds: minX=\(String(format: "%.1f", dataBounds.minX)), maxX=\(String(format: "%.1f", dataBounds.maxX)), minY=\(String(format: "%.1f", dataBounds.minY)), maxY=\(String(format: "%.1f", dataBounds.maxY))
        ViewBounds: \(bounds)
        """
    }
    
    private func updateDebugInfo() {
        guard debugLabel != nil else { return }
        debugLabel.stringValue = getDebugInfo()
    }
    
    private func updateProjectionMatrix() {
        let aspect = Float(bounds.width / bounds.height)
        // Use orthographic projection for 2D flamegraph
        projectionMatrix = matrix_orthographic(left: -aspect, right: aspect, bottom: -1.0, top: 1.0, nearZ: -1.0, farZ: 1.0)
    }
    
    private func updateViewMatrix() {
        viewMatrix = camera.getViewMatrix()
        setNeedsDisplay(bounds)
    }
    
    func loadStackTrace(_ stackTrace: StackTrace) {
        self.stackTrace = stackTrace
        generateFlameGraphGeometry()
        setNeedsDisplay(bounds)
    }
    
    private func generateFlameGraphGeometry() {
        guard let stackTrace = stackTrace else { 
            print("No stack trace loaded, creating test geometry")
            createTestGeometry()
            return 
        }
        
        print("Generating geometry for \(stackTrace.totalCaptures) captures")
        
        var vertices: [Vertex] = []
        var indices: [UInt16] = []
        var textVertices: [TextVertex] = []
        var textIndices: [UInt32] = []
        frameRects = []
        
        let timelineWidth: Float = 2.0 * camera.timeScale  // Apply time scale to width
        let frameHeight: Float = 0.2  // Make frames much taller
        let captureWidth = timelineWidth / Float(stackTrace.totalCaptures)
        
        print("Capture width: \(captureWidth), Frame height: \(frameHeight)")
        
        var minY: Float = 0
        var maxY: Float = 0
        
        for (captureIndex, capture) in stackTrace.captures.enumerated() {
            let baseX = Float(captureIndex) * captureWidth - 1.0
            
            for frame in capture.frames {
                // Invert depth: depth 0 at top, growing downward
                let invertedDepth = capture.stackDepth - frame.depth - 1
                let y = Float(invertedDepth) * frameHeight - 1.0
                let color = colorForFunction(frame.function)
                
                // Make each level slightly narrower than the one below
                let depthFactor = Float(invertedDepth) * 0.02  // 2% narrower per level
                let inset = captureWidth * depthFactor * 0.5
                let x = baseX + inset
                let width = captureWidth - (inset * 2)
                
                minY = min(minY, y)
                maxY = max(maxY, y + frameHeight)
                
                let topLeft = Vertex(position: SIMD3<Float>(x, y + frameHeight, 0), color: color)
                let topRight = Vertex(position: SIMD3<Float>(x + width, y + frameHeight, 0), color: color)
                let bottomLeft = Vertex(position: SIMD3<Float>(x, y, 0), color: color)
                let bottomRight = Vertex(position: SIMD3<Float>(x + width, y, 0), color: color)
                
                let baseIndex = UInt16(vertices.count)
                vertices.append(contentsOf: [topLeft, topRight, bottomLeft, bottomRight])
                
                indices.append(contentsOf: [
                    baseIndex, baseIndex + 1, baseIndex + 2,
                    baseIndex + 1, baseIndex + 3, baseIndex + 2
                ])
                
                // Store frame rect for text rendering
                let rect = NSRect(x: CGFloat(x), y: CGFloat(y), width: CGFloat(width), height: CGFloat(frameHeight))
                frameRects.append((rect: rect, functionName: frame.function))
                
                // Add text for wider rectangles
                if width > captureWidth * 0.5 {  // Lower threshold for testing
                    let simplifiedName = frame.function  // Use full function name
                    let charHeight = frameHeight * 0.6  // Larger text
                    let charWidth = charHeight * 0.6  // Proper monospace width
                    
                    var currentX = x + width * 0.05  // Start with some padding
                    let textY = y + (frameHeight - charHeight) * 0.5
                    
                    // Generate vertices for each character
                    let effectiveCharWidth = charWidth * 0.3  // Account for tighter spacing
                    let maxChars = Int((width - width * 0.1) / effectiveCharWidth)  // Leave some padding
                    for char in simplifiedName.prefix(maxChars) {
                        let textColor = SIMD4<Float>(1, 1, 1, 1)  // White text
                        let charPosition = SIMD3<Float>(currentX, textY, 0)
                        let charSize = SIMD2<Float>(charWidth, charHeight)
                        
                        if let charVertices = createTextQuad(character: char, position: charPosition, size: charSize, color: textColor) {
                            let baseIndex = UInt32(textVertices.count)
                            textVertices.append(contentsOf: charVertices)
                            
                            textIndices.append(contentsOf: [
                                baseIndex, baseIndex + 1, baseIndex + 2,
                                baseIndex + 1, baseIndex + 3, baseIndex + 2
                            ])
                        }
                        
                        currentX += charWidth * 0.3  // Even tighter character spacing
                    }
                }
            }
        }
        
        // Update data bounds
        dataBounds = (minX: -1.0, maxX: -1.0 + timelineWidth, minY: minY, maxY: maxY)
        
        // Only setup initial view if this is the first time loading
        if camera.position.x == 0 && camera.position.y == 0 {
            setupInitialView()
        } else {
            // Just update bounds without resetting position
            let aspect = Float(bounds.width / bounds.height)
            camera.setupBounds(dataBounds: dataBounds, screenAspect: aspect)
        }
        
        
        
        
        print("Generated \(vertices.count) vertices and \(indices.count) indices")
        print("Generated \(textVertices.count) text vertices")
        
        vertexBuffer = metalDevice.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<Vertex>.size, options: [])
        indexBuffer = metalDevice.makeBuffer(bytes: indices, length: indices.count * MemoryLayout<UInt16>.size, options: [])
        
        if !textVertices.isEmpty {
            textVertexBuffer = metalDevice.makeBuffer(bytes: textVertices, length: textVertices.count * MemoryLayout<TextVertex>.size, options: [])
            textIndexBuffer = metalDevice.makeBuffer(bytes: textIndices, length: textIndices.count * MemoryLayout<UInt32>.size, options: [])
        }
    }
    
    private func createTestGeometry() {
        let vertices = [
            Vertex(position: SIMD3<Float>(-0.5, -0.5, 0), color: SIMD4<Float>(1, 0, 0, 1)),
            Vertex(position: SIMD3<Float>(0.5, -0.5, 0), color: SIMD4<Float>(0, 1, 0, 1)),
            Vertex(position: SIMD3<Float>(0.0, 0.5, 0), color: SIMD4<Float>(0, 0, 1, 1))
        ]
        
        let indices: [UInt16] = [0, 1, 2]
        
        vertexBuffer = metalDevice.makeBuffer(bytes: vertices, length: vertices.count * MemoryLayout<Vertex>.size, options: [])
        indexBuffer = metalDevice.makeBuffer(bytes: indices, length: indices.count * MemoryLayout<UInt16>.size, options: [])
    }
    
    private func colorForFunction(_ functionName: String) -> SIMD4<Float> {
        let hash = abs(functionName.hashValue)
        let r = Float((hash >> 16) & 0xFF) / 255.0
        let g = Float((hash >> 8) & 0xFF) / 255.0
        let b = Float(hash & 0xFF) / 255.0
        return SIMD4<Float>(r, g, b, 0.8)
    }
    
    private func hsvToRgb(h: Float, s: Float, v: Float) -> SIMD4<Float> {
        let c = v * s
        let x = c * (1 - abs((h * 6).truncatingRemainder(dividingBy: 2) - 1))
        let m = v - c
        
        let (r, g, b): (Float, Float, Float)
        switch Int(h * 6) {
        case 0: (r, g, b) = (c, x, 0)
        case 1: (r, g, b) = (x, c, 0)
        case 2: (r, g, b) = (0, c, x)
        case 3: (r, g, b) = (0, x, c)
        case 4: (r, g, b) = (x, 0, c)
        default: (r, g, b) = (c, 0, x)
        }
        
        return SIMD4<Float>(r + m, g + m, b + m, 1.0)
    }
    
    override func draw(_ rect: NSRect) {
        guard let drawable = currentDrawable,
              let renderPassDescriptor = currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1.0)
        
        let uniforms = Uniforms(modelViewProjectionMatrix: matrix_multiply(projectionMatrix, viewMatrix))
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
        
        // Render text if we have text geometry (using identity matrix for screen space)
        if let textVertexBuffer = textVertexBuffer, let textIndexBuffer = textIndexBuffer {
            renderEncoder.setRenderPipelineState(textRenderPipelineState)
            renderEncoder.setVertexBuffer(textVertexBuffer, offset: 0, index: 0)
            
            // Use the same view matrix as the flame graph so text aligns with rectangles
            let textUniforms = Uniforms(modelViewProjectionMatrix: matrix_multiply(projectionMatrix, viewMatrix))
            let textUniformsPointer = textUniformBuffer.contents().bindMemory(to: Uniforms.self, capacity: 1)
            textUniformsPointer.pointee = textUniforms
            
            renderEncoder.setVertexBuffer(textUniformBuffer, offset: 0, index: 1)
            renderEncoder.setFragmentTexture(fontAtlasTexture, index: 0)
            renderEncoder.setFragmentSamplerState(samplerState, index: 0)
            
            let textIndexCount = textIndexBuffer.length / MemoryLayout<UInt32>.size
            renderEncoder.drawIndexedPrimitives(type: .triangle,
                                              indexCount: textIndexCount,
                                              indexType: .uint32,
                                              indexBuffer: textIndexBuffer,
                                              indexBufferOffset: 0)
        }
        
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    private func drawTextOverlay() {
        guard let context = NSGraphicsContext.current?.cgContext else { 
            print("No graphics context available")
            return 
        }
        
        print("Drawing text overlay for \(frameRects.count) frames")
        
        // Transform coordinates from normalized device coordinates to view coordinates
        let mvpMatrix = matrix_multiply(projectionMatrix, self.viewMatrix)
        
        var visibleCount = 0
        for frameData in frameRects {
            let rect = frameData.rect
            let functionName = frameData.functionName
            
            // Transform rect from normalized coordinates to view coordinates
            let transformedRect = transformRectToView(rect, using: mvpMatrix)
            
            // Only draw text if the rect is large enough and visible
            if transformedRect.width > 30 && transformedRect.height > 10 && 
               transformedRect.maxX > 0 && transformedRect.minX < bounds.width &&
               transformedRect.maxY > 0 && transformedRect.minY < bounds.height {
                visibleCount += 1
                drawText(functionName, in: transformedRect, context: context)
            }
        }
        print("Drew text for \(visibleCount) visible frames")
    }
    
    private func transformRectToView(_ rect: NSRect, using matrix: matrix_float4x4) -> NSRect {
        // Transform the rectangle corners
        let bottomLeft = transformPoint(SIMD4<Float>(Float(rect.minX), Float(rect.minY), 0, 1), using: matrix)
        let topRight = transformPoint(SIMD4<Float>(Float(rect.maxX), Float(rect.maxY), 0, 1), using: matrix)
        
        // Convert from normalized device coordinates (-1 to 1) to view coordinates
        let viewSize = bounds.size
        let x = CGFloat((bottomLeft.x + 1) * 0.5) * viewSize.width
        let y = CGFloat((bottomLeft.y + 1) * 0.5) * viewSize.height
        let width = CGFloat((topRight.x - bottomLeft.x) * 0.5) * viewSize.width
        let height = CGFloat((topRight.y - bottomLeft.y) * 0.5) * viewSize.height
        
        return NSRect(x: x, y: y, width: width, height: height)
    }
    
    private func transformPoint(_ point: SIMD4<Float>, using matrix: matrix_float4x4) -> SIMD4<Float> {
        return matrix * point
    }
    
    private func drawText(_ text: String, in rect: NSRect, context: CGContext) {
        // Shorten function names for display
        let displayText = simplifyFunctionName(text)
        
        context.saveGState()
        
        // Set text attributes
        let fontSize = min(12.0, rect.height * 0.8)
        let font = NSFont.systemFont(ofSize: fontSize)
        let textColor = NSColor.white
        
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: textColor
        ]
        
        let attributedString = NSAttributedString(string: displayText, attributes: attributes)
        let textSize = attributedString.size()
        
        // Center text in rectangle
        let textRect = NSRect(
            x: rect.midX - textSize.width / 2,
            y: rect.midY - textSize.height / 2,
            width: textSize.width,
            height: textSize.height
        )
        
        attributedString.draw(in: textRect)
        
        context.restoreGState()
    }
    
    private func simplifyFunctionName(_ functionName: String) -> String {
        // Remove common prefixes and suffixes to make names more readable
        var simplified = functionName
        
        // Remove Rust mangling
        if simplified.contains("::h") {
            simplified = String(simplified.prefix(while: { $0 != ":" }))
        }
        
        // Remove template parameters
        if let angleIndex = simplified.firstIndex(of: "<") {
            simplified = String(simplified.prefix(upTo: angleIndex))
        }
        
        // Remove module paths, keep only the function name
        if let lastDoubleColon = simplified.range(of: "::", options: .backwards) {
            simplified = String(simplified[lastDoubleColon.upperBound...])
        }
        
        // Truncate if still too long
        if simplified.count > 20 {
            simplified = String(simplified.prefix(17)) + "..."
        }
        
        return simplified
    }
    
    override func viewDidEndLiveResize() {
        super.viewDidEndLiveResize()
        updateProjectionMatrix()
    }
}

func matrix_orthographic(left: Float, right: Float, bottom: Float, top: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let width = right - left
    let height = top - bottom
    let depth = farZ - nearZ
    
    return matrix_float4x4(columns: (
        SIMD4<Float>(2.0 / width, 0, 0, 0),
        SIMD4<Float>(0, 2.0 / height, 0, 0),
        SIMD4<Float>(0, 0, -2.0 / depth, 0),
        SIMD4<Float>(-(right + left) / width, -(top + bottom) / height, -(farZ + nearZ) / depth, 1)
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