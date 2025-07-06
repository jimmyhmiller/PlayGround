import SwiftUI
import SwiftFontAtlas
import Metal
import MetalKit

@main
struct TextRenderingApp: App {
    init() {
        print("🚀 Starting Text Rendering App...")
        print("   This app renders text using SwiftFontAtlas + Metal")
        
        // Force app to front
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.regular)
            NSApp.activate(ignoringOtherApps: true)
        }
    }
    
    var body: some Scene {
        WindowGroup("SwiftFontAtlas Text Renderer") {
            ContentView()
        }
        .defaultSize(width: 1000, height: 700)
    }
}

struct ContentView: View {
    @StateObject private var viewModel = TextRenderingViewModel()
    
    var body: some View {
        HSplitView {
            // Left panel - Controls
            VStack(alignment: .leading, spacing: 16) {
                Text("Text Renderer")
                    .font(.title)
                    .fontWeight(.bold)
                
                // Font settings
                GroupBox("Font Settings") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Font:")
                            Picker("", selection: $viewModel.selectedFont) {
                                ForEach(viewModel.availableFonts, id: \.self) { font in
                                    Text(font).tag(font)
                                }
                            }
                            .pickerStyle(.menu)
                        }
                        
                        HStack {
                            Text("Size:")
                            Slider(value: $viewModel.fontSize, in: 8...48, step: 1)
                            Text("\(Int(viewModel.fontSize))pt")
                                .frame(width: 35)
                        }
                        
                        Button("Apply Font Changes") {
                            viewModel.createFontAtlas()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(viewModel.isCreating)
                    }
                }
                
                // Text input
                GroupBox("Text to Render") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Enter text to render:")
                        
                        TextEditor(text: $viewModel.inputText)
                            .font(.system(.body, design: .monospaced))
                            .frame(height: 100)
                            .border(Color.gray.opacity(0.3))
                        
                        HStack {
                            Button("Render Text") {
                                viewModel.renderText()
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(!viewModel.hasAtlas)
                            
                            Button("Clear") {
                                viewModel.clearRenderedText()
                            }
                            .disabled(viewModel.renderedQuads.isEmpty)
                        }
                        
                        HStack {
                            Button("Demo Text") {
                                viewModel.inputText = "Hello, SwiftFontAtlas!\nThis text is rendered using\nour custom font atlas system.\n\nFeatures:\n• Efficient rectangle packing\n• Metal GPU rendering\n• Unicode support: αβγ 🚀\n• Thread-safe operations"
                                viewModel.renderText()
                            }
                            .buttonStyle(.bordered)
                            
                            Button("ASCII Test") {
                                viewModel.inputText = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()"
                                viewModel.renderText()
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
                
                // Rendering options
                GroupBox("Rendering Options") {
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle("Show Atlas", isOn: $viewModel.showAtlas)
                        Toggle("Show Glyph Bounds", isOn: $viewModel.showGlyphBounds)
                        
                        HStack {
                            Text("Text Color:")
                            ColorPicker("", selection: $viewModel.textColor)
                                .frame(width: 50)
                        }
                        
                        HStack {
                            Text("Background:")
                            ColorPicker("", selection: $viewModel.backgroundColor)
                                .frame(width: 50)
                        }
                    }
                }
                
                // Statistics
                GroupBox("Statistics") {
                    VStack(alignment: .leading, spacing: 4) {
                        if let stats = viewModel.stats {
                            Text("Atlas: \(stats.atlasSize)×\(stats.atlasSize)")
                            Text("Glyphs: \(stats.glyphCount)")
                            Text("Quads: \(stats.quadCount)")
                            Text("Memory: \(stats.memoryUsage)")
                            Text("FPS: \(stats.fps)")
                        } else {
                            Text("No atlas created")
                                .foregroundColor(.secondary)
                        }
                    }
                    .font(.system(.caption, design: .monospaced))
                }
                
                Spacer()
            }
            .frame(width: 280)
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            // Main rendering area
            VStack {
                if viewModel.hasAtlas {
                    MetalTextView(viewModel: viewModel)
                        .background(Color.black)
                        .cornerRadius(4)
                } else {
                    VStack(spacing: 20) {
                        Image(systemName: "textformat")
                            .font(.system(size: 64))
                            .foregroundColor(.secondary)
                        
                        VStack(spacing: 8) {
                            Text("Text Renderer Ready")
                                .font(.title2)
                                .foregroundColor(.secondary)
                            
                            Text("Configure font settings and click 'Apply Font Changes' to start rendering text with SwiftFontAtlas.")
                                .multilineTextAlignment(.center)
                                .foregroundColor(.secondary)
                                .frame(maxWidth: 400)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .padding()
        }
        .onAppear {
            print("✅ Text rendering app appeared!")
            NSApp.activate(ignoringOtherApps: true)
            viewModel.initializeMetal()
        }
    }
}

struct MetalTextView: NSViewRepresentable {
    let viewModel: TextRenderingViewModel
    
    func makeNSView(context: Context) -> MTKView {
        let metalView = MTKView()
        metalView.device = viewModel.device
        metalView.delegate = viewModel
        metalView.preferredFramesPerSecond = 60
        metalView.enableSetNeedsDisplay = false
        metalView.isPaused = false
        metalView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        
        print("✅ Metal view created")
        return metalView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        // Updates handled by MTKViewDelegate
    }
}

@MainActor
class TextRenderingViewModel: NSObject, ObservableObject, MTKViewDelegate {
    // UI Properties
    @Published var selectedFont = "SF Mono"
    @Published var fontSize: Double = 18
    @Published var inputText = "Hello, SwiftFontAtlas!\n\nType your text here and click 'Render Text' to see it rendered using our custom font atlas system."
    @Published var hasAtlas = false
    @Published var isCreating = false
    @Published var showAtlas = false
    @Published var showGlyphBounds = false
    @Published var textColor = Color.white
    @Published var backgroundColor = Color.black
    @Published var stats: RenderStats?
    @Published var renderedQuads: [TextQuad] = []
    
    let availableFonts = ["SF Mono", "Menlo", "Monaco", "Helvetica", "Arial", "Times"]
    
    // Metal and rendering
    var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var renderPipelineState: MTLRenderPipelineState!
    private var fontManager: FontAtlasManager?
    private var atlasTexture: MTLTexture?
    private var vertexBuffer: MTLBuffer!
    private var uniformBuffer: MTLBuffer!
    private var lastFrameTime: CFAbsoluteTime = 0
    private var frameCount = 0
    
    struct RenderStats {
        let atlasSize: UInt32
        let glyphCount: Int
        let quadCount: Int
        let memoryUsage: String
        let fps: String
    }
    
    struct TextQuad {
        let position: SIMD2<Float>
        let size: SIMD2<Float>
        let texCoords: SIMD4<Float> // u1, v1, u2, v2
        let color: SIMD4<Float>
    }
    
    struct Vertex {
        let position: SIMD2<Float>
        let texCoord: SIMD2<Float>
        let color: SIMD4<Float>
    }
    
    struct Uniforms {
        let projectionMatrix: simd_float4x4
        let screenSize: SIMD2<Float>
    }
    
    override init() {
        super.init()
        lastFrameTime = CFAbsoluteTimeGetCurrent()
    }
    
    func initializeMetal() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("❌ Failed to create Metal device")
            return
        }
        
        self.device = device
        self.commandQueue = device.makeCommandQueue()
        
        setupRenderPipeline()
        createBuffers()
        
        print("✅ Metal initialized")
    }
    
    func createFontAtlas() {
        isCreating = true
        
        Task {
            do {
                let manager = try FontAtlasManager(
                    fontName: selectedFont,
                    fontSize: Float(fontSize),
                    atlasSize: 512
                )
                
                // Pre-render ASCII for better performance
                _ = manager.prerenderASCII()
                
                await MainActor.run {
                    self.fontManager = manager
                    self.createAtlasTexture()
                    self.hasAtlas = true
                    self.isCreating = false
                    self.updateStats()
                    print("✅ Font atlas created and ready for rendering")
                }
            } catch {
                await MainActor.run {
                    self.isCreating = false
                    print("❌ Failed to create font atlas: \(error)")
                }
            }
        }
    }
    
    func renderText() {
        guard let manager = fontManager else { return }
        
        renderedQuads.removeAll()
        
        let lines = inputText.components(separatedBy: .newlines)
        var currentY: Float = 50.0
        let lineHeight = Float(manager.cellSize.height + 4)
        
        for line in lines {
            var currentX: Float = 20.0
            
            for character in line {
                if let glyph = manager.renderCharacter(character) {
                    let quad = TextQuad(
                        position: SIMD2(currentX + Float(glyph.offsetX), currentY - Float(glyph.offsetY)),
                        size: SIMD2(Float(glyph.width), Float(glyph.height)),
                        texCoords: getTextureCoordinates(for: glyph),
                        color: colorToSIMD4(textColor)
                    )
                    
                    renderedQuads.append(quad)
                    currentX += glyph.advanceX
                }
            }
            
            currentY += lineHeight
        }
        
        createAtlasTexture() // Update texture with new glyphs
        updateStats()
        
        print("✅ Rendered \(renderedQuads.count) quads for text")
    }
    
    func clearRenderedText() {
        renderedQuads.removeAll()
        updateStats()
    }
    
    private func getTextureCoordinates(for glyph: RenderedGlyph) -> SIMD4<Float> {
        guard let manager = fontManager else { return SIMD4(0, 0, 0, 0) }
        
        var coords: SIMD4<Float> = SIMD4(0, 0, 0, 0)
        manager.withAtlas { atlas in
            let (u1, v1, u2, v2) = atlas.normalizedCoordinates(for: glyph)
            coords = SIMD4(u1, v1, u2, v2)
        }
        return coords
    }
    
    private func colorToSIMD4(_ color: Color) -> SIMD4<Float> {
        let nsColor = NSColor(color)
        let ciColor = CIColor(color: nsColor)
        return SIMD4(Float(ciColor.red), Float(ciColor.green), Float(ciColor.blue), Float(ciColor.alpha))
    }
    
    private func createAtlasTexture() {
        guard let manager = fontManager else { return }
        
        manager.withAtlas { atlas in
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .r8Unorm,
                width: Int(atlas.size),
                height: Int(atlas.size),
                mipmapped: false
            )
            descriptor.usage = [.shaderRead]
            
            guard let texture = device.makeTexture(descriptor: descriptor) else {
                print("❌ Failed to create atlas texture")
                return
            }
            
            atlas.data.withUnsafeBytes { bytes in
                texture.replace(
                    region: MTLRegion(
                        origin: MTLOrigin(x: 0, y: 0, z: 0),
                        size: MTLSize(width: Int(atlas.size), height: Int(atlas.size), depth: 1)
                    ),
                    mipmapLevel: 0,
                    withBytes: bytes.baseAddress!,
                    bytesPerRow: Int(atlas.size)
                )
            }
            
            self.atlasTexture = texture
        }
    }
    
    private func setupRenderPipeline() {
        let library = device.makeDefaultLibrary()
        
        let vertexFunction = library?.makeFunction(name: "vertex_main")
        let fragmentFunction = library?.makeFunction(name: "fragment_main")
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        // Enable blending for text rendering
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        do {
            renderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("❌ Failed to create render pipeline: \(error)")
        }
    }
    
    private func createBuffers() {
        // Create buffers for vertices and uniforms
        vertexBuffer = device.makeBuffer(length: 6 * 1000 * MemoryLayout<Vertex>.stride, options: [])
        uniformBuffer = device.makeBuffer(length: MemoryLayout<Uniforms>.stride, options: [])
    }
    
    private func updateStats() {
        guard let manager = fontManager else {
            stats = nil
            return
        }
        
        manager.withAtlas { atlas in
            let memoryMB = Double(atlas.data.count) / (1024 * 1024)
            
            stats = RenderStats(
                atlasSize: atlas.size,
                glyphCount: Int(atlas.modificationCount.withLock { $0 }),
                quadCount: renderedQuads.count,
                memoryUsage: String(format: "%.1f MB", memoryMB),
                fps: String(format: "%.0f", 60.0) // Approximate
            )
        }
    }
    
    // MARK: - MTKViewDelegate
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Handle resize if needed
    }
    
    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        // Set pipeline state
        renderEncoder.setRenderPipelineState(renderPipelineState)
        
        // Set uniforms
        let uniforms = Uniforms(
            projectionMatrix: orthographicMatrix(size: view.drawableSize),
            screenSize: SIMD2(Float(view.drawableSize.width), Float(view.drawableSize.height))
        )
        
        uniformBuffer.contents().copyMemory(
            from: [uniforms],
            byteCount: MemoryLayout<Uniforms>.stride
        )
        
        renderEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        renderEncoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 1)
        
        // Set atlas texture
        if let texture = atlasTexture {
            renderEncoder.setFragmentTexture(texture, index: 0)
        }
        
        // Generate vertices for text quads
        var vertices: [Vertex] = []
        
        for quad in renderedQuads {
            let v1 = Vertex(
                position: quad.position,
                texCoord: SIMD2(quad.texCoords.x, quad.texCoords.y),
                color: quad.color
            )
            let v2 = Vertex(
                position: SIMD2(quad.position.x + quad.size.x, quad.position.y),
                texCoord: SIMD2(quad.texCoords.z, quad.texCoords.y),
                color: quad.color
            )
            let v3 = Vertex(
                position: SIMD2(quad.position.x, quad.position.y + quad.size.y),
                texCoord: SIMD2(quad.texCoords.x, quad.texCoords.w),
                color: quad.color
            )
            let v4 = Vertex(
                position: SIMD2(quad.position.x + quad.size.x, quad.position.y + quad.size.y),
                texCoord: SIMD2(quad.texCoords.z, quad.texCoords.w),
                color: quad.color
            )
            
            // Two triangles per quad
            vertices.append(contentsOf: [v1, v2, v3, v2, v4, v3])
        }
        
        if !vertices.isEmpty {
            // Copy vertices to buffer
            let vertexData = UnsafeMutableRawPointer(vertexBuffer.contents())
            vertexData.copyMemory(from: vertices, byteCount: vertices.count * MemoryLayout<Vertex>.stride)
            
            renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: vertices.count)
        }
        
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
        
        // Update frame rate
        frameCount += 1
        let currentTime = CFAbsoluteTimeGetCurrent()
        if currentTime - lastFrameTime >= 1.0 {
            updateStats()
            lastFrameTime = currentTime
            frameCount = 0
        }
    }
    
    private func orthographicMatrix(size: CGSize) -> simd_float4x4 {
        let left: Float = 0
        let right = Float(size.width)
        let bottom = Float(size.height)
        let top: Float = 0
        let near: Float = -1
        let far: Float = 1
        
        return simd_float4x4(
            SIMD4(2 / (right - left), 0, 0, 0),
            SIMD4(0, 2 / (top - bottom), 0, 0),
            SIMD4(0, 0, -2 / (far - near), 0),
            SIMD4(-(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1)
        )
    }
}