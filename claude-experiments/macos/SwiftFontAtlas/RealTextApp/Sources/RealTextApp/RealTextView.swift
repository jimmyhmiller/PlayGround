import SwiftUI
import SwiftFontAtlas
import AppKit
import CoreGraphics
import MetalKit
import simd

struct RealTextView: View {
    @StateObject private var viewModel = RealTextViewModel()
    
    var body: some View {
        HSplitView {
            // Controls
            VStack(alignment: .leading, spacing: 16) {
                Text("Real Text Renderer")
                    .font(.title)
                    .fontWeight(.bold)
                
                GroupBox("Font Settings") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Font:")
                            Picker("", selection: $viewModel.selectedFont) {
                                Text("SF Mono").tag("SF Mono")
                                Text("Menlo").tag("Menlo")
                                Text("Monaco").tag("Monaco")
                                Text("Courier New").tag("Courier New")
                                Text("Helvetica").tag("Helvetica")
                                Text("System Monospace").tag("System Monospace")
                            }
                            .pickerStyle(.menu)
                        }
                        
                        HStack {
                            Text("Size:")
                            Slider(value: $viewModel.fontSize, in: 12...72, step: 1)
                            Text("\(Int(viewModel.fontSize))pt")
                                .frame(width: 35)
                        }
                        
                        Button("Apply Changes") {
                            viewModel.createAtlas()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(viewModel.isCreating)
                    }
                }
                
                GroupBox("Text Input") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Type your text:")
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
                                viewModel.clearText()
                            }
                        }
                        
                        Button("Load Demo Text") {
                            viewModel.loadDemoText()
                        }
                        .buttonStyle(.bordered)
                    }
                }
                
                GroupBox("View Options") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Scale:")
                            Slider(value: $viewModel.renderScale, in: 0.5...4.0, step: 0.1)
                            Text("\(String(format: "%.1f", viewModel.renderScale))x")
                                .frame(width: 35)
                        }
                        
                        Toggle("Show Atlas", isOn: $viewModel.showAtlas)
                    }
                }
                
                GroupBox("Statistics") {
                    VStack(alignment: .leading, spacing: 4) {
                        if let stats = viewModel.stats {
                            Text("Atlas: \(stats.atlasSize)×\(stats.atlasSize)")
                            Text("Rendered: \(stats.charCount) chars")
                            Text("Memory: \(stats.memoryUsage)")
                            Text("Utilization: \(stats.utilization)%")
                        } else {
                            Text("No atlas created")
                                .foregroundColor(.secondary)
                        }
                    }
                    .font(.system(.caption, design: .monospaced))
                }
                
                Spacer()
            }
            .frame(width: 320)
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            // Text rendering area
            VStack {
                if viewModel.hasAtlas {
                    ScrollView([.horizontal, .vertical]) {
                        VStack {
                            MetalTextView(viewModel: viewModel)
                                .scaleEffect(viewModel.renderScale)
                                .background(Color.white)
                                .cornerRadius(4)
                                .frame(width: 800, height: 600)
                            
                            if viewModel.showAtlas, let atlasImage = viewModel.atlasImage {
                                VStack {
                                    Text("Font Atlas Texture")
                                        .font(.headline)
                                        .padding(.top)
                                    
                                    Image(nsImage: atlasImage)
                                        .interpolation(.none)
                                        .scaleEffect(2.0)
                                        .background(Color.black)
                                        .cornerRadius(4)
                                    
                                    Text("White pixels = rendered glyphs")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                        .padding()
                    }
                } else {
                    VStack(spacing: 20) {
                        Image(systemName: "textformat")
                            .font(.system(size: 64))
                            .foregroundColor(.secondary)
                        
                        Text("Real Text Rendering")
                            .font(.title2)
                            .foregroundColor(.secondary)
                        
                        Text("This app renders actual readable text using SwiftFontAtlas. Configure your font settings and start typing!")
                            .multilineTextAlignment(.center)
                            .foregroundColor(.secondary)
                            .frame(maxWidth: 400)
                        
                        Button("Get Started") {
                            viewModel.createAtlas()
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .padding()
        }
        .onAppear {
            print("✅ Real text app appeared!")
            NSApp.activate(ignoringOtherApps: true)
        }
    }
}

@MainActor
class RealTextViewModel: NSObject, ObservableObject, MTKViewDelegate {
    @Published var selectedFont = "Menlo"
    @Published var fontSize: Double = 56
    @Published var inputText = "Hello, SwiftFontAtlas!\n\nThis is real text being rendered\nthrough our custom font atlas system.\n\nYou can type anything here and see\nit rendered with proper typography!"
    @Published var hasAtlas = false
    @Published var isCreating = false
    @Published var renderScale: Double = 1.0
    @Published var showAtlas = false
    @Published var stats: Stats?
    @Published var atlasImage: NSImage?
    
    nonisolated(unsafe) var fontManager: FontAtlasManager?
    nonisolated(unsafe) var currentInputText: String = ""
    
    // Metal properties
    var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var renderPipelineState: MTLRenderPipelineState!
    private var atlasTexture: MTLTexture?
    private var vertexBuffer: MTLBuffer!
    private var uniformBuffer: MTLBuffer!
    
    struct Stats {
        let atlasSize: UInt32
        let charCount: Int
        let memoryUsage: String
        let utilization: String
    }
    
    override init() {
        super.init()
        initializeMetal()
        createAtlas()
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
    
    private func setupRenderPipeline() {
        // Try to load the shader source and compile it manually
        guard let path = Bundle.main.path(forResource: "Shaders", ofType: "metal"),
              let source = try? String(contentsOfFile: path) else {
            print("⚠️ Could not load Shaders.metal, using embedded shader source")
            setupRenderPipelineWithEmbeddedShaders()
            return
        }
        
        do {
            let library = try device.makeLibrary(source: source, options: nil)
            print("📚 Available functions: \(library.functionNames)")
            
            guard let vertexFunction = library.makeFunction(name: "vertex_main") else {
                fatalError("Failed to create vertex function 'vertex_main'")
            }
            
            guard let fragmentFunction = library.makeFunction(name: "fragment_main") else {
                fatalError("Failed to create fragment function 'fragment_main'")
            }
            
            createPipelineState(vertex: vertexFunction, fragment: fragmentFunction)
        } catch {
            print("❌ Failed to compile shaders: \(error)")
            setupRenderPipelineWithEmbeddedShaders()
        }
    }
    
    private func setupRenderPipelineWithEmbeddedShaders() {
        let shaderSource = """
#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float2 position;
    float2 texCoord;
    float4 color;
};

struct Uniforms {
    float4x4 projectionMatrix;
    float2 screenSize;
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
    float4 color;
};

vertex VertexOut vertex_main(uint vertexID [[vertex_id]],
                             constant Vertex* vertices [[buffer(0)]],
                             constant Uniforms& uniforms [[buffer(1)]]) {
    VertexOut out;
    
    Vertex in = vertices[vertexID];
    
    out.position = uniforms.projectionMatrix * float4(in.position, 0.0, 1.0);
    out.texCoord = in.texCoord;
    out.color = in.color;
    
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]],
                              texture2d<float> atlasTexture [[texture(0)]],
                              constant Uniforms& uniforms [[buffer(1)]]) {
    constexpr sampler textureSampler(coord::normalized,
                                     address::clamp_to_edge,
                                     filter::linear);
    
    // Sample the atlas texture (grayscale)
    float alpha = atlasTexture.sample(textureSampler, in.texCoord).r;
    
    // Use the sampled alpha with the vertex color
    float4 color = in.color;
    color.a *= alpha;
    
    return color;
}
"""
        
        do {
            let library = try device.makeLibrary(source: shaderSource, options: nil)
            let vertexFunction = library.makeFunction(name: "vertex_main")!
            let fragmentFunction = library.makeFunction(name: "fragment_main")!
            
            createPipelineState(vertex: vertexFunction, fragment: fragmentFunction)
        } catch {
            fatalError("Failed to compile embedded shaders: \(error)")
        }
    }
    
    private func createPipelineState(vertex: MTLFunction, fragment: MTLFunction) {
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertex
        pipelineDescriptor.fragmentFunction = fragment
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
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
            fatalError("Failed to create render pipeline state: \(error)")
        }
    }
    
    private func createBuffers() {
        let maxVertices = 10000 // Support lots of text
        let vertexBufferSize = maxVertices * MemoryLayout<TextVertex>.size
        vertexBuffer = device.makeBuffer(length: vertexBufferSize, options: [])
        
        uniformBuffer = device.makeBuffer(length: MemoryLayout<TextUniforms>.size, options: [])
    }
    
    // MARK: - MTKViewDelegate
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        updateProjectionMatrix(size: size)
    }
    
    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }
        
        renderEncoder.setRenderPipelineState(renderPipelineState)
        
        // Generate vertices for current text
        let vertices = generateVertices()
        if !vertices.isEmpty {
            // Update vertex buffer
            let bufferPointer = vertexBuffer.contents().bindMemory(to: TextVertex.self, capacity: vertices.count)
            for (index, vertex) in vertices.enumerated() {
                bufferPointer[index] = vertex
            }
            
            renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            renderEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
            renderEncoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 1)
            
            if let texture = atlasTexture {
                renderEncoder.setFragmentTexture(texture, index: 0)
                print("✅ Atlas texture bound: \(texture.width)x\(texture.height)")
            } else {
                print("⚠️ No atlas texture available!")
            }
            
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: vertices.count)
            print("🎨 Drawing \(vertices.count) vertices")
        }
        
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    private func generateVertices() -> [TextVertex] {
        guard let manager = fontManager else { return [] }
        
        var vertices: [TextVertex] = []
        let lines = currentInputText.components(separatedBy: CharacterSet.newlines)
        let lineHeight = manager.lineHeight
        
        // print("📊 Font metrics: ascent=\(manager.metrics.ascent), descent=\(manager.metrics.descent), lineHeight=\(lineHeight)")
        
        // Match TextRenderingApp exactly: simple baseline positioning
        let startY: Float = 50.0
        let textLineHeight: Float = Float(manager.cellSize.height + 4)
        
        for (lineIndex, line) in lines.enumerated() {
            // Use the same simple approach as the working TextRenderingApp
            var currentX: Float = 20.0
            let currentY: Float = startY + Float(lineIndex) * textLineHeight
            
            // print("📏 Line \(lineIndex): Y=\(currentY), processing '\(line)'")
            
            for character in line {
                if let glyph = manager.renderCharacter(character) {
                    if glyph.width > 0 && glyph.height > 0 {

                        // Use the library's baselinePosition method directly
                        let glyphX = currentX + Float(glyph.offsetX)
                        let glyphY = currentY - Float(glyph.height) - Float(glyph.offsetY)
                        
                        let textColor = simd_float4(0.0, 0.0, 0.0, 1.0)
                        let quad = createTextQuad(
                            glyph: glyph,
                            x: glyphX,
                            y: glyphY,
                            color: textColor,
                            atlasSize: Float(manager.withAtlas { $0.size })
                        )
                        vertices.append(contentsOf: quad)
                    } else {
                        print("📍 Char '\(character)': empty glyph (space?)")
                    }
                    
                    // Always advance, even for empty glyphs (spaces)
                    currentX += glyph.advanceX
                }
            }
        }
        
        // Add test quad if no text - this helps debug if the shader pipeline is working
        // if vertices.isEmpty {
        //     let testColor = simd_float4(1.0, 0.0, 0.0, 1.0)
        //     vertices = [
        //         TextVertex(position: simd_float2(100, 100), texCoord: simd_float2(0, 0), color: testColor),
        //         TextVertex(position: simd_float2(200, 100), texCoord: simd_float2(1, 0), color: testColor),
        //         TextVertex(position: simd_float2(100, 200), texCoord: simd_float2(0, 1), color: testColor),
        //         TextVertex(position: simd_float2(200, 100), texCoord: simd_float2(1, 0), color: testColor),
        //         TextVertex(position: simd_float2(200, 200), texCoord: simd_float2(1, 1), color: testColor),
        //         TextVertex(position: simd_float2(100, 200), texCoord: simd_float2(0, 1), color: testColor)
        //     ]
        // }
        
        return vertices
    }
    
    private func createTextQuad(glyph: RenderedGlyph, x: Float, y: Float, color: simd_float4, atlasSize: Float) -> [TextVertex] {
        let left = x
        let right = x + Float(glyph.width)
        let top = y
        let bottom = y + Float(glyph.height)
        
        let u1 = Float(glyph.atlasX) / atlasSize
        let v1 = Float(glyph.atlasY) / atlasSize
        let u2 = Float(glyph.atlasX + glyph.width) / atlasSize
        let v2 = Float(glyph.atlasY + glyph.height) / atlasSize
        
        // print("📐 Glyph UV: (\(u1),\(v1)) to (\(u2),\(v2)) atlas:\(glyph.atlasX),\(glyph.atlasY) size:\(glyph.width)x\(glyph.height)")
        
        return [
            TextVertex(position: simd_float2(left, top), texCoord: simd_float2(u1, v1), color: color),
            TextVertex(position: simd_float2(right, top), texCoord: simd_float2(u2, v1), color: color),
            TextVertex(position: simd_float2(left, bottom), texCoord: simd_float2(u1, v2), color: color),
            TextVertex(position: simd_float2(right, top), texCoord: simd_float2(u2, v1), color: color),
            TextVertex(position: simd_float2(right, bottom), texCoord: simd_float2(u2, v2), color: color),
            TextVertex(position: simd_float2(left, bottom), texCoord: simd_float2(u1, v2), color: color)
        ]
    }
    
    private func updateProjectionMatrix(size: CGSize) {
        guard size.width > 0 && size.height > 0 else { return }
        
        let projectionMatrix = orthographicProjection(
            left: 0,
            right: Float(size.width),
            bottom: Float(size.height),
            top: 0,
            near: -1,
            far: 1
        )
        
        let uniforms = TextUniforms(
            projectionMatrix: projectionMatrix,
            screenSize: simd_float2(Float(size.width), Float(size.height))
        )
        
        let pointer = uniformBuffer.contents().bindMemory(to: TextUniforms.self, capacity: 1)
        pointer.pointee = uniforms
    }
    
    private func orthographicProjection(left: Float, right: Float, bottom: Float, top: Float, near: Float, far: Float) -> simd_float4x4 {
        return simd_float4x4(
            simd_float4(2 / (right - left), 0, 0, 0),
            simd_float4(0, 2 / (top - bottom), 0, 0),
            simd_float4(0, 0, -2 / (far - near), 0),
            simd_float4(-(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1)
        )
    }
    
    private func createAtlasTexture() {
        guard let manager = fontManager,
              let device = device else { return }
        
        manager.withAtlas { atlas in
            if let texture = atlas.createTexture(device: device) {
                atlas.updateTexture(texture)
                atlasTexture = texture
                print("✅ Created atlas texture: \(texture.width)x\(texture.height)")
                
                // Debug: Check if atlas has any data
                var nonZeroCount = 0
                for byte in atlas.data {
                    if byte != 0 { nonZeroCount += 1 }
                }
                print("📊 Atlas data: \(nonZeroCount) non-zero pixels out of \(atlas.data.count)")
            } else {
                print("❌ Failed to create atlas texture")
            }
        }
    }
    
    func createAtlas() {
        isCreating = true
        print("🚀 Creating font atlas with \(selectedFont) \(Int(fontSize))pt...")
        
        Task {
            do {
                let actualFontName = selectedFont == "System Monospace" ? ".AppleSystemUIFontMonospaced" : selectedFont
                
                let manager = try FontAtlasManager(
                    fontName: actualFontName,
                    fontSize: Float(fontSize),
                    atlasSize: 1024
                )
                
                let asciiCount = manager.prerenderASCII()
                print("✅ Pre-rendered \(asciiCount) ASCII characters")
                
                await MainActor.run {
                    self.fontManager = manager
                    self.createAtlasTexture()
                    self.hasAtlas = true
                    self.isCreating = false
                    self.updateAtlasImage()
                    self.updateStats()
                    self.renderText()
                    print("✅ Font atlas created with \(selectedFont) \(Int(fontSize))pt")
                }
            } catch {
                await MainActor.run {
                    self.isCreating = false
                    print("❌ Failed to create atlas: \(error)")
                }
            }
        }
    }
    
    func renderText() {
        guard let manager = fontManager else { return }
        
        currentInputText = inputText
        _ = manager.prerenderString(inputText)
        
        // Update the atlas texture after rendering new glyphs
        if let texture = atlasTexture {
            manager.withAtlas { atlas in
                atlas.updateTexture(texture)
                print("🔄 Updated atlas texture after rendering text")
            }
        }
        
        updateAtlasImage()
        updateStats()
        
        print("✅ Text prepared for Metal rendering: \(inputText.count) characters")
    }
    
    func clearText() {
        inputText = ""
    }
    
    func loadDemoText() {
        inputText = """
        SwiftFontAtlas Demo
        ==================
        
        This text is rendered using our custom font atlas!
        
        Features:
        • Efficient rectangle bin packing
        • High-quality CoreText rendering
        • Unicode support: áéíóú αβγδε 🚀
        • Thread-safe operations
        • Metal-ready textures
        
        Lorem ipsum dolor sit amet, consectetur
        adipiscing elit. Sed do eiusmod tempor
        incididunt ut labore et dolore magna aliqua.
        
        The quick brown fox jumps over the lazy dog.
        THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG.
        1234567890 !@#$%^&*()_+-=[]{}|;:,.<>?
        """
        renderText()
    }
    
    private func updateAtlasImage() {
        guard let manager = fontManager else { return }
        
        manager.withAtlas { atlas in
            atlasImage = createAtlasImage(from: atlas)
        }
    }
    
    private func createAtlasImage(from atlas: FontAtlas) -> NSImage? {
        let size = Int(atlas.size)
        
        guard let context = CGContext(
            data: nil,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }
        
        atlas.data.withUnsafeBytes { bytes in
            if let data = context.data {
                data.copyMemory(from: bytes.baseAddress!, byteCount: atlas.data.count)
            }
        }
        
        guard let cgImage = context.makeImage() else { return nil }
        return NSImage(cgImage: cgImage, size: NSSize(width: size, height: size))
    }
    
    private func updateStats() {
        guard let manager = fontManager else {
            stats = nil
            return
        }
        
        manager.withAtlas { atlas in
            let memoryMB = Double(atlas.data.count) / (1024 * 1024)
            
            var nonZeroPixels = 0
            for byte in atlas.data {
                if byte != 0 { nonZeroPixels += 1 }
            }
            let utilization = Double(nonZeroPixels) / Double(atlas.data.count) * 100
            
            stats = Stats(
                atlasSize: atlas.size,
                charCount: inputText.count,
                memoryUsage: String(format: "%.1f MB", memoryMB),
                utilization: String(format: "%.1f", utilization)
            )
        }
    }
}