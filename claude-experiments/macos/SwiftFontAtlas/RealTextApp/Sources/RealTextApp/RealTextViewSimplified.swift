import SwiftUI
import MetalKit
import SwiftFontAtlas
import Combine

struct RealTextViewSimplified: View {
    @StateObject private var viewModel = SimplifiedTextViewModel()
    @State private var showAtlasTexture = false
    @State private var showStatistics = true
    
    var body: some View {
        HSplitView {
            // Left panel - controls
            VStack(alignment: .leading, spacing: 20) {
                // Font Settings
                GroupBox(label: Label("Font Settings", systemImage: "textformat")) {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Font:")
                            TextField("Font Name", text: $viewModel.fontName)
                                .frame(width: 150)
                            Button("Apply") {
                                viewModel.recreateFontManager()
                            }
                        }
                        
                        HStack {
                            Text("Size:")
                            Slider(value: $viewModel.fontSize, in: 8...72, step: 1)
                                .frame(width: 120)
                            Text("\(Int(viewModel.fontSize))pt")
                                .frame(width: 40)
                        }
                        .onChange(of: viewModel.fontSize) { _, _ in
                            viewModel.recreateFontManager()
                        }
                    }
                }
                
                // Text Input
                GroupBox(label: Label("Text Input", systemImage: "text.alignleft")) {
                    VStack(alignment: .leading) {
                        TextEditor(text: $viewModel.currentInputText)
                            .font(.system(.body, design: .monospaced))
                            .frame(height: 150)
                        
                        Button("Load Sample Text") {
                            viewModel.loadSampleText()
                        }
                    }
                }
                
                // View Options
                GroupBox(label: Label("View Options", systemImage: "eye")) {
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle("Show Atlas Texture", isOn: $showAtlasTexture)
                        Toggle("Show Statistics", isOn: $showStatistics)
                    }
                }
                
                Spacer()
            }
            .padding()
            .frame(minWidth: 350, maxWidth: 400)
            
            // Right panel - Metal view
            VStack(spacing: 0) {
                SimplifiedMetalTextView(viewModel: viewModel)
                    .frame(maxHeight: showAtlasTexture ? .infinity : .infinity)
                
                if showAtlasTexture {
                    Divider()
                    // Simple texture view - for now just show a placeholder
                    Rectangle()
                        .fill(Color.gray.opacity(0.3))
                        .overlay(
                            Text("Atlas Texture")
                                .foregroundColor(.secondary)
                        )
                        .frame(height: 256)
                        .background(Color.black)
                }
                
                if showStatistics {
                    StatisticsView(viewModel: viewModel)
                        .padding()
                }
            }
        }
        .frame(minWidth: 800, minHeight: 600)
    }
}

// Simplified ViewModel using the high-level TextRenderer
@MainActor
class SimplifiedTextViewModel: NSObject, ObservableObject, MTKViewDelegate {
    @Published var fontName = "SF Mono"
    @Published var fontSize: Float = 16
    @Published var currentInputText = "Hello, SwiftFontAtlas!\nThis is the high-level API in action.\n\nFeatures:\n• Automatic newline handling\n• Proper text positioning\n• Easy to use!"
    @Published var atlasTexture: MTLTexture?
    @Published var glyphCount = 0
    @Published var atlasUtilization: Float = 0
    @Published var memoryUsage: String = "0 KB"
    
    var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var textRenderer: SwiftFontAtlas.TextRenderer?
    var metalView: MTKView?
    
    override init() {
        super.init()
        setupMetal()
        recreateFontManager()
        
        // Set up change observers
        $currentInputText
            .sink { [weak self] _ in
                self?.triggerRedraw()
            }
            .store(in: &cancellables)
        
        $fontName
            .sink { [weak self] _ in
                self?.recreateFontManager()
                self?.triggerRedraw()
            }
            .store(in: &cancellables)
        
        $fontSize
            .sink { [weak self] _ in
                self?.recreateFontManager()
                self?.triggerRedraw()
            }
            .store(in: &cancellables)
    }
    
    private var cancellables = Set<AnyCancellable>()
    
    private func triggerRedraw() {
        metalView?.needsDisplay = true
    }
    
    private func setupMetal() {
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
    }
    
    func recreateFontManager() {
        do {
            // Create text renderer with top-left origin for Metal
            textRenderer = try SwiftFontAtlas.TextRenderer(
                device: device,
                fontName: fontName,
                fontSize: fontSize,
                maxCharacters: 50000,  // Increase buffer capacity significantly
                coordinateOrigin: .topLeft
            )
            
            // Pre-render ASCII characters
            textRenderer?.fontManager.prerenderASCII()
            
            // Update statistics
            updateStatistics()
        } catch {
            print("Failed to create text renderer: \(error)")
        }
    }
    
    func loadSampleText() {
        currentInputText = """
        SwiftFontAtlas Demo Text
        ========================
        
        The Quick Brown Fox Jumps Over The Lazy Dog
        THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
        0123456789 !@#$%^&*()_+-=[]{}\\|;':",./<>?
        
        Features Demonstrated:
        • Multiline text rendering
        • Automatic newline handling
        • Proper character positioning
        • Font atlas caching
        
        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        Ut enim ad minim veniam, quis nostrud exercitation.
        """
    }
    
    private func updateStatistics() {
        guard let renderer = textRenderer else { return }
        
        let manager = renderer.fontManager
        // Calculate glyph count from cache (simple estimation)
        glyphCount = currentInputText.filter { !$0.isWhitespace }.count
        
        // Calculate atlas utilization
        let atlasSize = manager.withAtlas { $0.size }
        let bytesPerPixel = manager.withAtlas { $0.format.bytesPerPixel }
        let totalPixels = Int(atlasSize) * Int(atlasSize)
        
        // For now, estimate utilization based on glyph count
        atlasUtilization = min(Float(glyphCount) / 256.0, 1.0)
        
        // Calculate memory usage
        let totalBytes = totalPixels * bytesPerPixel
        memoryUsage = ByteCountFormatter.string(fromByteCount: Int64(totalBytes), countStyle: .memory)
        
        // Update atlas texture
        let managedTexture = manager.createManagedTexture(device: device)
        atlasTexture = managedTexture.metalTexture
    }
    
    // MARK: - MTKViewDelegate
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Nothing needed here
    }
    
    func draw(in view: MTKView) {
        guard let renderer = textRenderer,
              let drawable = view.currentDrawable,
              let descriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else {
            print("❌ Failed to get Metal rendering components")
            return
        }
        
        // Set clear color
        descriptor.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 1, blue: 1, alpha: 1)
        
        // Set projection matrix
        renderer.setProjectionMatrix(using: renderEncoder, viewportSize: view.drawableSize)
        
        // No debug output
        
        // Draw text - it's this simple!
        renderer.drawText(
            currentInputText,
            at: CGPoint(x: 40, y: 80),  // More margin from left edge, higher up
            color: simd_float4(0, 0, 0, 1),  // Black text
            using: renderEncoder
        )

        renderer.drawText(
            currentInputText,
            at: CGPoint(x: 30, y: 380),  // More margin from left edge, higher up
            color: simd_float4(0, 0, 0, 1),  // Black text
            using: renderEncoder
        )
        
        // Example of wrapped text - positioned lower to avoid overlap
        let wrappedText = "This text is automatically wrapped to fit within a 300-point wide area. The TextRenderer handles all the complexity of word wrapping, line breaking, and vertex generation."
        renderer.drawWrappedText(
            wrappedText,
            in: CGRect(x: 400, y: 150, width: 300, height: 200),
            wrapMode: TextLayout.WrapMode.word,
            alignment: SwiftFontAtlas.TextRenderer.TextAlignment.left,
            color: simd_float4(0.2, 0.2, 0.8, 1),  // Blue text
            using: renderEncoder
        )
        
        // Example of centered text
        renderer.drawText(
            "Centered Text Example",
            in: CGRect(x: 400, y: 380, width: 300, height: 50),
            alignment: SwiftFontAtlas.TextRenderer.TextAlignment.center,
            color: simd_float4(0.8, 0.2, 0.2, 1),  // Red text
            using: renderEncoder
        )
        
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
        
        // Update statistics after rendering
        updateStatistics()
    }
}

// Statistics View
struct StatisticsView: View {
    @ObservedObject var viewModel: SimplifiedTextViewModel
    
    var body: some View {
        HStack(spacing: 20) {
            StatItem(label: "Glyphs", value: "\(viewModel.glyphCount)")
            StatItem(label: "Atlas Usage", value: String(format: "%.1f%%", viewModel.atlasUtilization * 100))
            StatItem(label: "Memory", value: viewModel.memoryUsage)
        }
        .padding(.vertical, 8)
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
    }
}

struct StatItem: View {
    let label: String
    let value: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Text(value)
                .font(.system(.body, design: .monospaced))
        }
    }
}

// Simplified Metal Text View
struct SimplifiedMetalTextView: NSViewRepresentable {
    @ObservedObject var viewModel: SimplifiedTextViewModel
    
    func makeNSView(context: Context) -> MTKView {
        let metalView = MTKView()
        metalView.device = viewModel.device
        metalView.delegate = viewModel
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.clearColor = MTLClearColor(red: 1, green: 1, blue: 1, alpha: 1) // White background
        metalView.isPaused = false
        metalView.enableSetNeedsDisplay = true
        
        // Store reference to view in view model for updates
        viewModel.metalView = metalView
        
        return metalView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        // Trigger redraw when text changes
        DispatchQueue.main.async {
            nsView.needsDisplay = true
        }
    }
}