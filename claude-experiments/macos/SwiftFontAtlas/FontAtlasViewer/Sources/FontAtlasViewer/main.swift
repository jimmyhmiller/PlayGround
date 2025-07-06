import SwiftUI
import SwiftFontAtlas
import AppKit

@main
struct FontAtlasViewerApp: App {
    init() {
        print("🚀 FontAtlasViewer starting...")
        print("   If you don't see a window, check your Dock or use Cmd+Tab")
    }
    
    var body: some Scene {
        WindowGroup("Font Atlas Viewer") {
            ContentView()
                .onAppear {
                    print("✅ SwiftUI ContentView appeared!")
                }
        }
        .windowResizability(.contentSize)
        .defaultSize(width: 1200, height: 800)
    }
}

struct ContentView: View {
    @StateObject private var viewModel = AtlasViewModel()
    
    var body: some View {
        HStack(spacing: 0) {
            // Left sidebar
            VStack(alignment: .leading, spacing: 20) {
                Text("Font Atlas Viewer")
                    .font(.title)
                    .fontWeight(.bold)
                
                // Font controls
                GroupBox("Font Settings") {
                    VStack(alignment: .leading, spacing: 12) {
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
                            Slider(value: $viewModel.fontSize, in: 8...72, step: 1)
                            Text("\(Int(viewModel.fontSize))pt")
                                .frame(width: 35)
                        }
                        
                        HStack {
                            Text("Atlas:")
                            Picker("", selection: $viewModel.atlasSize) {
                                Text("256×256").tag(UInt32(256))
                                Text("512×512").tag(UInt32(512))
                                Text("1024×1024").tag(UInt32(1024))
                            }
                            .pickerStyle(.segmented)
                        }
                        
                        Button("Create Atlas") {
                            viewModel.createAtlas()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(viewModel.isCreating)
                    }
                }
                
                // Actions
                GroupBox("Actions") {
                    VStack(spacing: 8) {
                        Button("Render ASCII (32-126)") {
                            viewModel.renderASCII()
                        }
                        .disabled(!viewModel.hasAtlas)
                        
                        HStack {
                            TextField("Custom text", text: $viewModel.customText)
                                .textFieldStyle(.roundedBorder)
                            Button("Render") {
                                viewModel.renderCustomText()
                            }
                            .disabled(!viewModel.hasAtlas)
                        }
                        
                        Button("Unicode Test") {
                            viewModel.renderUnicodeTest()
                        }
                        .disabled(!viewModel.hasAtlas)
                        
                        Button("Clear Atlas") {
                            viewModel.clearAtlas()
                        }
                        .disabled(!viewModel.hasAtlas)
                    }
                }
                
                // Statistics
                GroupBox("Statistics") {
                    VStack(alignment: .leading, spacing: 4) {
                        if let stats = viewModel.stats {
                            Text("Atlas: \(stats.atlasSize)×\(stats.atlasSize)")
                            Text("Glyphs: \(stats.glyphCount)")
                            Text("Memory: \(stats.memoryUsage)")
                            Text("Utilization: \(stats.utilization)%")
                            Text("Cell size: \(stats.cellSize)")
                            Text("Last op: \(stats.lastOpTime)ms")
                        } else {
                            Text("No atlas created")
                                .foregroundColor(.secondary)
                        }
                    }
                    .font(.system(.caption, design: .monospaced))
                }
                
                // Live log
                GroupBox("Log") {
                    ScrollViewReader { proxy in
                        ScrollView {
                            LazyVStack(alignment: .leading, spacing: 1) {
                                ForEach(Array(viewModel.logMessages.enumerated()), id: \.offset) { index, message in
                                    Text(message)
                                        .font(.system(.caption, design: .monospaced))
                                        .textSelection(.enabled)
                                        .id(index)
                                }
                            }
                        }
                        .frame(height: 120)
                        .onChange(of: viewModel.logMessages.count) {
                            proxy.scrollTo(viewModel.logMessages.count - 1)
                        }
                    }
                }
                
                Spacer()
            }
            .frame(width: 300)
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            // Main content area
            VStack {
                Text("Atlas Visualization")
                    .font(.title2)
                    .padding(.top)
                
                if let atlasImage = viewModel.atlasImage {
                    ZStack {
                        // Atlas image
                        ScrollView([.horizontal, .vertical]) {
                            VStack {
                                Image(nsImage: atlasImage)
                                    .interpolation(.none)
                                    .scaleEffect(viewModel.zoomLevel)
                                    .background(
                                        Rectangle()
                                            .fill(Color.black)
                                            .scaleEffect(viewModel.zoomLevel)
                                    )
                                    .onTapGesture(count: 2) {
                                        viewModel.resetZoom()
                                    }
                                
                                // Grid overlay showing cells
                                if viewModel.showGrid && viewModel.zoomLevel > 2 {
                                    GridOverlay(
                                        atlasSize: atlasImage.size,
                                        cellSize: viewModel.cellSize,
                                        zoom: viewModel.zoomLevel
                                    )
                                }
                            }
                        }
                        .background(Color.gray.opacity(0.2))
                        
                        // Zoom controls
                        VStack {
                            HStack {
                                Spacer()
                                VStack(spacing: 4) {
                                    Button("+") { viewModel.zoomIn() }
                                    Button("−") { viewModel.zoomOut() }
                                    Button("1:1") { viewModel.resetZoom() }
                                    Text("\(Int(viewModel.zoomLevel * 100))%")
                                        .font(.caption)
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                            }
                            
                            Spacer()
                            
                            HStack {
                                Toggle("Grid", isOn: $viewModel.showGrid)
                                    .toggleStyle(.checkbox)
                                    .disabled(viewModel.zoomLevel <= 2)
                                Spacer()
                            }
                        }
                        .padding()
                    }
                } else {
                    VStack(spacing: 20) {
                        Image(systemName: "textformat")
                            .font(.system(size: 64))
                            .foregroundColor(.secondary)
                        
                        VStack(spacing: 8) {
                            Text("No Atlas Created")
                                .font(.title2)
                                .foregroundColor(.secondary)
                            
                            Text("Create a font atlas to see the glyph packing visualization.\nWhite pixels show rendered glyphs on a black background.")
                                .multilineTextAlignment(.center)
                                .foregroundColor(.secondary)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
        }
        .frame(minWidth: 1000, minHeight: 700)
        .onAppear {
            print("✅ App appeared - bringing window to front")
            NSApp.activate(ignoringOtherApps: true)
            viewModel.logMessage("Font Atlas Viewer started")
        }
    }
}

struct GridOverlay: View {
    let atlasSize: NSSize
    let cellSize: CGSize
    let zoom: Double
    
    var body: some View {
        Canvas { context, size in
            let cellWidth = cellSize.width * zoom
            let cellHeight = cellSize.height * zoom
            
            context.stroke(
                Path { path in
                    // Vertical lines
                    var x: Double = 0
                    while x < size.width {
                        path.move(to: CGPoint(x: x, y: 0))
                        path.addLine(to: CGPoint(x: x, y: size.height))
                        x += cellWidth
                    }
                    
                    // Horizontal lines
                    var y: Double = 0
                    while y < size.height {
                        path.move(to: CGPoint(x: 0, y: y))
                        path.addLine(to: CGPoint(x: size.width, y: y))
                        y += cellHeight
                    }
                },
                with: .color(.blue.opacity(0.3)),
                lineWidth: 0.5
            )
        }
        .frame(
            width: atlasSize.width * zoom,
            height: atlasSize.height * zoom
        )
    }
}

@MainActor
class AtlasViewModel: ObservableObject {
    @Published var selectedFont = "SF Mono"
    @Published var fontSize: Double = 18
    @Published var atlasSize: UInt32 = 512
    @Published var customText = "Hello, SwiftFontAtlas! 🚀"
    @Published var atlasImage: NSImage?
    @Published var stats: Stats?
    @Published var zoomLevel: Double = 2.0
    @Published var showGrid = true
    @Published var hasAtlas = false
    @Published var isCreating = false
    @Published var logMessages: [String] = []
    
    private var fontManager: FontAtlasManager?
    private var glyphCount = 0
    
    let availableFonts = [
        "SF Mono", "Menlo", "Monaco", "Courier", 
        "Helvetica", "Arial", "Times", "Georgia",
        "Avenir", "Palatino"
    ]
    
    var cellSize: CGSize {
        fontManager?.cellSize ?? CGSize(width: 8, height: 16)
    }
    
    struct Stats {
        let atlasSize: UInt32
        let glyphCount: Int
        let memoryUsage: String
        let utilization: String
        let cellSize: String
        let lastOpTime: String
    }
    
    func createAtlas() {
        isCreating = true
        let fontName = selectedFont
        let fontSize = self.fontSize
        let atlasSize = self.atlasSize
        
        logMessage("Creating \(fontName) \(Int(fontSize))pt atlas (\(atlasSize)×\(atlasSize))...")
        
        Task {
            let start = CFAbsoluteTimeGetCurrent()
            
            do {
                let manager = try FontAtlasManager(
                    fontName: fontName,
                    fontSize: Float(fontSize),
                    atlasSize: atlasSize
                )
                
                let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
                
                await MainActor.run {
                    self.fontManager = manager
                    self.glyphCount = 0
                    self.hasAtlas = true
                    self.isCreating = false
                    self.updateVisualization()
                    self.updateStats(lastOpTime: time)
                    self.logMessage("✅ Atlas created in \(String(format: "%.2f", time))ms")
                }
            } catch {
                await MainActor.run {
                    self.isCreating = false
                    self.logMessage("❌ Failed to create atlas: \(error)")
                }
            }
        }
    }
    
    func renderASCII() {
        guard let manager = fontManager else { return }
        
        logMessage("Rendering ASCII characters...")
        
        Task {
            let start = CFAbsoluteTimeGetCurrent()
            let count = manager.prerenderASCII()
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            await MainActor.run {
                self.glyphCount += count
                self.updateVisualization()
                self.updateStats(lastOpTime: time)
                self.logMessage("✅ Rendered \(count) ASCII chars in \(String(format: "%.2f", time))ms")
            }
        }
    }
    
    func renderCustomText() {
        guard let manager = fontManager else { return }
        
        let text = customText
        logMessage("Rendering: '\(text)'")
        
        Task {
            let start = CFAbsoluteTimeGetCurrent()
            let count = manager.prerenderString(text)
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            await MainActor.run {
                self.glyphCount += count
                self.updateVisualization()
                self.updateStats(lastOpTime: time)
                self.logMessage("✅ Rendered \(count) chars in \(String(format: "%.2f", time))ms")
            }
        }
    }
    
    func renderUnicodeTest() {
        guard let manager = fontManager else { return }
        
        let testStrings = [
            "áéíóúàèìòùâêîôûäëïöüç", // Accented
            "αβγδεζηθικλμνξοπρστυφχψω", // Greek
            "→←↑↓⇒⇐⇑⇓⇔⇕", // Arrows
            "©®™℠°±×÷", // Symbols
            "🚀🎯📱💻⚡🔥" // Emoji
        ]
        
        logMessage("Unicode test starting...")
        
        Task {
            let start = CFAbsoluteTimeGetCurrent()
            var totalCount = 0
            for testString in testStrings {
                totalCount += manager.prerenderString(testString)
            }
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            await MainActor.run {
                self.glyphCount += totalCount
                self.updateVisualization()
                self.updateStats(lastOpTime: time)
                self.logMessage("✅ Unicode test: \(totalCount) chars in \(String(format: "%.2f", time))ms")
            }
        }
    }
    
    func clearAtlas() {
        guard let manager = fontManager else { return }
        
        logMessage("Clearing atlas...")
        let start = CFAbsoluteTimeGetCurrent()
        
        manager.withAtlas { atlas in
            atlas.clear()
        }
        
        let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
        glyphCount = 0
        updateVisualization()
        updateStats(lastOpTime: time)
        logMessage("✅ Atlas cleared in \(String(format: "%.2f", time))ms")
    }
    
    private func updateVisualization() {
        guard let manager = fontManager else {
            atlasImage = nil
            return
        }
        
        manager.withAtlas { atlas in
            atlasImage = createAtlasImage(from: atlas)
        }
    }
    
    private func createAtlasImage(from atlas: FontAtlas) -> NSImage? {
        let size = Int(atlas.size)
        
        // Create a bitmap context
        guard let context = CGContext(
            data: nil,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }
        
        // Copy atlas data
        atlas.data.withUnsafeBytes { bytes in
            if let data = context.data {
                data.copyMemory(from: bytes.baseAddress!, byteCount: atlas.data.count)
            }
        }
        
        guard let cgImage = context.makeImage() else { return nil }
        
        let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: size, height: size))
        return nsImage
    }
    
    private func updateStats(lastOpTime: Double) {
        guard let manager = fontManager else {
            stats = nil
            return
        }
        
        manager.withAtlas { atlas in
            let memoryMB = Double(atlas.data.count) / (1024 * 1024)
            
            // Calculate utilization
            var nonZeroPixels = 0
            for byte in atlas.data {
                if byte != 0 { nonZeroPixels += 1 }
            }
            let utilization = Double(nonZeroPixels) / Double(atlas.data.count) * 100
            
            stats = Stats(
                atlasSize: atlas.size,
                glyphCount: glyphCount,
                memoryUsage: String(format: "%.2f MB", memoryMB),
                utilization: String(format: "%.1f", utilization),
                cellSize: String(format: "%.0f×%.0f", manager.cellSize.width, manager.cellSize.height),
                lastOpTime: String(format: "%.2f", lastOpTime)
            )
        }
    }
    
    func logMessage(_ message: String) {
        let timestamp = DateFormatter().apply {
            $0.dateFormat = "HH:mm:ss"
        }.string(from: Date())
        
        logMessages.append("[\(timestamp)] \(message)")
        
        // Keep last 20 messages
        if logMessages.count > 20 {
            logMessages.removeFirst()
        }
    }
    
    func zoomIn() {
        zoomLevel = min(zoomLevel * 1.5, 8.0)
    }
    
    func zoomOut() {
        zoomLevel = max(zoomLevel / 1.5, 0.5)
    }
    
    func resetZoom() {
        zoomLevel = 2.0
    }
}

extension DateFormatter {
    func apply(_ closure: (DateFormatter) -> Void) -> DateFormatter {
        closure(self)
        return self
    }
}