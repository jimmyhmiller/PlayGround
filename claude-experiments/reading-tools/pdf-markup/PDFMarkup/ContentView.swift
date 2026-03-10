import SwiftUI
import PencilKit
import PDFKit

#if os(macOS)
import AppKit
#else
import UIKit
#endif

struct ContentView: View {
    @State private var selectedColor: Color = .yellow
    @State private var pdfDocument: PDFDocument?
    @State private var selectedPDF: PDFMetadata?
    @State private var library: PDFLibrary?
    @State private var isLoadingLibrary = false
    @State private var errorMessage: String?
    @State private var currentPDFHash: String?
    @State private var currentDownloadTask: Task<Void, Never>?
    @State private var loadRequestCounter: Int = 0

    @StateObject private var downloader = PDFDownloader()
    @EnvironmentObject var sharedPDFManager: SharedPDFManager

    var body: some View {
        NavigationSplitView {
            // Sidebar
            if let library = library {
                PDFLibrarySidebar(library: library, selectedPDF: $selectedPDF, onSelectSharedPDF: { hash, document in
                    currentDownloadTask?.cancel()
                    currentDownloadTask = nil
                    loadRequestCounter += 1
                    currentPDFHash = hash
                    pdfDocument = document
                    selectedPDF = nil
                })
            } else if isLoadingLibrary {
                VStack {
                    ProgressView()
                    Text("Loading PDF library...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            } else {
                VStack {
                    Image(systemName: "book.closed")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)
                    Text("PDF library not loaded")
                        .font(.headline)
                    if let error = errorMessage {
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                            .padding()
                    }
                }
            }
        } detail: {
            ZStack {
                if let pdfDocument = pdfDocument, let pdfHash = currentPDFHash ?? selectedPDF?.hash {
                    PDFMarkupView(pdfDocument: pdfDocument, pdfHash: pdfHash, selectedColor: $selectedColor)
                } else {
                    VStack(spacing: 20) {
                        Image(systemName: "doc.fill")
                            .font(.system(size: 64))
                            .foregroundStyle(.tertiary)
                        Text("Select a PDF from the sidebar")
                            .font(.title3)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color.systemGroupedBackground)
                }

                if pdfDocument != nil {
                    HStack {
                        Spacer()
                        FloatingColorPalette(selectedColor: $selectedColor)
                            .padding(.trailing, 16)
                    }
                }

                if downloader.isDownloading {
                    VStack {
                        HStack {
                            Spacer()
                            HStack(spacing: 8) {
                                ProgressView()
                                    .scaleEffect(0.7)
                                Text("Downloading...")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(.ultraThinMaterial)
                            .cornerRadius(20)
                            .padding()
                        }
                        Spacer()
                    }
                }
            }
        }
        .task {
            await loadLibrary()
        }
        .onChange(of: selectedPDF) { oldValue, newValue in
            print("selectedPDF changed: \(newValue?.hash ?? "nil")")
            currentDownloadTask?.cancel()
            loadRequestCounter += 1

            if let pdf = newValue {
                print("Will load PDF: \(pdf.displayTitle)")
                let requestId = loadRequestCounter
                currentPDFHash = pdf.hash
                currentDownloadTask = Task {
                    await loadPDF(metadata: pdf, requestId: requestId)
                }
            }
        }
        .onChange(of: sharedPDFManager.pendingPDF) { oldValue, newValue in
            if let shared = newValue {
                currentDownloadTask?.cancel()
                currentDownloadTask = nil
                loadRequestCounter += 1
                currentPDFHash = shared.hash
                pdfDocument = shared.document
                selectedPDF = nil
                sharedPDFManager.clearPending()
            }
        }
    }

    func loadLibrary() async {
        isLoadingLibrary = true
        defer { isLoadingLibrary = false }

        await Task.detached {
            S3StateManager.shared.loadState()

            do {
                guard AWSCredentials.isConfigured() else {
                    await MainActor.run {
                        self.errorMessage = "AWS credentials not configured"
                    }
                    return
                }

                guard let data = try await DrawingSyncManager.shared.signedDownloadData(key: "pdf-index.json") else {
                    await MainActor.run {
                        self.errorMessage = "pdf-index.json not found on S3"
                    }
                    return
                }
                let pdfs = try JSONDecoder().decode([PDFMetadata].self, from: data)

                let s3PDFs = pdfs.filter { pdf in
                    S3StateManager.shared.s3Key(for: pdf.hash) != nil
                }

                await MainActor.run {
                    self.library = PDFLibrary(pdfs: s3PDFs)
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "Failed to load PDF library: \(error.localizedDescription)"
                    print("Error loading library: \(error)")
                }
            }
        }.value
    }

    func loadPDF(metadata: PDFMetadata, requestId: Int) async {
        do {
            print("Loading PDF: \(metadata.hash)")

            let document = try await downloader.downloadPDF(metadata: metadata)

            guard !Task.isCancelled else { return }
            guard requestId == loadRequestCounter else {
                print("Discarding stale PDF load (request \(requestId), current \(loadRequestCounter))")
                return
            }

            pdfDocument = document

            // Sync drawings in the background after PDF is displayed
            Task {
                do {
                    try await DrawingSyncManager.shared.sync(pdfHash: metadata.hash)
                    print("Sync completed for \(metadata.hash)")
                } catch {
                    print("Sync failed: \(error)")
                }
            }
        } catch {
            if !Task.isCancelled {
                errorMessage = "Failed to load PDF: \(error.localizedDescription)"
                print("Error loading PDF: \(error)")
            }
        }
    }
}

// MARK: - Floating Color Palette

struct FloatingColorPalette: View {
    @Binding var selectedColor: Color
    @StateObject private var customColors = CustomColorStore()
    @State private var showingColorPicker = false
    @State private var newColor = Color.cyan

    private let defaultColors: [(Color, String)] = [
        (Color(red: 1.0, green: 0.95, blue: 0.4), "yellow"),
        (Color(red: 0.6, green: 0.95, blue: 0.6), "green"),
        (Color(red: 1.0, green: 0.45, blue: 0.45), "red"),
        (Color(red: 1.0, green: 0.75, blue: 0.4), "orange"),
        (Color(red: 0.6, green: 0.8, blue: 1.0), "blue"),
        (Color(red: 0.85, green: 0.7, blue: 1.0), "purple"),
    ]

    var body: some View {
        VStack(spacing: 6) {
            EraserButton(isSelected: selectedColor == .clear) {
                withAnimation(.easeInOut(duration: 0.15)) {
                    selectedColor = .clear
                }
            }

            Divider()
                .frame(width: 20)
                .padding(.vertical, 2)

            ForEach(defaultColors, id: \.1) { color, name in
                ColorDot(
                    color: color,
                    isSelected: isDefaultColorSelected(name: name),
                    action: { selectDefaultColor(name: name) }
                )
            }

            ForEach(Array(customColors.colors.enumerated()), id: \.offset) { index, colorData in
                ColorDot(
                    color: colorData.color,
                    isSelected: isCustomColorSelected(colorData),
                    action: { selectCustomColor(colorData) }
                )
                .contextMenu {
                    Button(role: .destructive) {
                        customColors.remove(at: index)
                    } label: {
                        Label("Remove", systemImage: "trash")
                    }
                }
            }

            if customColors.colors.count < 6 {
                Divider()
                    .frame(width: 20)
                    .padding(.vertical, 2)

                AddColorButton {
                    showingColorPicker = true
                }
            }
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 6)
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(.ultraThinMaterial)
                .shadow(color: Color.black.opacity(0.15), radius: 8, x: 0, y: 2)
        )
        .sheet(isPresented: $showingColorPicker) {
            ColorPickerSheet(color: $newColor) {
                customColors.add(newColor)
                selectedColor = newColor
                showingColorPicker = false
            }
        }
    }

    private func isDefaultColorSelected(name: String) -> Bool {
        switch name {
        case "yellow": return selectedColor == .yellow
        case "green": return selectedColor == .green
        case "red": return selectedColor == .red
        case "orange": return selectedColor == .orange
        case "blue": return selectedColor == .blue
        case "purple": return selectedColor == .purple
        default: return false
        }
    }

    private func selectDefaultColor(name: String) {
        withAnimation(.easeInOut(duration: 0.15)) {
            switch name {
            case "yellow": selectedColor = .yellow
            case "green": selectedColor = .green
            case "red": selectedColor = .red
            case "orange": selectedColor = .orange
            case "blue": selectedColor = .blue
            case "purple": selectedColor = .purple
            default: break
            }
        }
    }

    private func isCustomColorSelected(_ colorData: StorableColor) -> Bool {
        return colorData.matches(selectedColor)
    }

    private func selectCustomColor(_ colorData: StorableColor) {
        withAnimation(.easeInOut(duration: 0.15)) {
            selectedColor = colorData.color
        }
    }
}

// MARK: - Custom Color Storage

struct StorableColor: Codable, Equatable {
    let red: Double
    let green: Double
    let blue: Double
    let alpha: Double

    init(_ color: Color) {
        let components = color.rgbaComponents()
        self.red = components.red
        self.green = components.green
        self.blue = components.blue
        self.alpha = components.alpha
    }

    var color: Color {
        Color(red: red, green: green, blue: blue, opacity: alpha)
    }

    func matches(_ other: Color) -> Bool {
        let otherStorable = StorableColor(other)
        return abs(red - otherStorable.red) < 0.01 &&
               abs(green - otherStorable.green) < 0.01 &&
               abs(blue - otherStorable.blue) < 0.01
    }
}

class CustomColorStore: ObservableObject {
    @Published var colors: [StorableColor] = []

    private let key = "customHighlightColors"

    init() {
        load()
    }

    func add(_ color: Color) {
        let storable = StorableColor(color)
        if !colors.contains(storable) {
            colors.append(storable)
            save()
        }
    }

    func remove(at index: Int) {
        colors.remove(at: index)
        save()
    }

    private func save() {
        if let data = try? JSONEncoder().encode(colors) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }

    private func load() {
        if let data = UserDefaults.standard.data(forKey: key),
           let decoded = try? JSONDecoder().decode([StorableColor].self, from: data) {
            colors = decoded
        }
    }
}

// MARK: - Color Picker Sheet

struct ColorPickerSheet: View {
    @Binding var color: Color
    @Environment(\.dismiss) var dismiss
    let onAdd: () -> Void

    var body: some View {
        #if os(macOS)
        VStack(spacing: 24) {
            ColorPicker("Choose a highlight color", selection: $color, supportsOpacity: false)
                .labelsHidden()

            RoundedRectangle(cornerRadius: 12)
                .fill(color.opacity(0.5))
                .frame(height: 60)
                .overlay(
                    Text("Preview")
                        .foregroundColor(.primary)
                )
                .padding(.horizontal)

            HStack {
                Button("Cancel") { dismiss() }
                Spacer()
                Button("Add") { onAdd() }
            }
            .padding(.horizontal)
        }
        .padding(24)
        .frame(width: 300, height: 200)
        #else
        NavigationView {
            VStack(spacing: 24) {
                ColorPicker("Choose a highlight color", selection: $color, supportsOpacity: false)
                    .labelsHidden()
                    .scaleEffect(1.5)

                RoundedRectangle(cornerRadius: 12)
                    .fill(color.opacity(0.5))
                    .frame(height: 60)
                    .overlay(
                        Text("Preview")
                            .foregroundColor(.primary)
                    )
                    .padding(.horizontal)

                Spacer()
            }
            .padding(.top, 40)
            .navigationTitle("Add Color")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Add") { onAdd() }
                }
            }
        }
        #endif
    }
}

// MARK: - Add Color Button

struct AddColorButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(Color.systemGray5)
                    .frame(width: 28, height: 28)
                Image(systemName: "plus")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.secondary)
            }
            .overlay(
                Circle()
                    .strokeBorder(Color.black.opacity(0.1), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}

struct EraserButton: View {
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(Color.systemGray5)
                    .frame(width: 28, height: 28)
                Image(systemName: "eraser.fill")
                    .font(.system(size: 14))
                    .foregroundColor(.secondary)
            }
            .overlay(
                Circle()
                    .strokeBorder(Color.white, lineWidth: isSelected ? 3 : 0)
                    .shadow(color: .black.opacity(0.2), radius: 1, x: 0, y: 1)
            )
            .overlay(
                Circle()
                    .strokeBorder(Color.black.opacity(0.1), lineWidth: 1)
            )
            .scaleEffect(isSelected ? 1.1 : 1.0)
        }
        .buttonStyle(.plain)
    }
}

struct ColorDot: View {
    let color: Color
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Circle()
                .fill(color)
                .frame(width: 28, height: 28)
                .overlay(
                    Circle()
                        .strokeBorder(Color.white, lineWidth: isSelected ? 3 : 0)
                        .shadow(color: .black.opacity(0.2), radius: 1, x: 0, y: 1)
                )
                .overlay(
                    Circle()
                        .strokeBorder(Color.black.opacity(0.1), lineWidth: 1)
                )
                .scaleEffect(isSelected ? 1.1 : 1.0)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - PDF Markup View

struct PDFMarkupView: View {
    let pdfDocument: PDFDocument
    let pdfHash: String
    @Binding var selectedColor: Color

    var body: some View {
        NativePDFView(pdfDocument: pdfDocument, pdfHash: pdfHash, selectedColor: $selectedColor)
            .id(pdfHash)
    }
}

// MARK: - PDF Page View (Platform-specific)

#if os(macOS)

/// Drawing overlay that lives in PDF page coordinate space.
/// Placed via PDFPageOverlayViewProvider so it scales with PDFView zoom automatically.
class DrawingCanvasView: NSView {
    var drawing = PKDrawing()
    var currentPoints: [PKStrokePoint] = []
    var currentColor: NSColor = NSColor(red: 1, green: 1, blue: 0, alpha: 0.15)
    var isErasing = false
    var onDrawingChanged: (() -> Void)?
    var pdfHash: String = ""
    var pageIndex: Int = 0

    // Undo/redo stacks store snapshots of strokes
    private var undoStack: [[PKStroke]] = []
    private var redoStack: [[PKStroke]] = []
    // Track pre-erase state for batching erase drag into one undo step
    private var preEraseStrokes: [PKStroke]?

    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        let image = drawing.image(from: bounds, scale: NSScreen.main?.backingScaleFactor ?? 2.0)
        if !drawing.strokes.isEmpty {
            Swift.print("[Canvas] draw page=\(pageIndex) bounds=\(bounds) drawingBounds=\(drawing.bounds) strokes=\(drawing.strokes.count) imageSize=\(image.size)")
        }
        image.draw(in: bounds)

        if currentPoints.count >= 2 {
            let ink = PKInk(.marker, color: currentColor)
            let strokePath = PKStrokePath(controlPoints: currentPoints, creationDate: Date())
            let stroke = PKStroke(ink: ink, path: strokePath)
            var preview = PKDrawing()
            preview.strokes = [stroke]
            let previewImage = preview.image(from: bounds, scale: NSScreen.main?.backingScaleFactor ?? 2.0)
            previewImage.draw(in: bounds)
        }
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        let loc = convert(event.locationInWindow, from: nil)
        if isErasing {
            preEraseStrokes = drawing.strokes
            eraseAt(loc)
            return
        }
        currentPoints = [makePoint(at: loc)]
        needsDisplay = true
    }

    override func mouseDragged(with event: NSEvent) {
        let loc = convert(event.locationInWindow, from: nil)
        if isErasing {
            eraseAt(loc)
            return
        }
        currentPoints.append(makePoint(at: loc))
        needsDisplay = true
    }

    override func mouseUp(with event: NSEvent) {
        if isErasing {
            // Push one undo step for the entire erase drag
            if let pre = preEraseStrokes {
                undoStack.append(pre)
                redoStack.removeAll()
                onDrawingChanged?()
            }
            preEraseStrokes = nil
            return
        }
        guard currentPoints.count >= 2 else {
            currentPoints = []
            return
        }

        undoStack.append(drawing.strokes)
        redoStack.removeAll()

        let ink = PKInk(.marker, color: currentColor)
        let path = PKStrokePath(controlPoints: currentPoints, creationDate: Date())
        let stroke = PKStroke(ink: ink, path: path)
        drawing.strokes.append(stroke)
        currentPoints = []
        needsDisplay = true
        onDrawingChanged?()
    }

    override func keyDown(with event: NSEvent) {
        if event.modifierFlags.contains(.command) && event.charactersIgnoringModifiers?.lowercased() == "z" {
            if event.modifierFlags.contains(.shift) {
                redo()
            } else {
                undo()
            }
            return
        }
        super.keyDown(with: event)
    }

    func undo() {
        guard let previous = undoStack.popLast() else { return }
        redoStack.append(drawing.strokes)
        drawing.strokes = previous
        needsDisplay = true
        onDrawingChanged?()
    }

    func redo() {
        guard let next = redoStack.popLast() else { return }
        undoStack.append(drawing.strokes)
        drawing.strokes = next
        needsDisplay = true
        onDrawingChanged?()
    }

    private func makePoint(at location: CGPoint) -> PKStrokePoint {
        PKStrokePoint(location: location, timeOffset: 0, size: CGSize(width: 20, height: 20),
                      opacity: 1.0, force: 1.0, azimuth: 0, altitude: .pi / 2)
    }

    private func eraseAt(_ point: CGPoint) {
        let eraseRadius: CGFloat = 10
        let eraseRect = CGRect(x: point.x - eraseRadius, y: point.y - eraseRadius,
                               width: eraseRadius * 2, height: eraseRadius * 2)

        var newStrokes: [PKStroke] = []
        for stroke in drawing.strokes {
            let allPoints = stroke.path.interpolatedPoints(by: .distance(3)).map { $0 }
            // Find which points are hit by the eraser
            let hitIndices = Set(allPoints.enumerated().compactMap { (i, pt) in
                eraseRect.contains(pt.location) ? i : nil
            })

            if hitIndices.isEmpty {
                // Stroke not touched, keep as-is
                newStrokes.append(stroke)
                continue
            }

            // Split into runs of non-erased points
            var runStart: Int? = nil
            for i in 0...allPoints.count {
                let isHit = (i == allPoints.count) || hitIndices.contains(i)
                if isHit {
                    if let start = runStart, i - start >= 2 {
                        let runPoints = Array(allPoints[start..<i])
                        let controlPoints = runPoints.map { makePoint(at: $0.location) }
                        let newPath = PKStrokePath(controlPoints: controlPoints, creationDate: Date())
                        let newStroke = PKStroke(ink: stroke.ink, path: newPath)
                        newStrokes.append(newStroke)
                    }
                    runStart = nil
                } else if runStart == nil {
                    runStart = i
                }
            }
        }

        drawing.strokes = newStrokes
        needsDisplay = true
    }

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func scrollWheel(with event: NSEvent) {
        nextResponder?.scrollWheel(with: event)
    }

    override func magnify(with event: NSEvent) {
        nextResponder?.magnify(with: event)
    }

    override func smartMagnify(with event: NSEvent) {
        nextResponder?.smartMagnify(with: event)
    }

    func saveDrawing() {
        let mediaBoxSize = bounds.size
        guard mediaBoxSize.width > 0 else { return }
        // Don't save empty drawings — this prevents blank canvases from overwriting
        // real drawings that were synced from another device
        guard !drawing.strokes.isEmpty else { return }
        // The overlay view is sized to the page's mediaBox, so drawing coords ARE mediaBox coords
        DrawingManager.shared.scheduleSave(drawing, pdfHash: pdfHash, page: pageIndex)
    }

    func loadDrawing() {
        undoStack.removeAll()
        redoStack.removeAll()
        Swift.print("[Canvas] loadDrawing page=\(pageIndex) hash=\(pdfHash)")
        guard let loaded = DrawingManager.shared.loadDrawing(pdfHash: pdfHash, page: pageIndex) else {
            Swift.print("[Canvas] No drawing found for page \(pageIndex)")
            drawing = PKDrawing()
            needsDisplay = true
            return
        }
        Swift.print("[Canvas] Loaded drawing for page \(pageIndex): \(loaded.strokes.count) strokes, drawingBounds=\(loaded.bounds), canvasBounds=\(bounds)")
        drawing = loaded
        needsDisplay = true
    }
}

/// PDFView subclass that routes mouse events to drawing overlays when active
class MarkupPDFView: PDFView {
    weak var overlayProvider: DrawingOverlayProvider?

    private func canvasUnderEvent(_ event: NSEvent) -> DrawingCanvasView? {
        guard let provider = overlayProvider else {
            Swift.print("[MarkupPDFView] No overlay provider")
            return nil
        }
        Swift.print("[MarkupPDFView] overlays count: \(provider.overlays.count)")
        let locationInView = convert(event.locationInWindow, from: nil)
        Swift.print("[MarkupPDFView] locationInView: \(locationInView)")
        guard let page = page(for: locationInView, nearest: false) else {
            Swift.print("[MarkupPDFView] No page found at location")
            // Try nearest
            if let nearestPage = page(for: locationInView, nearest: true) {
                Swift.print("[MarkupPDFView] Nearest page exists: \(nearestPage), has overlay: \(provider.overlays[nearestPage] != nil)")
            }
            return nil
        }
        Swift.print("[MarkupPDFView] Found page: \(page), has overlay: \(provider.overlays[page] != nil)")
        return provider.overlays[page]
    }

    override func mouseDown(with event: NSEvent) {
        Swift.print("[MarkupPDFView] mouseDown called")
        if let canvas = canvasUnderEvent(event) {
            Swift.print("[MarkupPDFView] Routing to canvas")
            canvas.mouseDown(with: event)
            return
        }
        Swift.print("[MarkupPDFView] Falling through to super")
        super.mouseDown(with: event)
    }

    override func mouseDragged(with event: NSEvent) {
        if let canvas = canvasUnderEvent(event) {
            canvas.mouseDragged(with: event)
            return
        }
        super.mouseDragged(with: event)
    }

    override func mouseUp(with event: NSEvent) {
        if let canvas = canvasUnderEvent(event) {
            canvas.mouseUp(with: event)
            return
        }
        super.mouseUp(with: event)
    }
}

/// Provides drawing overlay views for each PDF page
class DrawingOverlayProvider: NSObject, PDFPageOverlayViewProvider {
    var pdfHash: String = ""
    var currentColor: NSColor = NSColor(red: 1, green: 1, blue: 0, alpha: 0.15)
    var isErasing = false
    /// Keeps strong references to overlays so they aren't deallocated
    var overlays: [PDFPage: DrawingCanvasView] = [:]

    func pdfView(_ view: PDFView, overlayViewFor page: PDFPage) -> NSView? {
        Swift.print("[OverlayProvider] overlayViewFor called, page: \(page)")
        if let existing = overlays[page] {
            return existing
        }
        let canvas = DrawingCanvasView()
        let pageIndex = view.document?.index(for: page) ?? 0
        canvas.pdfHash = pdfHash
        canvas.pageIndex = pageIndex
        canvas.currentColor = currentColor
        canvas.isErasing = isErasing
        canvas.onDrawingChanged = { [weak canvas] in
            canvas?.saveDrawing()
        }
        canvas.loadDrawing()
        overlays[page] = canvas
        return canvas
    }

    func pdfView(_ view: PDFView, willDisplayOverlayView overlayView: NSView, for page: PDFPage) {
        if let canvas = overlayView as? DrawingCanvasView {
            canvas.currentColor = currentColor
            canvas.isErasing = isErasing
        }
    }

    func updateAllCanvases() {
        for (_, canvas) in overlays {
            canvas.currentColor = currentColor
            canvas.isErasing = isErasing
        }
    }

    func reloadAllDrawings(pdfHash: String) {
        self.pdfHash = pdfHash
        for (page, canvas) in overlays {
            // page index might be needed
            if let doc = canvas.superview?.superview as? PDFView {
                canvas.pageIndex = doc.document?.index(for: page) ?? canvas.pageIndex
            }
            canvas.pdfHash = pdfHash
            canvas.loadDrawing()
        }
    }
}

struct NativePDFView: NSViewRepresentable {
    let pdfDocument: PDFDocument
    let pdfHash: String
    @Binding var selectedColor: Color

    func makeCoordinator() -> Coordinator {
        Coordinator(pdfHash: pdfHash)
    }

    func makeNSView(context: Context) -> MarkupPDFView {
        let pdfView = MarkupPDFView()
        pdfView.autoScales = true
        pdfView.displayMode = .singlePageContinuous
        pdfView.displayDirection = .vertical
        pdfView.backgroundColor = NSColor.windowBackgroundColor

        let provider = context.coordinator.overlayProvider
        provider.pdfHash = pdfHash
        pdfView.isInMarkupMode = true
        pdfView.pageOverlayViewProvider = provider
        pdfView.overlayProvider = provider

        // Set document AFTER provider so PDFKit calls overlayViewFor during layout
        pdfView.document = pdfDocument
        print("[NativePDFView] makeNSView done, provider set: \(pdfView.pageOverlayViewProvider != nil)")

        context.coordinator.pdfView = pdfView

        return pdfView
    }

    func updateNSView(_ pdfView: MarkupPDFView, context: Context) {
        if pdfView.document !== pdfDocument {
            context.coordinator.overlayProvider.overlays.removeAll()
            pdfView.document = pdfDocument
        }

        let provider = context.coordinator.overlayProvider
        if provider.pdfHash != pdfHash {
            provider.reloadAllDrawings(pdfHash: pdfHash)
        }

        if selectedColor == .clear {
            provider.isErasing = true
        } else {
            provider.isErasing = false
            provider.currentColor = PlatformColor.highlightColor(from: selectedColor)
        }
        provider.updateAllCanvases()
    }

    class Coordinator: NSObject {
        let overlayProvider = DrawingOverlayProvider()
        weak var pdfView: MarkupPDFView?
        var pdfHash: String

        init(pdfHash: String) {
            self.pdfHash = pdfHash
            super.init()
        }
    }
}

#else

/// Drawing overlay that lives in PDF page coordinate space (iOS).
/// Uses PKCanvasView for native Apple Pencil support.
class DrawingCanvasUIView: UIView {
    var drawing = PKDrawing()
    var currentPoints: [PKStrokePoint] = []
    var currentColor: UIColor = UIColor(red: 1, green: 1, blue: 0, alpha: 0.15)
    var isErasing = false
    var onDrawingChanged: (() -> Void)?
    var pdfHash: String = ""
    var pageIndex: Int = 0

    private var undoStack: [[PKStroke]] = []
    private var redoStack: [[PKStroke]] = []
    private var preEraseStrokes: [PKStroke]?

    override func draw(_ rect: CGRect) {
        super.draw(rect)
        let scale = UIScreen.main.scale
        let image = drawing.image(from: bounds, scale: scale)
        image.draw(in: bounds)

        if currentPoints.count >= 2 {
            let ink = PKInk(.marker, color: currentColor)
            let strokePath = PKStrokePath(controlPoints: currentPoints, creationDate: Date())
            let stroke = PKStroke(ink: ink, path: strokePath)
            var preview = PKDrawing()
            preview.strokes = [stroke]
            let previewImage = preview.image(from: bounds, scale: scale)
            previewImage.draw(in: bounds)
        }
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        let loc = touch.location(in: self)
        if isErasing {
            preEraseStrokes = drawing.strokes
            eraseAt(loc)
            return
        }
        currentPoints = [makePoint(at: loc)]
        setNeedsDisplay()
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        let loc = touch.location(in: self)
        if isErasing {
            eraseAt(loc)
            return
        }
        currentPoints.append(makePoint(at: loc))
        setNeedsDisplay()
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        if isErasing {
            if let pre = preEraseStrokes {
                undoStack.append(pre)
                redoStack.removeAll()
                onDrawingChanged?()
            }
            preEraseStrokes = nil
            return
        }
        guard currentPoints.count >= 2 else {
            currentPoints = []
            return
        }

        undoStack.append(drawing.strokes)
        redoStack.removeAll()

        let ink = PKInk(.marker, color: currentColor)
        let path = PKStrokePath(controlPoints: currentPoints, creationDate: Date())
        let stroke = PKStroke(ink: ink, path: path)
        drawing.strokes.append(stroke)
        currentPoints = []
        setNeedsDisplay()
        onDrawingChanged?()
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        currentPoints = []
        preEraseStrokes = nil
        setNeedsDisplay()
    }

    func undo() {
        guard let previous = undoStack.popLast() else { return }
        redoStack.append(drawing.strokes)
        drawing.strokes = previous
        setNeedsDisplay()
        onDrawingChanged?()
    }

    func redo() {
        guard let next = redoStack.popLast() else { return }
        undoStack.append(drawing.strokes)
        drawing.strokes = next
        setNeedsDisplay()
        onDrawingChanged?()
    }

    private func makePoint(at location: CGPoint) -> PKStrokePoint {
        PKStrokePoint(location: location, timeOffset: 0, size: CGSize(width: 20, height: 20),
                      opacity: 1.0, force: 1.0, azimuth: 0, altitude: .pi / 2)
    }

    private func eraseAt(_ point: CGPoint) {
        let eraseRadius: CGFloat = 10
        let eraseRect = CGRect(x: point.x - eraseRadius, y: point.y - eraseRadius,
                               width: eraseRadius * 2, height: eraseRadius * 2)

        var newStrokes: [PKStroke] = []
        for stroke in drawing.strokes {
            let allPoints = stroke.path.interpolatedPoints(by: .distance(3)).map { $0 }
            let hitIndices = Set(allPoints.enumerated().compactMap { (i, pt) in
                eraseRect.contains(pt.location) ? i : nil
            })

            if hitIndices.isEmpty {
                newStrokes.append(stroke)
                continue
            }

            var runStart: Int? = nil
            for i in 0...allPoints.count {
                let isHit = (i == allPoints.count) || hitIndices.contains(i)
                if isHit {
                    if let start = runStart, i - start >= 2 {
                        let runPoints = Array(allPoints[start..<i])
                        let controlPoints = runPoints.map { makePoint(at: $0.location) }
                        let newPath = PKStrokePath(controlPoints: controlPoints, creationDate: Date())
                        let newStroke = PKStroke(ink: stroke.ink, path: newPath)
                        newStrokes.append(newStroke)
                    }
                    runStart = nil
                } else if runStart == nil {
                    runStart = i
                }
            }
        }

        drawing.strokes = newStrokes
        setNeedsDisplay()
    }

    func saveDrawing() {
        let mediaBoxSize = bounds.size
        guard mediaBoxSize.width > 0 else { return }
        // Don't save empty drawings — this prevents blank canvases from overwriting
        // real drawings that were synced from another device
        guard !drawing.strokes.isEmpty else { return }
        DrawingManager.shared.scheduleSave(drawing, pdfHash: pdfHash, page: pageIndex)
    }

    func loadDrawing() {
        undoStack.removeAll()
        redoStack.removeAll()
        guard let loaded = DrawingManager.shared.loadDrawing(pdfHash: pdfHash, page: pageIndex) else {
            drawing = PKDrawing()
            setNeedsDisplay()
            return
        }
        drawing = loaded
        setNeedsDisplay()
    }
}

/// Gesture recognizer that fails for pencil touches, used to prevent PDFView's
/// scroll view from capturing pencil input (so it goes to the drawing overlay instead).
class BlockPencilGestureRecognizer: UIGestureRecognizer {
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent) {
        for touch in touches {
            if touch.type == .pencil {
                // Do nothing — just having this recognizer on the scroll view
                // with cancelsTouchesInView won't help. We need the delegate approach below.
            }
        }
    }
}

/// PDFView subclass that prevents its internal scroll view from capturing pencil touches
class MarkupPDFView: PDFView {
    weak var overlayProvider: DrawingOverlayProviderIOS?

    override func didMoveToWindow() {
        super.didMoveToWindow()
        disablePencilScrolling()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        disablePencilScrolling()
    }

    private var didConfigureScrollView = false

    private func disablePencilScrolling() {
        guard !didConfigureScrollView else { return }
        guard let scrollView = findScrollView() else { return }
        didConfigureScrollView = true

        // Make all existing pan/pinch gesture recognizers on the scroll view
        // only respond to direct (finger) touches, not pencil
        for gestureRecognizer in scrollView.gestureRecognizers ?? [] {
            if let panGR = gestureRecognizer as? UIPanGestureRecognizer {
                panGR.allowedTouchTypes = [NSNumber(value: UITouch.TouchType.direct.rawValue)]
            }
        }
    }

    private func findScrollView() -> UIScrollView? {
        func find(in view: UIView) -> UIScrollView? {
            for subview in view.subviews {
                if let sv = subview as? UIScrollView { return sv }
                if let found = find(in: subview) { return found }
            }
            return nil
        }
        return find(in: self)
    }
}

/// Provides drawing overlay views for each PDF page (iOS)
class DrawingOverlayProviderIOS: NSObject, PDFPageOverlayViewProvider {
    var pdfHash: String = ""
    var currentColor: UIColor = UIColor(red: 1, green: 1, blue: 0, alpha: 0.15)
    var isErasing = false
    var overlays: [PDFPage: DrawingCanvasUIView] = [:]

    func pdfView(_ view: PDFView, overlayViewFor page: PDFPage) -> UIView? {
        if let existing = overlays[page] {
            return existing
        }
        let canvas = DrawingCanvasUIView()
        canvas.isOpaque = false
        canvas.backgroundColor = .clear
        let pageIndex = view.document?.index(for: page) ?? 0
        canvas.pdfHash = pdfHash
        canvas.pageIndex = pageIndex
        canvas.currentColor = currentColor
        canvas.isErasing = isErasing
        canvas.onDrawingChanged = { [weak canvas] in
            canvas?.saveDrawing()
        }
        canvas.loadDrawing()
        overlays[page] = canvas
        return canvas
    }

    func pdfView(_ view: PDFView, willDisplayOverlayView overlayView: UIView, for page: PDFPage) {
        if let canvas = overlayView as? DrawingCanvasUIView {
            canvas.currentColor = currentColor
            canvas.isErasing = isErasing
        }
    }

    func updateAllCanvases() {
        for (_, canvas) in overlays {
            canvas.currentColor = currentColor
            canvas.isErasing = isErasing
        }
    }

    func reloadAllDrawings(pdfHash: String) {
        self.pdfHash = pdfHash
        for (page, canvas) in overlays {
            if let doc = canvas.superview?.superview as? PDFView {
                canvas.pageIndex = doc.document?.index(for: page) ?? canvas.pageIndex
            }
            canvas.pdfHash = pdfHash
            canvas.loadDrawing()
        }
    }
}

struct NativePDFView: UIViewRepresentable {
    let pdfDocument: PDFDocument
    let pdfHash: String
    @Binding var selectedColor: Color

    func makeCoordinator() -> Coordinator {
        Coordinator(pdfHash: pdfHash)
    }

    func makeUIView(context: Context) -> MarkupPDFView {
        let pdfView = MarkupPDFView()
        pdfView.autoScales = true
        pdfView.displayMode = .singlePageContinuous
        pdfView.displayDirection = .vertical
        pdfView.backgroundColor = UIColor.systemBackground

        let provider = context.coordinator.overlayProvider
        provider.pdfHash = pdfHash
        pdfView.isInMarkupMode = true
        pdfView.pageOverlayViewProvider = provider
        pdfView.overlayProvider = provider

        pdfView.document = pdfDocument

        context.coordinator.pdfView = pdfView

        return pdfView
    }

    func updateUIView(_ pdfView: MarkupPDFView, context: Context) {
        if pdfView.document !== pdfDocument {
            context.coordinator.overlayProvider.overlays.removeAll()
            pdfView.document = pdfDocument
        }

        let provider = context.coordinator.overlayProvider
        if provider.pdfHash != pdfHash {
            provider.reloadAllDrawings(pdfHash: pdfHash)
        }

        if selectedColor == .clear {
            provider.isErasing = true
        } else {
            provider.isErasing = false
            provider.currentColor = PlatformColor.highlightColor(from: selectedColor)
        }
        provider.updateAllCanvases()
    }

    class Coordinator: NSObject {
        let overlayProvider = DrawingOverlayProviderIOS()
        weak var pdfView: MarkupPDFView?
        var pdfHash: String

        init(pdfHash: String) {
            self.pdfHash = pdfHash
            super.init()
        }
    }
}

#endif
