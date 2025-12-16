import SwiftUI
import PencilKit
import PDFKit

struct ContentView: View {
    @State private var selectedColor: Color = .yellow
    @State private var pdfDocument: PDFDocument?
    @State private var selectedPDF: PDFMetadata?
    @State private var library: PDFLibrary?
    @State private var isLoadingLibrary = false
    @State private var errorMessage: String?

    @StateObject private var downloader = PDFDownloader()

    var body: some View {
        NavigationSplitView {
            // Sidebar
            if let library = library {
                PDFLibrarySidebar(library: library, selectedPDF: $selectedPDF)
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
                // PDF viewer
                if let pdfDocument = pdfDocument, let pdfHash = selectedPDF?.hash {
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
                    .background(Color(.systemGroupedBackground))
                }

                // Floating color palette on the right
                if pdfDocument != nil {
                    HStack {
                        Spacer()
                        FloatingColorPalette(selectedColor: $selectedColor)
                            .padding(.trailing, 16)
                    }
                }

                // Download indicator overlay
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
            if let pdf = newValue {
                Task {
                    await loadPDF(metadata: pdf)
                }
            }
        }
    }

    func loadLibrary() async {
        isLoadingLibrary = true
        defer { isLoadingLibrary = false }

        // Do heavy work off main thread
        await Task.detached {
            // Load S3 state first (off main thread)
            S3StateManager.shared.loadState()

            do {
                // Fetch pdf-index.json from S3
                guard AWSCredentials.isConfigured() else {
                    await MainActor.run {
                        self.errorMessage = "AWS credentials not configured"
                    }
                    return
                }

                let indexURL = URL(string: "https://\(AWSCredentials.bucket).s3.\(AWSCredentials.region).amazonaws.com/pdf-index.json")!
                let (data, _) = try await URLSession.shared.data(from: indexURL)
                let pdfs = try JSONDecoder().decode([PDFMetadata].self, from: data)

                // Only keep PDFs that are in S3
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

    func loadPDF(metadata: PDFMetadata) async {
        do {
            let document = try await downloader.downloadPDF(metadata: metadata)
            pdfDocument = document
        } catch {
            errorMessage = "Failed to load PDF: \(error.localizedDescription)"
            print("Error loading PDF: \(error)")
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
            // Eraser at top
            EraserButton(isSelected: selectedColor == .clear) {
                withAnimation(.easeInOut(duration: 0.15)) {
                    selectedColor = .clear
                }
            }

            Divider()
                .frame(width: 20)
                .padding(.vertical, 2)

            // Default color dots
            ForEach(defaultColors, id: \.1) { color, name in
                ColorDot(
                    color: color,
                    isSelected: isDefaultColorSelected(name: name),
                    action: { selectDefaultColor(name: name) }
                )
            }

            // Custom colors
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

            // Add color button
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
        // Compare RGB components
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
        let uiColor = UIColor(color)
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        uiColor.getRed(&r, green: &g, blue: &b, alpha: &a)
        self.red = Double(r)
        self.green = Double(g)
        self.blue = Double(b)
        self.alpha = Double(a)
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
        NavigationView {
            VStack(spacing: 24) {
                ColorPicker("Choose a highlight color", selection: $color, supportsOpacity: false)
                    .labelsHidden()
                    .scaleEffect(1.5)

                // Preview
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
                    Button("Cancel") {
                        dismiss()
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Add") {
                        onAdd()
                    }
                }
            }
        }
    }
}

// MARK: - Add Color Button

struct AddColorButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(Color(.systemGray5))
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
                    .fill(Color(.systemGray5))
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
        ScrollView {
            LazyVStack(spacing: 24) {
                ForEach(0..<pdfDocument.pageCount, id: \.self) { pageIndex in
                    if let page = pdfDocument.page(at: pageIndex) {
                        PDFPageView(page: page, pdfHash: pdfHash, selectedColor: $selectedColor, pageIndex: pageIndex)
                            .aspectRatio(page.bounds(for: .mediaBox).width / page.bounds(for: .mediaBox).height, contentMode: .fit)
                            .frame(maxWidth: 800)
                            .background(Color.white)
                            .cornerRadius(4)
                            .shadow(color: Color.black.opacity(0.08), radius: 12, x: 0, y: 4)
                            .frame(minHeight: 600)
                    }
                }
            }
            .padding(.vertical, 24)
            .padding(.horizontal, 40)
        }
        .background(Color(.systemGroupedBackground))
        .scrollIndicators(.visible)
        .id(pdfHash) // Force view recreation when PDF changes
    }
}

struct PDFPageView: UIViewRepresentable {
    let page: PDFPage
    let pdfHash: String
    @Binding var selectedColor: Color
    let pageIndex: Int

    func makeCoordinator() -> Coordinator {
        Coordinator(selectedColor: $selectedColor, pdfHash: pdfHash, pageIndex: pageIndex)
    }

    func makeUIView(context: Context) -> UIView {
        let containerView = UIView()

        // PDF view
        let pdfView = PDFView()
        pdfView.document = PDFDocument()
        pdfView.document?.insert(page, at: 0)
        pdfView.autoScales = true
        pdfView.displayMode = .singlePage
        pdfView.translatesAutoresizingMaskIntoConstraints = false
        containerView.addSubview(pdfView)

        // PencilKit canvas overlay
        let canvasView = PKCanvasView()

        // On iPad: only allow Apple Pencil, not finger/mouse
        // On Mac Catalyst: allow any input
        #if targetEnvironment(macCatalyst)
        canvasView.drawingPolicy = .anyInput
        #else
        canvasView.drawingPolicy = .pencilOnly
        #endif

        canvasView.isOpaque = false
        canvasView.backgroundColor = .clear
        canvasView.translatesAutoresizingMaskIntoConstraints = false

        // Allow scroll gestures to pass through to the parent ScrollView
        canvasView.isUserInteractionEnabled = true

        // Important: Allow simultaneous gestures so scrolling still works
        if let scrollView = canvasView.subviews.first(where: { $0 is UIScrollView }) as? UIScrollView {
            scrollView.isScrollEnabled = false
        }

        containerView.addSubview(canvasView)

        // Setup tool with initial color (or eraser)
        if context.coordinator.selectedColor == .clear {
            canvasView.tool = PKEraserTool(.bitmap)
        } else {
            let ink = PKInkingTool(.marker, color: UIColor.highlightColor(from: context.coordinator.selectedColor), width: 20)
            canvasView.tool = ink
        }

        context.coordinator.canvasView = canvasView
        context.coordinator.containerView = containerView

        // Set up delegate to save drawings when changed
        canvasView.delegate = context.coordinator

        NSLayoutConstraint.activate([
            pdfView.topAnchor.constraint(equalTo: containerView.topAnchor),
            pdfView.leadingAnchor.constraint(equalTo: containerView.leadingAnchor),
            pdfView.trailingAnchor.constraint(equalTo: containerView.trailingAnchor),
            pdfView.bottomAnchor.constraint(equalTo: containerView.bottomAnchor),

            canvasView.topAnchor.constraint(equalTo: containerView.topAnchor),
            canvasView.leadingAnchor.constraint(equalTo: containerView.leadingAnchor),
            canvasView.trailingAnchor.constraint(equalTo: containerView.trailingAnchor),
            canvasView.bottomAnchor.constraint(equalTo: containerView.bottomAnchor)
        ])

        // Load any saved drawings for this page
        context.coordinator.loadDrawing()

        return containerView
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        // Update the tool when selectedColor changes
        if let canvasView = context.coordinator.canvasView {
            if selectedColor == .clear {
                canvasView.tool = PKEraserTool(.bitmap)
            } else {
                let ink = PKInkingTool(.marker, color: UIColor.highlightColor(from: selectedColor), width: 20)
                canvasView.tool = ink
            }
        }
    }

    class Coordinator: NSObject, PKCanvasViewDelegate {
        @Binding var selectedColor: Color
        weak var canvasView: PKCanvasView?
        weak var containerView: UIView?
        let pdfHash: String
        let pageIndex: Int

        init(selectedColor: Binding<Color>, pdfHash: String, pageIndex: Int) {
            self._selectedColor = selectedColor
            self.pdfHash = pdfHash
            self.pageIndex = pageIndex
            super.init()
        }

        // Called when drawing changes - save it
        func canvasViewDrawingDidChange(_ canvasView: PKCanvasView) {
            DrawingManager.shared.scheduleSave(canvasView.drawing, pdfHash: pdfHash, page: pageIndex)
        }

        func loadDrawing() {
            if let canvas = canvasView,
               let drawing = DrawingManager.shared.loadDrawing(pdfHash: pdfHash, page: pageIndex) {
                canvas.drawing = drawing
            }
        }
    }
}

extension UIColor {
    static func highlightColor(from color: Color) -> UIColor {
        // Handle known system colors with specific highlighter values
        switch color {
        case .yellow:
            return UIColor(red: 1.0, green: 1.0, blue: 0.0, alpha: 0.5)
        case .green:
            return UIColor(red: 0.0, green: 1.0, blue: 0.0, alpha: 0.5)
        case .red:
            return UIColor(red: 1.0, green: 0.35, blue: 0.35, alpha: 0.5)
        case .orange:
            return UIColor(red: 1.0, green: 0.65, blue: 0.0, alpha: 0.5)
        case .blue:
            return UIColor(red: 0.0, green: 0.5, blue: 1.0, alpha: 0.5)
        case .purple:
            return UIColor(red: 0.6, green: 0.3, blue: 0.9, alpha: 0.5)
        default:
            // For custom colors, use cgColor to extract components
            if let cgColor = color.cgColor,
               let components = cgColor.components,
               components.count >= 3 {
                return UIColor(red: components[0], green: components[1], blue: components[2], alpha: 0.5)
            }
            // Fallback
            return UIColor(red: 1.0, green: 1.0, blue: 0.0, alpha: 0.5)
        }
    }
}
