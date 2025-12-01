import SwiftUI
import PencilKit
import PDFKit

struct ContentView: View {
    @State private var selectedColor: Color = .yellow
    @State private var showingFilePicker = false
    @State private var pdfDocument: PDFDocument?
    @State private var selectedPDF: PDFMetadata?
    @State private var library: PDFLibrary?
    @State private var isLoadingLibrary = false
    @State private var errorMessage: String?

    @StateObject private var downloader = PDFDownloader()

    let highlightColors: [Color] = [.yellow, .green, .pink, .orange, .blue, .purple]

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
            VStack(spacing: 0) {
                // Toolbar
                HStack(spacing: 16) {
                    Text("Highlight:")
                        .font(.headline)

                    ForEach(highlightColors, id: \.self) { color in
                        Circle()
                            .fill(color.opacity(0.5))
                            .frame(width: 40, height: 40)
                            .overlay(
                                Circle()
                                    .strokeBorder(Color.primary, lineWidth: selectedColor == color ? 3 : 1)
                            )
                            .onTapGesture {
                                selectedColor = color
                            }
                    }

                    Spacer()

                    // Download indicator
                    if downloader.isDownloading {
                        ProgressView()
                            .scaleEffect(0.8)
                        Text("Downloading...")
                            .font(.caption)
                    }

                    Button("Open File") {
                        showingFilePicker = true
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding()
                .background(Color(.systemBackground))

                Divider()

                // PDF viewer
                if let pdfDocument = pdfDocument {
                    PDFMarkupView(pdfDocument: pdfDocument, selectedColor: $selectedColor)
                } else {
                    VStack(spacing: 20) {
                        Image(systemName: "doc.fill")
                            .font(.system(size: 64))
                            .foregroundStyle(.secondary)
                        Text("Select a PDF from the sidebar or open a file")
                            .font(.title2)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
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
        .fileImporter(
            isPresented: $showingFilePicker,
            allowedContentTypes: [.pdf]
        ) { result in
            switch result {
            case .success(let url):
                if url.startAccessingSecurityScopedResource() {
                    defer { url.stopAccessingSecurityScopedResource() }
                    if let doc = PDFDocument(url: url) {
                        pdfDocument = doc
                    }
                }
            case .failure(let error):
                print("Error selecting file: \(error)")
            }
        }
    }

    func loadLibrary() async {
        isLoadingLibrary = true
        defer { isLoadingLibrary = false }

        let indexPath = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/reading-tools/pdf-indexer/pdf-index.json"

        // Do heavy work off main thread
        await Task.detached {
            // Load S3 state first (off main thread)
            S3StateManager.shared.loadState()

            do {
                let data = try Data(contentsOf: URL(fileURLWithPath: indexPath))
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

struct PDFMarkupView: View {
    let pdfDocument: PDFDocument
    @Binding var selectedColor: Color

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 20) {
                ForEach(0..<pdfDocument.pageCount, id: \.self) { pageIndex in
                    if let page = pdfDocument.page(at: pageIndex) {
                        VStack(spacing: 8) {
                            // Page number indicator
                            HStack {
                                Text("Page \(pageIndex + 1)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 4)
                                    .background(Color(.systemGray5))
                                    .cornerRadius(12)
                                Spacer()
                            }
                            .padding(.horizontal)

                            // PDF page with PencilKit overlay
                            PDFPageView(page: page, selectedColor: $selectedColor, pageIndex: pageIndex)
                                .aspectRatio(page.bounds(for: .mediaBox).width / page.bounds(for: .mediaBox).height, contentMode: .fit)
                                .frame(maxWidth: 800)
                                .shadow(color: Color.black.opacity(0.1), radius: 8, x: 0, y: 4)
                                .frame(minHeight: 600) // Ensure minimum height for each page
                        }
                    }
                }
            }
            .padding()
        }
        .scrollIndicators(.visible)
    }
}

struct PDFPageView: UIViewRepresentable {
    let page: PDFPage
    @Binding var selectedColor: Color
    let pageIndex: Int

    func makeCoordinator() -> Coordinator {
        Coordinator(selectedColor: $selectedColor, pageIndex: pageIndex)
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

        // Setup tool with initial color
        let ink = PKInkingTool(.marker, color: UIColor(context.coordinator.selectedColor), width: 20)
        canvasView.tool = ink

        context.coordinator.canvasView = canvasView
        context.coordinator.containerView = containerView

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
        // Update the tool color when selectedColor changes
        if let canvasView = context.coordinator.canvasView {
            let ink = PKInkingTool(.marker, color: UIColor(selectedColor), width: 20)
            canvasView.tool = ink
        }
    }

    class Coordinator: NSObject {
        @Binding var selectedColor: Color
        weak var canvasView: PKCanvasView?
        weak var containerView: UIView?
        let pageIndex: Int

        // Store drawings per page
        static var pageDrawings: [Int: PKDrawing] = [:]

        init(selectedColor: Binding<Color>, pageIndex: Int) {
            self._selectedColor = selectedColor
            self.pageIndex = pageIndex
            super.init()
        }

        func saveDrawing() {
            if let canvas = canvasView {
                Coordinator.pageDrawings[pageIndex] = canvas.drawing
            }
        }

        func loadDrawing() {
            if let canvas = canvasView, let drawing = Coordinator.pageDrawings[pageIndex] {
                canvas.drawing = drawing
            }
        }
    }
}

extension UIColor {
    convenience init(_ color: Color) {
        // Use UIColor's native color system instead of trying to extract components
        switch color {
        case .yellow:
            self.init(red: 1.0, green: 1.0, blue: 0.0, alpha: 0.5)
        case .green:
            self.init(red: 0.0, green: 1.0, blue: 0.0, alpha: 0.5)
        case .pink:
            self.init(red: 1.0, green: 0.75, blue: 0.8, alpha: 0.5)
        case .orange:
            self.init(red: 1.0, green: 0.65, blue: 0.0, alpha: 0.5)
        case .blue:
            self.init(red: 0.0, green: 0.5, blue: 1.0, alpha: 0.5)
        case .purple:
            self.init(red: 0.5, green: 0.0, blue: 0.5, alpha: 0.5)
        default:
            // Fallback to yellow
            self.init(red: 1.0, green: 1.0, blue: 0.0, alpha: 0.5)
        }
    }
}
