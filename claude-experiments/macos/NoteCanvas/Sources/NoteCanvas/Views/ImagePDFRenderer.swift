import AppKit
import PDFKit
import Foundation

// MARK: - Notification Names

extension Notification.Name {
    static let pdfAnnotationsDidUpdate = Notification.Name("pdfAnnotationsDidUpdate")
}

// MARK: - Serializable Annotation Models

struct SerializableAnnotation: Codable {
    let points: [CGPoint]
    let color: SerializableColor
    let lineWidth: CGFloat
    let tool: String
    let timestamp: Date
    
    init(from drawnPath: DrawingCanvasView.DrawnPath) {
        // Extract points from NSBezierPath
        var pathPoints: [CGPoint] = []
        let path = drawnPath.path
        
        for i in 0..<path.elementCount {
            var points = [NSPoint](repeating: NSPoint.zero, count: 3)
            let elementType = path.element(at: i, associatedPoints: &points)
            
            switch elementType {
            case .moveTo, .lineTo:
                pathPoints.append(points[0])
            case .curveTo, .cubicCurveTo:
                pathPoints.append(points[0])
                pathPoints.append(points[1])
                pathPoints.append(points[2])
            case .quadraticCurveTo:
                pathPoints.append(points[0])
                pathPoints.append(points[1])
            case .closePath:
                break
            @unknown default:
                break
            }
        }
        
        self.points = pathPoints
        self.color = SerializableColor(from: drawnPath.color)
        self.lineWidth = drawnPath.lineWidth
        self.tool = String(describing: drawnPath.tool)
        self.timestamp = Date()
    }
    
    func toDrawnPath() -> DrawingCanvasView.DrawnPath {
        let path = NSBezierPath()
        
        if !points.isEmpty {
            path.move(to: points[0])
            for point in points.dropFirst() {
                path.line(to: point)
            }
        }
        
        return DrawingCanvasView.DrawnPath(
            path: path,
            color: color.toNSColor(),
            lineWidth: lineWidth,
            tool: ImagePDFRenderer.Tool.from(string: tool)
        )
    }
}

struct SerializableColor: Codable {
    let red: CGFloat
    let green: CGFloat
    let blue: CGFloat
    let alpha: CGFloat
    
    init(from nsColor: NSColor) {
        let rgbColor = nsColor.usingColorSpace(.deviceRGB) ?? nsColor
        self.red = rgbColor.redComponent
        self.green = rgbColor.greenComponent
        self.blue = rgbColor.blueComponent
        self.alpha = rgbColor.alphaComponent
    }
    
    func toNSColor() -> NSColor {
        return NSColor(red: red, green: green, blue: blue, alpha: alpha)
    }
}

struct PDFAnnotations: Codable {
    let pdfPath: String
    let annotations: [SerializableAnnotation]
    let lastModified: Date
}

// MARK: - Annotation Persistence Manager

class AnnotationPersistenceManager {
    private static let annotationsDirectory = "PDF_Annotations"
    
    static func saveAnnotations(_ annotations: [DrawingCanvasView.DrawnPath], for pdfPath: URL) {
        let serializableAnnotations = annotations.map { SerializableAnnotation(from: $0) }
        let pdfAnnotations = PDFAnnotations(
            pdfPath: pdfPath.absoluteString,
            annotations: serializableAnnotations,
            lastModified: Date()
        )
        
        guard let data = try? JSONEncoder().encode(pdfAnnotations) else {
            print("Failed to encode annotations")
            return
        }
        
        let filename = annotationFilename(for: pdfPath)
        let annotationsDir = getAnnotationsDirectory()
        let filePath = annotationsDir.appendingPathComponent(filename)
        
        do {
            try data.write(to: filePath)
            print("Saved annotations for: \(pdfPath.lastPathComponent)")
            
            // Post notification that annotations were updated
            NotificationCenter.default.post(
                name: .pdfAnnotationsDidUpdate,
                object: nil,
                userInfo: ["pdfPath": pdfPath]
            )
        } catch {
            print("Failed to save annotations: \(error)")
        }
    }
    
    static func loadAnnotations(for pdfPath: URL) -> [DrawingCanvasView.DrawnPath] {
        let filename = annotationFilename(for: pdfPath)
        let annotationsDir = getAnnotationsDirectory()
        let filePath = annotationsDir.appendingPathComponent(filename)
        
        guard let data = try? Data(contentsOf: filePath),
              let pdfAnnotations = try? JSONDecoder().decode(PDFAnnotations.self, from: data) else {
            print("No saved annotations found for: \(pdfPath.lastPathComponent)")
            return []
        }
        
        print("Loaded \(pdfAnnotations.annotations.count) annotations for: \(pdfPath.lastPathComponent)")
        return pdfAnnotations.annotations.map { $0.toDrawnPath() }
    }
    
    private static func annotationFilename(for pdfPath: URL) -> String {
        let hash = String(pdfPath.absoluteString.hash)
        return "annotations_\(hash).json"
    }
    
    private static func getAnnotationsDirectory() -> URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let appDir = appSupport.appendingPathComponent("NoteCanvas")
        let annotationsDir = appDir.appendingPathComponent(annotationsDirectory)
        
        // Create directory if it doesn't exist
        try? FileManager.default.createDirectory(at: annotationsDir, withIntermediateDirectories: true)
        
        return annotationsDir
    }
}

// MARK: - Tool Extension for Serialization

extension ImagePDFRenderer.Tool {
    static func from(string: String) -> ImagePDFRenderer.Tool {
        switch string {
        case "highlightYellow": return .highlightYellow
        case "highlightGreen": return .highlightGreen
        case "highlightRed": return .highlightRed
        case "highlightBlue": return .highlightBlue
        case "draw": return .draw
        case "eraser": return .eraser
        default: return .none
        }
    }
}

// MARK: - Image-based PDF Renderer
class ImagePDFRenderer: NSView {
    internal let pdfNote: PDFNote
    private var pdfDocument: PDFDocument?
    private var currentPageIndex: Int = 0
    private var scrollView: NSScrollView!
    internal var documentView: NSView!
    private var pageImageView: NSImageView!
    internal var drawingCanvas: DrawingCanvasView!
    
    // Tool state
    enum Tool {
        case none, highlightYellow, highlightGreen, highlightRed, highlightBlue, draw, eraser
    }
    private var currentTool: Tool = .none
    
    // Zoom state
    private var zoomLevel: CGFloat = 1.0
    private let minZoom: CGFloat = 0.5
    private let maxZoom: CGFloat = 5.0
    
    init(pdfNote: PDFNote, frame: NSRect) {
        self.pdfNote = pdfNote
        super.init(frame: frame)
        setupRenderer()
        loadPDF()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupRenderer() {
        wantsLayer = true
        // Use canvas background color instead of control background
        layer?.backgroundColor = NSColor(red: 0.95, green: 0.95, blue: 0.97, alpha: 1.0).cgColor
        
        // Setup scroll view
        scrollView = NSScrollView(frame: bounds)
        scrollView.autoresizingMask = [.width, .height]
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = true
        scrollView.autohidesScrollers = false
        scrollView.borderType = .noBorder
        scrollView.backgroundColor = NSColor(red: 0.95, green: 0.95, blue: 0.97, alpha: 1.0)
        
        // Setup document view (container for PDF content)
        documentView = NSView()
        documentView.wantsLayer = true
        documentView.layer?.backgroundColor = NSColor(red: 0.95, green: 0.95, blue: 0.97, alpha: 1.0).cgColor
        
        // Setup page image view
        pageImageView = NSImageView()
        pageImageView.imageFrameStyle = .none
        pageImageView.imageScaling = .scaleProportionallyDown
        pageImageView.wantsLayer = true
        // No rounded corners or styling - should appear flat on canvas
        pageImageView.layer?.backgroundColor = NSColor.clear.cgColor
        
        documentView.addSubview(pageImageView)
        
        // Setup drawing canvas overlay
        drawingCanvas = DrawingCanvasView()
        drawingCanvas.delegate = self
        
        documentView.addSubview(drawingCanvas)
        
        // Load saved annotations
        loadAnnotations()
        
        scrollView.documentView = documentView
        addSubview(scrollView)
        
        // Add zoom gesture recognizer
        let magnificationGesture = NSMagnificationGestureRecognizer(target: self, action: #selector(handleZoom(_:)))
        addGestureRecognizer(magnificationGesture)
    }
    
    private func loadPDF() {
        guard let pdfDocument = PDFDocument(url: pdfNote.pdfPath) else {
            print("Failed to load PDF from: \(pdfNote.pdfPath)")
            return
        }
        
        self.pdfDocument = pdfDocument
        self.currentPageIndex = pdfNote.currentPage
        
        renderCurrentPage()
    }
    
    private func renderCurrentPage() {
        guard pdfDocument != nil else { return }
        
        // Render all pages vertically stacked
        renderAllPages()
    }
    
    private func renderAllPages() {
        guard let pdfDocument = pdfDocument else { return }
        
        let pageCount = pdfDocument.pageCount
        guard pageCount > 0 else { return }
        
        let availableWidth = scrollView.contentSize.width - 40 // padding
        var totalHeight: CGFloat = 0
        var pageFrames: [NSRect] = []
        
        // Calculate page sizes and positions
        var pageSizes: [CGSize] = []
        
        // First pass: calculate all page sizes
        for i in 0..<pageCount {
            guard let page = pdfDocument.page(at: i) else { continue }
            let pageSize = page.bounds(for: .mediaBox).size
            
            // Scale to fit width while maintaining aspect ratio, with minimum zoom of 1.5x
            let baseScale = availableWidth / pageSize.width
            let defaultScale = max(min(baseScale, 2.0), 1.5) // Min 1.5x, max 2.0x default zoom
            let scale = defaultScale * zoomLevel // Apply user zoom on top of default scale
            let scaledSize = CGSize(
                width: pageSize.width * scale,
                height: pageSize.height * scale
            )
            
            pageSizes.append(scaledSize)
            totalHeight += scaledSize.height + 40 // page height + spacing
        }
        
        // Second pass: calculate frames from top to bottom (page 0 at top)
        var currentY = totalHeight - 20 // Start from bottom and work up
        for i in 0..<pageCount {
            let scaledSize = pageSizes[i]
            currentY -= scaledSize.height + 20 // Move up by page height + spacing
            
            let pageFrame = NSRect(
                x: (availableWidth - scaledSize.width) / 2 + 20,
                y: currentY,
                width: scaledSize.width,
                height: scaledSize.height
            )
            
            pageFrames.append(pageFrame)
            currentY -= 20 // Additional spacing
        }
        
        // Create a single large image containing all pages
        let documentSize = CGSize(width: availableWidth + 40, height: totalHeight + 20)
        let image = NSImage(size: documentSize)
        image.lockFocus()
        
        // Canvas background color
        NSColor(red: 0.95, green: 0.95, blue: 0.97, alpha: 1.0).setFill()
        NSRect(origin: .zero, size: documentSize).fill()
        
        // Render each page
        for (i, pageFrame) in pageFrames.enumerated() {
            guard let page = pdfDocument.page(at: i) else { continue }
            
            // White page background
            NSColor.white.setFill()
            pageFrame.fill()
            
            // No border - flat appearance
            
            // Save graphics state
            NSGraphicsContext.current?.saveGraphicsState()
            
            // Set clipping to page frame
            NSBezierPath(rect: pageFrame).addClip()
            
            // Transform to page coordinates
            let transform = NSAffineTransform()
            transform.translateX(by: pageFrame.origin.x, yBy: pageFrame.origin.y)
            
            let pageSize = page.bounds(for: .mediaBox).size
            let scale = min(pageFrame.width / pageSize.width, pageFrame.height / pageSize.height)
            transform.scale(by: scale)
            transform.concat()
            
            NSGraphicsContext.current?.imageInterpolation = .high
            page.draw(with: .mediaBox, to: NSGraphicsContext.current!.cgContext)
            
            NSGraphicsContext.current?.restoreGraphicsState()
        }
        
        image.unlockFocus()
        
        pageImageView.image = image
        
        // Update document view size
        documentView.frame = NSRect(origin: .zero, size: documentSize)
        pageImageView.frame = NSRect(origin: .zero, size: documentSize)
        drawingCanvas.frame = NSRect(origin: .zero, size: documentSize)
        
        // Update drawing canvas
        drawingCanvas.setPageSize(documentSize)
        
        // Scroll to top of document
        scrollToTop()
    }
    
    private func scrollToTop() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            // Scroll to top of document
            let topPoint = NSPoint(x: 0, y: self.documentView.frame.height - self.scrollView.contentSize.height)
            self.scrollView.contentView.setBoundsOrigin(topPoint)
        }
    }
    
    @objc private func handleZoom(_ gesture: NSMagnificationGestureRecognizer) {
        switch gesture.state {
        case .began:
            // Store scroll position
            break
        case .changed:
            let newZoom = zoomLevel * (1 + gesture.magnification)
            let clampedZoom = max(minZoom, min(maxZoom, newZoom))
            
            // Use transform for immediate feedback during gesture
            let transform = CATransform3DMakeScale(clampedZoom / zoomLevel, clampedZoom / zoomLevel, 1.0)
            documentView.layer?.transform = transform
            
            gesture.magnification = 0
        case .ended, .cancelled:
            // Finalize zoom level and re-render
            let newZoom = zoomLevel * documentView.layer!.transform.m11
            zoomLevel = max(minZoom, min(maxZoom, newZoom))
            
            // Reset transform and re-render at new zoom level
            documentView.layer?.transform = CATransform3DIdentity
            renderAllPages()
        default:
            break
        }
    }
    
    // Tool selection methods
    func selectTool(_ tool: Tool) {
        currentTool = tool
        drawingCanvas.setTool(tool)
    }
    
    override func layout() {
        super.layout()
        scrollView.frame = bounds
        renderCurrentPage()
    }
    
    // MARK: - Annotation Persistence
    
    private func loadAnnotations() {
        let savedAnnotations = AnnotationPersistenceManager.loadAnnotations(for: pdfNote.pdfPath)
        drawingCanvas.loadAnnotations(savedAnnotations)
    }
    
    private func saveAnnotations() {
        let currentAnnotations = drawingCanvas.getAnnotations()
        AnnotationPersistenceManager.saveAnnotations(currentAnnotations, for: pdfNote.pdfPath)
    }
}

// MARK: - Drawing Canvas Delegate
extension ImagePDFRenderer: DrawingCanvasDelegate {
    func drawingCanvasDidUpdate(_ canvas: DrawingCanvasView) {
        // Save annotations whenever they're updated
        saveAnnotations()
    }
}

// MARK: - Custom Drawing Canvas
protocol DrawingCanvasDelegate: AnyObject {
    func drawingCanvasDidUpdate(_ canvas: DrawingCanvasView)
}

class DrawingCanvasView: NSView {
    weak var delegate: DrawingCanvasDelegate?
    private var pageSize: CGSize = .zero
    var currentTool: ImagePDFRenderer.Tool = .none // Made public for debugging
    
    // Drawing state
    private var isDrawing = false
    private var currentPath: NSBezierPath?
    private var drawings: [DrawnPath] = []
    
    // Performance optimization
    private var lastDrawPoint: NSPoint = .zero
    private let minDrawDistance: CGFloat = 1.5
    
    struct DrawnPath {
        let path: NSBezierPath
        let color: NSColor
        let lineWidth: CGFloat
        let tool: ImagePDFRenderer.Tool
    }
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupCanvas()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupCanvas()
    }
    
    private func setupCanvas() {
        wantsLayer = true
        layer?.backgroundColor = NSColor.clear.cgColor
    }
    
    func setPageSize(_ size: CGSize) {
        pageSize = size
        needsDisplay = true
    }
    
    func setTool(_ tool: ImagePDFRenderer.Tool) {
        currentTool = tool
    }
    
    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        
        // Draw all completed paths
        for drawing in drawings {
            drawing.color.setStroke()
            drawing.path.lineWidth = drawing.lineWidth
            drawing.path.stroke()
        }
        
        // Draw current path being drawn
        if let currentPath = currentPath, isDrawing {
            getCurrentColor().setStroke()
            currentPath.lineWidth = getCurrentLineWidth()
            currentPath.stroke()
        }
    }
    
    private func getCurrentColor() -> NSColor {
        switch currentTool {
        case .highlightYellow:
            return NSColor.systemYellow.withAlphaComponent(0.4)
        case .highlightGreen:
            return NSColor.systemGreen.withAlphaComponent(0.4)
        case .highlightRed:
            return NSColor.systemRed.withAlphaComponent(0.4)
        case .highlightBlue:
            return NSColor.systemBlue.withAlphaComponent(0.4)
        case .draw:
            return NSColor.systemRed
        case .eraser:
            return NSColor.white // We'll implement proper erasing differently
        case .none:
            return NSColor.black
        }
    }
    
    private func getCurrentLineWidth() -> CGFloat {
        switch currentTool {
        case .highlightYellow, .highlightGreen, .highlightRed, .highlightBlue:
            return 8.0
        case .draw:
            return 2.0
        case .eraser:
            return 10.0
        case .none:
            return 1.0
        }
    }
    
    override func mouseDown(with event: NSEvent) {
        guard currentTool != .none else { 
            return 
        }
        
        let point = convert(event.locationInWindow, from: nil)
        
        if currentTool == .eraser {
            // For eraser, remove paths that intersect with the click point
            eraseAtPoint(point)
        } else {
            // Start drawing
            isDrawing = true
            lastDrawPoint = point
            currentPath = NSBezierPath()
            currentPath?.move(to: point)
            
            // Set line style based on tool type
            switch currentTool {
            case .highlightYellow, .highlightGreen, .highlightRed, .highlightBlue:
                currentPath?.lineCapStyle = .square
                currentPath?.lineJoinStyle = .miter
            case .draw:
                currentPath?.lineCapStyle = .round
                currentPath?.lineJoinStyle = .round
            default:
                currentPath?.lineCapStyle = .round
                currentPath?.lineJoinStyle = .round
            }
        }
        
        needsDisplay = true
    }
    
    override func mouseDragged(with event: NSEvent) {
        guard isDrawing, let path = currentPath else {
            if currentTool == .eraser {
                // Continue erasing while dragging
                let point = convert(event.locationInWindow, from: nil)
                eraseAtPoint(point)
                needsDisplay = true
            }
            return
        }
        
        let point = convert(event.locationInWindow, from: nil)
        
        // Performance optimization: only add points if far enough apart
        let distance = sqrt(pow(point.x - lastDrawPoint.x, 2) + pow(point.y - lastDrawPoint.y, 2))
        if distance >= minDrawDistance {
            path.line(to: point)
            lastDrawPoint = point
            needsDisplay = true
        }
    }
    
    override func mouseUp(with event: NSEvent) {
        guard isDrawing, let path = currentPath else { return }
        
        // Add completed path to drawings
        let drawnPath = DrawnPath(
            path: path.copy() as! NSBezierPath,
            color: getCurrentColor(),
            lineWidth: getCurrentLineWidth(),
            tool: currentTool
        )
        drawings.append(drawnPath)
        
        // Reset drawing state
        isDrawing = false
        currentPath = nil
        
        needsDisplay = true
        delegate?.drawingCanvasDidUpdate(self)
    }
    
    private func eraseAtPoint(_ point: NSPoint) {
        // Remove paths that contain or are close to the erase point
        drawings.removeAll { drawnPath in
            // Check if path bounds contain the point or if point is close to the path
            let bounds = drawnPath.path.bounds.insetBy(dx: -5, dy: -5)
            if bounds.contains(point) {
                // More precise check: see if point is actually on the path
                return drawnPath.path.contains(point) || 
                       isPointNearPath(point, path: drawnPath.path, tolerance: 8.0)
            }
            return false
        }
    }
    
    private func isPointNearPath(_ point: NSPoint, path: NSBezierPath, tolerance: CGFloat) -> Bool {
        // Simple approximation: check if point is within tolerance of path bounds
        let expandedBounds = path.bounds.insetBy(dx: -tolerance, dy: -tolerance)
        return expandedBounds.contains(point)
    }
    
    // Clear all drawings
    func clearDrawings() {
        drawings.removeAll()
        isDrawing = false
        currentPath = nil
        needsDisplay = true
    }
    
    // MARK: - Annotation Persistence
    
    func loadAnnotations(_ savedDrawings: [DrawnPath]) {
        drawings = savedDrawings
        needsDisplay = true
    }
    
    func getAnnotations() -> [DrawnPath] {
        return drawings
    }
}