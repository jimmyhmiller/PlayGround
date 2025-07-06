import SwiftUI
import AppKit

struct ImageCanvasView: NSViewRepresentable {
    @Binding var rectangles: [Rectangle]
    @Binding var textBubbles: [TextBubble]
    @Binding var selectedColor: Color
    @Binding var selectedRectangle: Rectangle?
    @Binding var selectedTextBubble: TextBubble?
    @Binding var selectedTool: Tool
    let pdfImages: [PDFImageService.PDFPageImage]
    
    func makeNSView(context: Context) -> ImageDrawingView {
        let view = ImageDrawingView()
        view.rectangles = rectangles
        view.textBubbles = textBubbles
        view.selectedColor = selectedColor
        view.selectedRectangle = selectedRectangle
        view.selectedTextBubble = selectedTextBubble
        view.selectedTool = selectedTool
        view.pdfImages = pdfImages
        view.delegate = context.coordinator
        
        // Scroll to top when first created with PDF images
        if !pdfImages.isEmpty {
            view.scrollToTop()
        }
        
        return view
    }
    
    func updateNSView(_ nsView: ImageDrawingView, context: Context) {
        nsView.rectangles = rectangles
        nsView.textBubbles = textBubbles
        nsView.selectedColor = selectedColor
        nsView.selectedRectangle = selectedRectangle
        nsView.selectedTextBubble = selectedTextBubble
        nsView.selectedTool = selectedTool
        nsView.pdfImages = pdfImages
        nsView.updateContent()
        nsView.needsDisplay = true
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, ImageDrawingViewDelegate {
        var parent: ImageCanvasView
        
        init(_ parent: ImageCanvasView) {
            self.parent = parent
        }
        
        func rectanglesDidChange(_ rectangles: [Rectangle]) {
            parent.rectangles = rectangles
        }
        
        func selectedRectangleDidChange(_ rectangle: Rectangle?) {
            parent.selectedRectangle = rectangle
        }
        
        func textBubblesDidChange(_ textBubbles: [TextBubble]) {
            parent.textBubbles = textBubbles
        }
        
        func selectedTextBubbleDidChange(_ textBubble: TextBubble?) {
            parent.selectedTextBubble = textBubble
        }
    }
}

protocol ImageDrawingViewDelegate: AnyObject {
    func rectanglesDidChange(_ rectangles: [Rectangle])
    func selectedRectangleDidChange(_ rectangle: Rectangle?)
    func textBubblesDidChange(_ textBubbles: [TextBubble])
    func selectedTextBubbleDidChange(_ textBubble: TextBubble?)
}

class ImageDrawingView: NSView {
    var rectangles: [Rectangle] = []
    var textBubbles: [TextBubble] = []
    var selectedColor: Color = .blue
    var selectedRectangle: Rectangle?
    var selectedTextBubble: TextBubble?
    var selectedTool: Tool = .select
    var pdfImages: [PDFImageService.PDFPageImage] = []
    weak var delegate: ImageDrawingViewDelegate?
    
    internal var scrollView: NSScrollView?
    private var contentView: ImageContentView?
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupScrollView()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupScrollView()
    }
    
    private func setupScrollView() {
        scrollView = NSScrollView()
        scrollView?.hasVerticalScroller = true
        scrollView?.hasHorizontalScroller = true
        scrollView?.autohidesScrollers = false
        scrollView?.allowsMagnification = true
        scrollView?.minMagnification = 0.25
        scrollView?.maxMagnification = 3.0
        scrollView?.translatesAutoresizingMaskIntoConstraints = false
        scrollView?.backgroundColor = NSColor.controlBackgroundColor
        
        contentView = ImageContentView()
        contentView?.drawingView = self
        contentView?.translatesAutoresizingMaskIntoConstraints = false
        scrollView?.documentView = contentView
        
        addSubview(scrollView!)
        
        NSLayoutConstraint.activate([
            scrollView!.topAnchor.constraint(equalTo: topAnchor),
            scrollView!.leadingAnchor.constraint(equalTo: leadingAnchor),
            scrollView!.trailingAnchor.constraint(equalTo: trailingAnchor),
            scrollView!.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
    }
    
    func updateContent() {
        contentView?.updateLayout()
    }
    
    func scrollToTop() {
        if let scrollView = scrollView,
           let contentView = contentView {
            DispatchQueue.main.async {
                scrollView.documentView?.scroll(NSPoint(x: 0, y: contentView.bounds.maxY))
            }
        }
    }
    
    override var acceptsFirstResponder: Bool {
        return true
    }
}

class ImageContentView: NSView {
    weak var drawingView: ImageDrawingView?
    
    private var isDrawing = false
    private var drawingStart: CGPoint = .zero
    private var currentDrawingRect: CGRect = .zero
    
    private var isDragging = false
    private var dragOffset: CGSize = .zero
    private var lastClickTime: TimeInterval = 0
    private var lastClickedTextBubble: TextBubble?
    private var textEditor: InlineTextEditor?
    private var editingTextBubble: TextBubble?
    
    // Page layout constants
    private let pageSpacing: CGFloat = 20
    private let pageMargin: CGFloat = 50
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupTrackingArea()
        wantsLayer = false // Disable layer backing for better performance
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupTrackingArea()
        wantsLayer = false // Disable layer backing for better performance
    }
    
    private func setupTrackingArea() {
        let trackingArea = NSTrackingArea(
            rect: bounds,
            options: [.activeInKeyWindow, .mouseMoved, .mouseEnteredAndExited],
            owner: self,
            userInfo: nil
        )
        addTrackingArea(trackingArea)
    }
    
    override func updateTrackingAreas() {
        trackingAreas.forEach { removeTrackingArea($0) }
        setupTrackingArea()
    }
    
    func updateLayout() {
        guard let drawingView = drawingView else { return }
        
        if drawingView.pdfImages.isEmpty {
            frame = CGRect(origin: .zero, size: CGSize(width: 800, height: 600))
            needsDisplay = true
            return
        }
        
        var totalHeight: CGFloat = pageMargin
        var maxWidth: CGFloat = 0
        
        for pageImage in drawingView.pdfImages {
            let imageSize = pageImage.image.size
            maxWidth = max(maxWidth, imageSize.width)
            totalHeight += imageSize.height + pageSpacing
        }
        
        totalHeight += pageMargin - pageSpacing // Remove last spacing, add bottom margin
        
        let contentSize = CGSize(
            width: max(maxWidth + 2 * pageMargin, 800), // Minimum width
            height: totalHeight
        )
        
        frame = CGRect(origin: .zero, size: contentSize)
        
        // Force scroll view to recognize new content size
        DispatchQueue.main.async { [weak self] in
            self?.superview?.needsLayout = true
        }
        
        needsDisplay = true
    }
    
    // Convert point to page coordinates and return which page
    private func pageInfo(for point: CGPoint) -> (pageIndex: Int, pagePoint: CGPoint, pageRect: CGRect)? {
        guard let drawingView = drawingView else { return nil }
        
        var currentY: CGFloat = bounds.height - pageMargin
        
        for (index, pageImage) in drawingView.pdfImages.enumerated() {
            let imageSize = pageImage.image.size
            currentY -= imageSize.height
            
            let pageRect = CGRect(
                x: pageMargin,
                y: currentY,
                width: imageSize.width,
                height: imageSize.height
            )
            
            if pageRect.contains(point) {
                let pagePoint = CGPoint(
                    x: point.x - pageRect.origin.x,
                    y: point.y - pageRect.origin.y
                )
                return (index, pagePoint, pageRect)
            }
            
            currentY -= pageSpacing
        }
        
        return nil
    }
    
    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        
        guard let context = NSGraphicsContext.current?.cgContext,
              let drawingView = drawingView else { return }
        
        // Fill background
        context.setFillColor(NSColor.controlBackgroundColor.cgColor)
        context.fill(bounds)
        
        // Draw PDF page images from top to bottom (only visible ones)
        var currentY: CGFloat = bounds.height - pageMargin
        
        for pageImage in drawingView.pdfImages {
            let imageSize = pageImage.image.size
            currentY -= imageSize.height
            
            let imageRect = CGRect(
                x: pageMargin,
                y: currentY,
                width: imageSize.width,
                height: imageSize.height
            )
            
            // Only draw if the page intersects with the dirty rect (performance optimization)
            if dirtyRect.intersects(imageRect) {
                // Draw page shadow
                let shadowRect = imageRect.insetBy(dx: -2, dy: -2).offsetBy(dx: 2, dy: 2)
                context.setFillColor(NSColor.black.withAlphaComponent(0.3).cgColor)
                context.fill(shadowRect)
                
                // Draw page image
                if let cgImage = pageImage.image.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                    context.draw(cgImage, in: imageRect)
                }
            }
            
            currentY -= pageSpacing
        }
        
        // Draw rectangles
        for rectangle in drawingView.rectangles {
            drawRectangle(rectangle, in: context)
        }
        
        // Draw text bubbles
        for textBubble in drawingView.textBubbles {
            drawTextBubble(textBubble, in: context)
        }
        
        // Draw current drawing rectangle
        if isDrawing && currentDrawingRect.size.width > 0 && currentDrawingRect.size.height > 0 {
            context.setStrokeColor(NSColor(drawingView.selectedColor).cgColor)
            context.setLineWidth(2)
            context.setLineDash(phase: 0, lengths: [5, 5])
            context.stroke(currentDrawingRect)
        }
    }
    
    private func drawRectangle(_ rectangle: Rectangle, in context: CGContext) {
        RectangleDrawing.drawRectangle(
            bounds: rectangle.frame,
            color: NSColor(rectangle.color),
            cornerRadius: rectangle.cornerRadius,
            isSelected: rectangle.isSelected,
            in: context
        )
    }
    
    private func drawTextBubble(_ textBubble: TextBubble, in context: CGContext) {
        let isEditing = editingTextBubble?.id == textBubble.id
        TextBubbleDrawing.drawTextBubble(
            bounds: textBubble.frame,
            text: textBubble.text,
            color: NSColor(textBubble.color),
            fontSize: textBubble.fontSize,
            isSelected: textBubble.isSelected,
            isEditing: isEditing,
            in: context
        )
    }
    
    override var acceptsFirstResponder: Bool {
        return true
    }
    
    override func mouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        let currentTime = Date().timeIntervalSince1970
        let isDoubleClick = (currentTime - lastClickTime) < 0.5
        
        guard let drawingView = drawingView else { return }
        
        switch drawingView.selectedTool {
        case .select:
            // Check if clicked on rectangle
            if let rectangle = drawingView.rectangles.first(where: { $0.contains(point: point) }) {
                selectRectangle(rectangle)
                startDragging(at: point, for: rectangle)
            }
            // Check if clicked on text bubble  
            else if let textBubble = drawingView.textBubbles.first(where: { $0.contains(point: point) }) {
                if isDoubleClick && lastClickedTextBubble?.id == textBubble.id {
                    startEditingTextBubble(textBubble)
                    lastClickTime = 0
                    lastClickedTextBubble = nil
                    return
                }
                
                lastClickTime = currentTime
                lastClickedTextBubble = textBubble
                selectTextBubble(textBubble)
                startDragging(at: point, for: textBubble)
            }
            else {
                finishEditingTextBubble()
                deselectAll()
            }
            
        case .rectangle:
            finishEditingTextBubble()
            deselectAll()
            isDrawing = true
            drawingStart = point
            currentDrawingRect = CGRect(origin: point, size: .zero)
            
        case .text:
            finishEditingTextBubble()
            deselectAll()
            
            // Create text bubble
            let newTextBubble = TextBubble(
                origin: point,
                text: "Text",
                color: drawingView.selectedColor
            )
            drawingView.textBubbles.append(newTextBubble)
            drawingView.delegate?.textBubblesDidChange(drawingView.textBubbles)
            
            // Start editing immediately
            startEditingTextBubble(newTextBubble)
        }
        
        needsDisplay = true
    }
    
    override func mouseDragged(with event: NSEvent) {
        let currentPoint = convert(event.locationInWindow, from: nil)
        
        guard let drawingView = drawingView else { return }
        
        if isDragging {
            let newOrigin = CGPoint(
                x: currentPoint.x - dragOffset.width,
                y: currentPoint.y - dragOffset.height
            )
            
            if let selectedRect = drawingView.selectedRectangle {
                updateRectanglePosition(selectedRect, to: newOrigin)
            } else if let selectedText = drawingView.selectedTextBubble {
                updateTextBubblePosition(selectedText, to: newOrigin)
            }
        } else if isDrawing {
            let origin = CGPoint(
                x: min(drawingStart.x, currentPoint.x),
                y: min(drawingStart.y, currentPoint.y)
            )
            let size = CGSize(
                width: abs(currentPoint.x - drawingStart.x),
                height: abs(currentPoint.y - drawingStart.y)
            )
            currentDrawingRect = CGRect(origin: origin, size: size)
        }
        
        needsDisplay = true
    }
    
    override func mouseUp(with event: NSEvent) {
        guard let drawingView = drawingView else { return }
        
        if isDragging {
            isDragging = false
            dragOffset = .zero
        } else if isDrawing {
            isDrawing = false
            
            if currentDrawingRect.size.width > 5 && currentDrawingRect.size.height > 5 {
                let newRectangle = Rectangle(
                    origin: currentDrawingRect.origin,
                    size: currentDrawingRect.size,
                    color: drawingView.selectedColor
                )
                drawingView.rectangles.append(newRectangle)
                drawingView.delegate?.rectanglesDidChange(drawingView.rectangles)
            }
            
            currentDrawingRect = .zero
        }
        
        needsDisplay = true
    }
    
    override func keyDown(with event: NSEvent) {
        guard let drawingView = drawingView else { return }
        
        if event.keyCode == 51 { // Delete key
            if let selected = drawingView.selectedRectangle {
                drawingView.rectangles.removeAll { $0.id == selected.id }
                drawingView.selectedRectangle = nil
                drawingView.delegate?.rectanglesDidChange(drawingView.rectangles)
                drawingView.delegate?.selectedRectangleDidChange(nil)
                needsDisplay = true
            } else if let selected = drawingView.selectedTextBubble {
                drawingView.textBubbles.removeAll { $0.id == selected.id }
                drawingView.selectedTextBubble = nil
                drawingView.delegate?.textBubblesDidChange(drawingView.textBubbles)
                drawingView.delegate?.selectedTextBubbleDidChange(nil)
                needsDisplay = true
            }
        } else {
            super.keyDown(with: event)
        }
    }
    
    // MARK: - Helper Methods
    
    private func selectRectangle(_ rectangle: Rectangle) {
        guard let drawingView = drawingView else { return }
        
        // Update selection state
        for i in 0..<drawingView.rectangles.count {
            drawingView.rectangles[i].isSelected = drawingView.rectangles[i].id == rectangle.id
        }
        for i in 0..<drawingView.textBubbles.count {
            drawingView.textBubbles[i].isSelected = false
        }
        
        drawingView.selectedRectangle = rectangle
        drawingView.selectedTextBubble = nil
        drawingView.delegate?.selectedRectangleDidChange(rectangle)
        drawingView.delegate?.selectedTextBubbleDidChange(nil)
        drawingView.delegate?.rectanglesDidChange(drawingView.rectangles)
        drawingView.delegate?.textBubblesDidChange(drawingView.textBubbles)
    }
    
    private func selectTextBubble(_ textBubble: TextBubble) {
        guard let drawingView = drawingView else { return }
        
        // Update selection state
        for i in 0..<drawingView.rectangles.count {
            drawingView.rectangles[i].isSelected = false
        }
        for i in 0..<drawingView.textBubbles.count {
            drawingView.textBubbles[i].isSelected = drawingView.textBubbles[i].id == textBubble.id
        }
        
        drawingView.selectedRectangle = nil
        drawingView.selectedTextBubble = textBubble
        drawingView.delegate?.selectedRectangleDidChange(nil)
        drawingView.delegate?.selectedTextBubbleDidChange(textBubble)
        drawingView.delegate?.rectanglesDidChange(drawingView.rectangles)
        drawingView.delegate?.textBubblesDidChange(drawingView.textBubbles)
    }
    
    private func deselectAll() {
        guard let drawingView = drawingView else { return }
        
        for i in 0..<drawingView.rectangles.count {
            drawingView.rectangles[i].isSelected = false
        }
        for i in 0..<drawingView.textBubbles.count {
            drawingView.textBubbles[i].isSelected = false
        }
        
        drawingView.selectedRectangle = nil
        drawingView.selectedTextBubble = nil
        drawingView.delegate?.selectedRectangleDidChange(nil)
        drawingView.delegate?.selectedTextBubbleDidChange(nil)
        drawingView.delegate?.rectanglesDidChange(drawingView.rectangles)
        drawingView.delegate?.textBubblesDidChange(drawingView.textBubbles)
    }
    
    private func startDragging(at point: CGPoint, for rectangle: Rectangle) {
        isDragging = true
        dragOffset = CGSize(
            width: point.x - rectangle.origin.x,
            height: point.y - rectangle.origin.y
        )
    }
    
    private func startDragging(at point: CGPoint, for textBubble: TextBubble) {
        isDragging = true
        dragOffset = CGSize(
            width: point.x - textBubble.origin.x,
            height: point.y - textBubble.origin.y
        )
    }
    
    private func updateRectanglePosition(_ rectangle: Rectangle, to newOrigin: CGPoint) {
        guard let drawingView = drawingView else { return }
        
        for i in 0..<drawingView.rectangles.count {
            if drawingView.rectangles[i].id == rectangle.id {
                drawingView.rectangles[i].origin = newOrigin
                drawingView.selectedRectangle = drawingView.rectangles[i]
                break
            }
        }
        
        drawingView.delegate?.rectanglesDidChange(drawingView.rectangles)
        drawingView.delegate?.selectedRectangleDidChange(drawingView.selectedRectangle)
    }
    
    private func updateTextBubblePosition(_ textBubble: TextBubble, to newOrigin: CGPoint) {
        guard let drawingView = drawingView else { return }
        
        for i in 0..<drawingView.textBubbles.count {
            if drawingView.textBubbles[i].id == textBubble.id {
                drawingView.textBubbles[i].origin = newOrigin
                drawingView.selectedTextBubble = drawingView.textBubbles[i]
                break
            }
        }
        
        drawingView.delegate?.textBubblesDidChange(drawingView.textBubbles)
        drawingView.delegate?.selectedTextBubbleDidChange(drawingView.selectedTextBubble)
    }
    
    private func startEditingTextBubble(_ textBubble: TextBubble) {
        // Remove any existing text editor
        textEditor?.removeFromSuperview()
        
        // Create new text editor
        let font = NSFont.systemFont(ofSize: textBubble.fontSize)
        let attributes: [NSAttributedString.Key: Any] = [.font: font]
        let textSize = (textBubble.text as NSString).size(withAttributes: attributes)
        
        let padding: CGFloat = 20
        let editorWidth = max(textSize.width + padding, 60)
        let editorHeight = max(textSize.height + 8, 24)
        
        let editorFrame = CGRect(
            x: textBubble.origin.x - 2,  // Fine-tuned offset
            y: textBubble.origin.y + (textBubble.size.height - editorHeight) / 2 - 4,  // Fine-tuned offset
            width: editorWidth,
            height: editorHeight
        )
        
        let editor = InlineTextEditor(frame: editorFrame)
        editor.startEditing(with: textBubble.text, in: editorFrame, color: NSColor(textBubble.color))
        
        editor.onTextChanged = { [weak self] newText in
            self?.updateTextBubbleText(textBubble, newText: newText)
        }
        
        editor.onEditingFinished = { [weak self] in
            self?.finishEditingTextBubble()
        }
        
        addSubview(editor)
        textEditor = editor
        editingTextBubble = textBubble
        
        isDragging = false
    }
    
    private func updateTextBubbleText(_ textBubble: TextBubble, newText: String) {
        guard let drawingView = drawingView,
              let editingIndex = drawingView.textBubbles.firstIndex(where: { $0.id == textBubble.id }) else { return }
        
        var updatedTextBubble = drawingView.textBubbles[editingIndex]
        updatedTextBubble.text = newText
        drawingView.textBubbles[editingIndex] = updatedTextBubble
        
        if drawingView.selectedTextBubble?.id == textBubble.id {
            drawingView.selectedTextBubble = updatedTextBubble
            drawingView.delegate?.selectedTextBubbleDidChange(updatedTextBubble)
        }
        
        drawingView.delegate?.textBubblesDidChange(drawingView.textBubbles)
        needsDisplay = true
    }
    
    private func finishEditingTextBubble() {
        // Make sure to save final text state before removing editor
        if let editor = textEditor,
           let editingBubble = editingTextBubble {
            updateTextBubbleText(editingBubble, newText: editor.stringValue)
        }
        
        textEditor?.removeFromSuperview()
        textEditor = nil
        editingTextBubble = nil
        needsDisplay = true
    }
}