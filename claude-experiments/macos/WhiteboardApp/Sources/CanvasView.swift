import SwiftUI
import AppKit
import Quartz

struct CanvasView: NSViewRepresentable {
    @Binding var rectangles: [Rectangle]
    @Binding var textBubbles: [TextBubble]
    @Binding var pdfItems: [PDFItem]
    @Binding var selectedColor: Color
    @Binding var selectedRectangle: Rectangle?
    @Binding var selectedTextBubble: TextBubble?
    @Binding var selectedPDF: PDFItem?
    @Binding var selectedTool: Tool
    var onPDFDoubleClick: ((PDFItem) -> Void)?
    
    func makeNSView(context: Context) -> DrawingView {
        let view = DrawingView()
        view.rectangles = rectangles
        view.textBubbles = textBubbles
        view.pdfItems = pdfItems
        view.selectedColor = selectedColor
        view.selectedRectangle = selectedRectangle
        view.selectedTextBubble = selectedTextBubble
        view.selectedPDF = selectedPDF
        view.selectedTool = selectedTool
        view.delegate = context.coordinator
        view.onPDFDoubleClick = onPDFDoubleClick
        return view
    }
    
    func updateNSView(_ nsView: DrawingView, context: Context) {
        nsView.rectangles = rectangles
        nsView.textBubbles = textBubbles
        nsView.pdfItems = pdfItems
        nsView.selectedColor = selectedColor
        nsView.selectedRectangle = selectedRectangle
        nsView.selectedTextBubble = selectedTextBubble
        nsView.selectedPDF = selectedPDF
        nsView.selectedTool = selectedTool
        nsView.needsDisplay = true
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, DrawingViewDelegate {
        var parent: CanvasView
        
        init(_ parent: CanvasView) {
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
        
        func pdfItemsDidChange(_ pdfItems: [PDFItem]) {
            parent.pdfItems = pdfItems
        }
        
        func selectedPDFDidChange(_ pdfItem: PDFItem?) {
            parent.selectedPDF = pdfItem
        }
    }
}

protocol DrawingViewDelegate: AnyObject {
    func rectanglesDidChange(_ rectangles: [Rectangle])
    func selectedRectangleDidChange(_ rectangle: Rectangle?)
    func textBubblesDidChange(_ textBubbles: [TextBubble])
    func selectedTextBubbleDidChange(_ textBubble: TextBubble?)
    func pdfItemsDidChange(_ pdfItems: [PDFItem])
    func selectedPDFDidChange(_ pdfItem: PDFItem?)
}

class DrawingView: NSView {
    var rectangles: [Rectangle] = []
    var textBubbles: [TextBubble] = []
    var pdfItems: [PDFItem] = []
    var selectedColor: Color = .blue
    var selectedRectangle: Rectangle?
    var selectedTextBubble: TextBubble?
    var selectedPDF: PDFItem?
    var selectedTool: Tool = .select
    weak var delegate: DrawingViewDelegate?
    var onPDFDoubleClick: ((PDFItem) -> Void)?
    
    
    private var isDrawing = false
    private var drawingStart: CGPoint = .zero
    private var currentDrawingRect: CGRect = .zero
    
    private var isDragging = false
    private var dragOffset: CGSize = .zero
    private var lastClickTime: TimeInterval = 0
    private var lastClickedPDF: PDFItem?
    private var lastClickedTextBubble: TextBubble?
    private var textEditor: InlineTextEditor?
    private var editingTextBubble: TextBubble?
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupTrackingArea()
        setupDragAndDrop()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupTrackingArea()
        setupDragAndDrop()
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
    
    override var acceptsFirstResponder: Bool {
        return true
    }
    
    override func keyDown(with event: NSEvent) {
        if event.keyCode == 51 { // Delete key
            if let selected = selectedRectangle {
                rectangles.removeAll { $0.id == selected.id }
                selectedRectangle = nil
                delegate?.rectanglesDidChange(rectangles)
                delegate?.selectedRectangleDidChange(nil)
                needsDisplay = true
            } else if let selected = selectedTextBubble {
                textBubbles.removeAll { $0.id == selected.id }
                selectedTextBubble = nil
                delegate?.textBubblesDidChange(textBubbles)
                delegate?.selectedTextBubbleDidChange(nil)
                needsDisplay = true
            } else if let selected = selectedPDF {
                pdfItems.removeAll { $0.id == selected.id }
                selectedPDF = nil
                delegate?.pdfItemsDidChange(pdfItems)
                delegate?.selectedPDFDidChange(nil)
                needsDisplay = true
            }
        } else {
            super.keyDown(with: event)
        }
    }
    
    private func setupDragAndDrop() {
        registerForDraggedTypes([.fileURL])
    }
    
    
    override func mouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        
        let currentTime = Date().timeIntervalSince1970
        let isDoubleClick = (currentTime - lastClickTime) < 0.5
        
        switch selectedTool {
        case .select:
            if let pdfItem = pdfItems.first(where: { $0.contains(point: point) }) {
                if isDoubleClick && lastClickedPDF?.id == pdfItem.id {
                    onPDFDoubleClick?(pdfItem)
                    lastClickTime = 0
                    lastClickedPDF = nil
                    return
                }
                
                lastClickTime = currentTime
                lastClickedPDF = pdfItem
                var updatedPDFs = pdfItems
                var updatedRectangles = rectangles
                for i in 0..<updatedPDFs.count {
                    updatedPDFs[i].isSelected = updatedPDFs[i].id == pdfItem.id
                }
                for i in 0..<updatedRectangles.count {
                    updatedRectangles[i].isSelected = false
                }
                pdfItems = updatedPDFs
                rectangles = updatedRectangles
                selectedPDF = pdfItem
                selectedRectangle = nil
                delegate?.selectedPDFDidChange(pdfItem)
                delegate?.selectedRectangleDidChange(nil)
                delegate?.pdfItemsDidChange(pdfItems)
                delegate?.rectanglesDidChange(rectangles)
                
                isDragging = true
                dragOffset = CGSize(
                    width: point.x - pdfItem.origin.x,
                    height: point.y - pdfItem.origin.y
                )
            } else if let rectangle = rectangles.first(where: { $0.contains(point: point) }) {
                lastClickedPDF = nil
                var updatedRectangles = rectangles
                var updatedTextBubbles = textBubbles
                var updatedPDFs = pdfItems
                for i in 0..<updatedRectangles.count {
                    updatedRectangles[i].isSelected = updatedRectangles[i].id == rectangle.id
                }
                for i in 0..<updatedTextBubbles.count {
                    updatedTextBubbles[i].isSelected = false
                }
                for i in 0..<updatedPDFs.count {
                    updatedPDFs[i].isSelected = false
                }
                rectangles = updatedRectangles
                textBubbles = updatedTextBubbles
                pdfItems = updatedPDFs
                selectedRectangle = rectangle
                selectedTextBubble = nil
                selectedPDF = nil
                delegate?.selectedRectangleDidChange(rectangle)
                delegate?.selectedTextBubbleDidChange(nil)
                delegate?.selectedPDFDidChange(nil)
                delegate?.rectanglesDidChange(rectangles)
                delegate?.textBubblesDidChange(textBubbles)
                delegate?.pdfItemsDidChange(pdfItems)
                
                isDragging = true
                dragOffset = CGSize(
                    width: point.x - rectangle.origin.x,
                    height: point.y - rectangle.origin.y
                )
            } else if let textBubble = textBubbles.first(where: { $0.contains(point: point) }) {
                lastClickedPDF = nil
                
                // Check for double-click
                if (currentTime - lastClickTime) < 0.5 && lastClickedTextBubble?.id == textBubble.id {
                    startEditingTextBubble(textBubble)
                    lastClickTime = 0
                    lastClickedTextBubble = nil
                    return
                }
                
                lastClickTime = currentTime
                lastClickedTextBubble = textBubble
                
                var updatedRectangles = rectangles
                var updatedTextBubbles = textBubbles
                var updatedPDFs = pdfItems
                for i in 0..<updatedRectangles.count {
                    updatedRectangles[i].isSelected = false
                }
                for i in 0..<updatedTextBubbles.count {
                    updatedTextBubbles[i].isSelected = updatedTextBubbles[i].id == textBubble.id
                }
                for i in 0..<updatedPDFs.count {
                    updatedPDFs[i].isSelected = false
                }
                rectangles = updatedRectangles
                textBubbles = updatedTextBubbles
                pdfItems = updatedPDFs
                selectedRectangle = nil
                selectedTextBubble = textBubble
                selectedPDF = nil
                delegate?.selectedRectangleDidChange(nil)
                delegate?.selectedTextBubbleDidChange(textBubble)
                delegate?.selectedPDFDidChange(nil)
                delegate?.rectanglesDidChange(rectangles)
                delegate?.textBubblesDidChange(textBubbles)
                delegate?.pdfItemsDidChange(pdfItems)
                
                isDragging = true
                dragOffset = CGSize(
                    width: point.x - textBubble.origin.x,
                    height: point.y - textBubble.origin.y
                )
            } else {
                lastClickedPDF = nil
                lastClickedTextBubble = nil
                finishEditingTextBubble()
                
                var updatedRectangles = rectangles
                var updatedTextBubbles = textBubbles
                var updatedPDFs = pdfItems
                for i in 0..<updatedRectangles.count {
                    updatedRectangles[i].isSelected = false
                }
                for i in 0..<updatedTextBubbles.count {
                    updatedTextBubbles[i].isSelected = false
                }
                for i in 0..<updatedPDFs.count {
                    updatedPDFs[i].isSelected = false
                }
                rectangles = updatedRectangles
                textBubbles = updatedTextBubbles
                pdfItems = updatedPDFs
                selectedRectangle = nil
                selectedTextBubble = nil
                selectedPDF = nil
                delegate?.selectedRectangleDidChange(nil)
                delegate?.selectedTextBubbleDidChange(nil)
                delegate?.selectedPDFDidChange(nil)
                delegate?.rectanglesDidChange(rectangles)
                delegate?.textBubblesDidChange(textBubbles)
                delegate?.pdfItemsDidChange(pdfItems)
            }
            
        case .rectangle:
            finishEditingTextBubble()
            var updatedRectangles = rectangles
            var updatedTextBubbles = textBubbles
            var updatedPDFs = pdfItems
            for i in 0..<updatedRectangles.count {
                updatedRectangles[i].isSelected = false
            }
            for i in 0..<updatedTextBubbles.count {
                updatedTextBubbles[i].isSelected = false
            }
            for i in 0..<updatedPDFs.count {
                updatedPDFs[i].isSelected = false
            }
            rectangles = updatedRectangles
            textBubbles = updatedTextBubbles
            pdfItems = updatedPDFs
            selectedRectangle = nil
            selectedTextBubble = nil
            selectedPDF = nil
            delegate?.selectedRectangleDidChange(nil)
            delegate?.selectedTextBubbleDidChange(nil)
            delegate?.selectedPDFDidChange(nil)
            delegate?.rectanglesDidChange(rectangles)
            delegate?.textBubblesDidChange(textBubbles)
            delegate?.pdfItemsDidChange(pdfItems)
            
            isDrawing = true
            drawingStart = point
            currentDrawingRect = CGRect(origin: point, size: .zero)
            
        case .text:
            finishEditingTextBubble()
            // Deselect everything
            var updatedRectangles = rectangles
            var updatedTextBubbles = textBubbles
            var updatedPDFs = pdfItems
            for i in 0..<updatedRectangles.count {
                updatedRectangles[i].isSelected = false
            }
            for i in 0..<updatedTextBubbles.count {
                updatedTextBubbles[i].isSelected = false
            }
            for i in 0..<updatedPDFs.count {
                updatedPDFs[i].isSelected = false
            }
            rectangles = updatedRectangles
            textBubbles = updatedTextBubbles
            pdfItems = updatedPDFs
            selectedRectangle = nil
            selectedTextBubble = nil
            selectedPDF = nil
            delegate?.selectedRectangleDidChange(nil)
            delegate?.selectedTextBubbleDidChange(nil)
            delegate?.selectedPDFDidChange(nil)
            delegate?.rectanglesDidChange(rectangles)
            delegate?.textBubblesDidChange(textBubbles)
            delegate?.pdfItemsDidChange(pdfItems)
            
            // Create text bubble with default text
            let newTextBubble = TextBubble(
                origin: point,
                text: "Text",
                color: selectedColor
            )
            textBubbles.append(newTextBubble)
            delegate?.textBubblesDidChange(textBubbles)
            
            // Immediately start editing the new text bubble
            startEditingTextBubble(newTextBubble)
            needsDisplay = true
        }
        
        needsDisplay = true
    }
    
    override func mouseDragged(with event: NSEvent) {
        let currentPoint = convert(event.locationInWindow, from: nil)
        
        if isDragging {
            let newOrigin = CGPoint(
                x: currentPoint.x - dragOffset.width,
                y: currentPoint.y - dragOffset.height
            )
            
            if let selectedRect = selectedRectangle {
                var updatedRectangles = rectangles
                for i in 0..<updatedRectangles.count {
                    if updatedRectangles[i].id == selectedRect.id {
                        updatedRectangles[i].origin = newOrigin
                        selectedRectangle = updatedRectangles[i]
                        break
                    }
                }
                rectangles = updatedRectangles
                delegate?.rectanglesDidChange(rectangles)
                delegate?.selectedRectangleDidChange(selectedRectangle)
            } else if let selectedTextBubbleItem = selectedTextBubble {
                var updatedTextBubbles = textBubbles
                for i in 0..<updatedTextBubbles.count {
                    if updatedTextBubbles[i].id == selectedTextBubbleItem.id {
                        updatedTextBubbles[i].origin = newOrigin
                        selectedTextBubble = updatedTextBubbles[i]
                        break
                    }
                }
                textBubbles = updatedTextBubbles
                delegate?.textBubblesDidChange(textBubbles)
                delegate?.selectedTextBubbleDidChange(selectedTextBubble)
            } else if let selectedPDFItem = selectedPDF {
                var updatedPDFs = pdfItems
                for i in 0..<updatedPDFs.count {
                    if updatedPDFs[i].id == selectedPDFItem.id {
                        updatedPDFs[i].origin = newOrigin
                        selectedPDF = updatedPDFs[i]
                        break
                    }
                }
                pdfItems = updatedPDFs
                delegate?.pdfItemsDidChange(pdfItems)
                delegate?.selectedPDFDidChange(selectedPDF)
            }
            needsDisplay = true
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
            needsDisplay = true
        }
    }
    
    override func mouseUp(with event: NSEvent) {
        if isDragging {
            isDragging = false
            dragOffset = .zero
        } else if isDrawing {
            isDrawing = false
            
            if currentDrawingRect.size.width > 5 && currentDrawingRect.size.height > 5 {
                let newRectangle = Rectangle(
                    origin: currentDrawingRect.origin,
                    size: currentDrawingRect.size,
                    color: selectedColor
                )
                rectangles.append(newRectangle)
                delegate?.rectanglesDidChange(rectangles)
            }
            
            currentDrawingRect = .zero
        }
        needsDisplay = true
    }
    
    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        return .copy
    }
    
    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        guard let items = sender.draggingPasteboard.readObjects(forClasses: [NSURL.self], options: nil) as? [URL] else {
            return false
        }
        
        let dropPoint = convert(sender.draggingLocation, from: nil)
        
        for fileURL in items {
            if fileURL.pathExtension.lowercased() == "pdf" {
                if let pdfDocument = PDFDocument(url: fileURL) {
                    let pdfItem = PDFItem(
                        origin: dropPoint,
                        size: CGSize(width: 200, height: 300),
                        pdfDocument: pdfDocument
                    )
                    pdfItems.append(pdfItem)
                    delegate?.pdfItemsDidChange(pdfItems)
                }
            }
        }
        
        needsDisplay = true
        return true
    }
    
    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        
        guard let context = NSGraphicsContext.current?.cgContext else { return }
        
        context.setFillColor(NSColor.white.cgColor)
        context.fill(bounds)
        
        for rectangle in rectangles {
            drawRectangle(rectangle, in: context)
        }
        
        for textBubble in textBubbles {
            drawTextBubble(textBubble, in: context)
        }
        
        for pdfItem in pdfItems {
            drawPDFItem(pdfItem, in: context)
        }
        
        if isDrawing && currentDrawingRect.size.width > 0 && currentDrawingRect.size.height > 0 {
            context.setStrokeColor(NSColor(selectedColor).cgColor)
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
    
    private func drawPDFItem(_ pdfItem: PDFItem, in context: CGContext) {
        guard let page = pdfItem.pdfDocument.page(at: 0) else { return }
        
        let pageRect = page.bounds(for: .mediaBox)
        let scaleX = pdfItem.size.width / pageRect.width
        let scaleY = pdfItem.size.height / pageRect.height
        let scale = min(scaleX, scaleY)
        
        let scaledSize = CGSize(width: pageRect.width * scale, height: pageRect.height * scale)
        let centeredOrigin = CGPoint(
            x: pdfItem.origin.x + (pdfItem.size.width - scaledSize.width) / 2,
            y: pdfItem.origin.y + (pdfItem.size.height - scaledSize.height) / 2
        )
        
        context.saveGState()
        context.translateBy(x: centeredOrigin.x, y: centeredOrigin.y)
        context.scaleBy(x: scale, y: scale)
        
        page.draw(with: .mediaBox, to: context)
        context.restoreGState()
        
        if pdfItem.isSelected {
            let selectionPath = CGPath(roundedRect: pdfItem.frame.insetBy(dx: -2, dy: -2), 
                                     cornerWidth: pdfItem.cornerRadius + 2, 
                                     cornerHeight: pdfItem.cornerRadius + 2, 
                                     transform: nil)
            context.setStrokeColor(NSColor.systemBlue.cgColor)
            context.setLineWidth(3)
            context.setLineDash(phase: 0, lengths: [])
            context.addPath(selectionPath)
            context.strokePath()
        }
    }
    
    private func startEditingTextBubble(_ textBubble: TextBubble) {
        // Remove any existing text editor
        textEditor?.removeFromSuperview()
        
        // Size the editor to be slightly larger than the current text, but not huge
        let font = NSFont.systemFont(ofSize: textBubble.fontSize)
        let attributes: [NSAttributedString.Key: Any] = [.font: font]
        let textSize = (textBubble.text as NSString).size(withAttributes: attributes)
        
        // Add some padding for expansion, but keep it reasonable
        let padding: CGFloat = 20
        let editorWidth = max(textSize.width + padding, 60) // Minimum 60px
        let editorHeight = max(textSize.height + 8, 24) // Minimum 24px
        
        // Position editor to start where the centered text would begin, with fine-tuned offset
        let originalTextStartX = textBubble.origin.x + (textBubble.size.width - textSize.width) / 2
        let editorFrame = CGRect(
            x: originalTextStartX - 2,  // Fine-tuned X offset
            y: textBubble.origin.y + (textBubble.size.height - editorHeight) / 2 - 4,  // Fine-tuned Y offset
            width: editorWidth,
            height: editorHeight
        )
        
        // Create new text editor with appropriately sized frame
        let editor = InlineTextEditor(frame: editorFrame)
        editor.startEditing(with: textBubble.text, in: editorFrame, color: NSColor(textBubble.color))
        
        // Set up callbacks
        editor.onTextChanged = { [weak self] newText in
            self?.updateTextBubbleText(textBubble, newText: newText)
        }
        
        editor.onEditingFinished = { [weak self] in
            self?.finishEditingTextBubble()
        }
        
        // Add to view and store references
        addSubview(editor)
        textEditor = editor
        editingTextBubble = textBubble
        
        // Don't start dragging when editing
        isDragging = false
    }
    
    private func updateTextBubbleText(_ textBubble: TextBubble, newText: String) {
        guard let editingIndex = textBubbles.firstIndex(where: { $0.id == textBubble.id }) else { return }
        
        var updatedTextBubble = textBubbles[editingIndex]
        updatedTextBubble.text = newText
        textBubbles[editingIndex] = updatedTextBubble
        
        // Update the selected text bubble reference
        if selectedTextBubble?.id == textBubble.id {
            selectedTextBubble = updatedTextBubble
            delegate?.selectedTextBubbleDidChange(updatedTextBubble)
        }
        
        delegate?.textBubblesDidChange(textBubbles)
        
        // The text editor already has enough space, just redraw
        needsDisplay = true
    }
    
    private func finishEditingTextBubble() {
        textEditor?.removeFromSuperview()
        textEditor = nil
        editingTextBubble = nil
        needsDisplay = true
    }
}