import AppKit
import PDFKit

class PDFNoteView: BaseNoteView, NoteViewProtocol {
    var note: PDFNote {
        didSet {
            updateAppearance()
        }
    }
    
    private var thumbnailImageView: NSImageView!
    private var pageLabel: NSTextField!
    
    init(note: PDFNote) {
        self.note = note
        super.init(frame: CGRect(origin: note.position, size: note.size))
        setupViews()
        updateAppearance()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        // Hide the white background from BaseNoteView
        contentLayer.backgroundColor = NSColor.clear.cgColor
        
        // Setup thumbnail image view
        thumbnailImageView = NSImageView(frame: bounds)
        thumbnailImageView.autoresizingMask = [.width, .height]
        thumbnailImageView.imageFrameStyle = .none
        thumbnailImageView.imageScaling = .scaleProportionallyUpOrDown
        thumbnailImageView.wantsLayer = true
        thumbnailImageView.layer?.cornerRadius = note.cornerRadius
        thumbnailImageView.layer?.masksToBounds = true
        thumbnailImageView.layer?.backgroundColor = NSColor.white.cgColor
        
        // Add subtle border to make PDF pages look more like documents
        thumbnailImageView.layer?.borderColor = NSColor.separatorColor.cgColor
        thumbnailImageView.layer?.borderWidth = 1
        
        addSubview(thumbnailImageView)
        
        // Setup page label (small indicator showing it's a PDF)
        pageLabel = NSTextField(labelWithString: "PDF")
        pageLabel.font = NSFont.systemFont(ofSize: 10, weight: .medium)
        pageLabel.textColor = NSColor.secondaryLabelColor
        pageLabel.backgroundColor = NSColor.controlBackgroundColor.withAlphaComponent(0.8)
        pageLabel.wantsLayer = true
        pageLabel.layer?.cornerRadius = 4
        pageLabel.layer?.masksToBounds = true
        pageLabel.alignment = .center
        
        addSubview(pageLabel)
        
        // Ensure contentLayer also has rounded corners
        contentLayer.cornerRadius = note.cornerRadius
        contentLayer.masksToBounds = true
    }
    
    func updateAppearance() {
        contentLayer.cornerRadius = note.cornerRadius
        thumbnailImageView.layer?.cornerRadius = note.cornerRadius
        
        // Generate and display thumbnail
        if let thumbnailImage = note.generateThumbnailImage() {
            thumbnailImageView.image = thumbnailImage
        } else {
            // Fallback: show a generic PDF icon
            thumbnailImageView.image = NSImage(systemSymbolName: "doc.fill", accessibilityDescription: "PDF Document")
            thumbnailImageView.symbolConfiguration = NSImage.SymbolConfiguration(pointSize: 48, weight: .light)
            thumbnailImageView.contentTintColor = NSColor.systemBlue
        }
        
        // Update page label
        if let pdfDocument = PDFDocument(url: note.pdfPath) {
            let pageCount = pdfDocument.pageCount
            pageLabel.stringValue = pageCount > 1 ? "PDF (\(pageCount) pages)" : "PDF"
        } else {
            pageLabel.stringValue = "PDF"
        }
        
        layoutPageLabel()
    }
    
    private func layoutPageLabel() {
        pageLabel.sizeToFit()
        let labelSize = pageLabel.frame.size
        let padding: CGFloat = 4
        
        pageLabel.frame = CGRect(
            x: bounds.width - labelSize.width - padding - 8,
            y: bounds.height - labelSize.height - padding - 8,
            width: labelSize.width + padding * 2,
            height: labelSize.height + padding
        )
    }
    
    func updateSelection() {
        animateSelection(note.isSelected)
    }
    
    func startDragging(at point: NSPoint) {
        // Handled by BaseNoteView
    }
    
    func handleResize(edge: ResizeEdge, delta: CGSize) {
        // TODO: Implement resize functionality
    }
    
    override func getNoteItem() -> (any NoteItem)? {
        return note
    }
    
    override func layout() {
        super.layout()
        thumbnailImageView.frame = bounds
        layoutPageLabel()
    }
    
}

// MARK: - PDF Overlay Delegate
protocol PDFOverlayDelegate: AnyObject {
    func pdfOverlayDidRequestClose(_ overlay: PDFOverlayView)
}

// PDF Overlay View for inline PDF editing
class PDFOverlayView: NSView {
    private let pdfNote: PDFNote
    private var pdfView: PDFView!
    private var imagePDFRenderer: ImagePDFRenderer!
    private var toolbarView: NSView!
    private var highlighterPanel: NSView!
    private var closeButton: NSButton!
    private var titleLabel: NSTextField!
    private var rendererToggleButton: NSButton!
    
    // Tool buttons
    private var highlightYellowButton: NSButton!
    private var highlightGreenButton: NSButton!
    private var highlightRedButton: NSButton!
    private var highlightBlueButton: NSButton!
    private var drawButton: NSButton!
    private var eraserButton: NSButton!
    
    // Current tool state
    private enum Tool {
        case none, highlightYellow, highlightGreen, highlightRed, highlightBlue, draw, eraser
    }
    private var currentTool: Tool = .highlightRed // Start with red highlighting for debugging
    
    // Renderer mode
    private enum RendererMode {
        case pdfKit, imageRenderer
    }
    private var rendererMode: RendererMode = .imageRenderer // Default to image renderer for better performance
    
    // Drawing state (for PDFKit mode)
    private var isDrawing = false
    private var currentPath: NSBezierPath?
    private var drawingStartPoint: NSPoint = .zero
    private var lastDrawPoint: NSPoint = .zero
    private let minDrawDistance: CGFloat = 2.0  // Minimum distance between points
    
    weak var delegate: PDFOverlayDelegate?
    
    init(pdfNote: PDFNote, frame: NSRect) {
        self.pdfNote = pdfNote
        super.init(frame: frame)
        setupOverlay()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupOverlay() {
        wantsLayer = true
        layer?.backgroundColor = NSColor.black.withAlphaComponent(0.8).cgColor
        
        setupToolbar()
        setupRenderers()
        setupHighlighterPanel()
        setupCloseButton()
        updateRendererVisibility()
        
        // Set initial tool in the image renderer
        if rendererMode == .imageRenderer {
            imagePDFRenderer.selectTool(convertTool(currentTool))
        }
    }
    
    private func setupToolbar() {
        // Create toolbar background
        toolbarView = NSView()
        toolbarView.wantsLayer = true
        toolbarView.layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
        toolbarView.layer?.cornerRadius = 8
        toolbarView.layer?.masksToBounds = true
        
        // Add subtle shadow
        toolbarView.layer?.shadowColor = NSColor.black.cgColor
        toolbarView.layer?.shadowOpacity = 0.3
        toolbarView.layer?.shadowOffset = CGSize(width: 0, height: -2)
        toolbarView.layer?.shadowRadius = 4
        toolbarView.layer?.masksToBounds = false
        
        addSubview(toolbarView)
        
        // Title label
        titleLabel = NSTextField(labelWithString: pdfNote.metadata.title ?? "PDF Document")
        titleLabel.font = NSFont.systemFont(ofSize: 16, weight: .medium)
        titleLabel.textColor = NSColor.labelColor
        toolbarView.addSubview(titleLabel)
        
        setupToolbarButtons()
    }
    
    private func setupToolbarButtons() {
        let buttonSpacing: CGFloat = 12
        var currentX: CGFloat = 16
        
        // Draw button
        drawButton = createToolButton(
            title: "Draw",
            imageName: "pencil",
            action: #selector(selectDrawTool)
        )
        drawButton.frame.origin = CGPoint(x: currentX, y: 8)
        toolbarView.addSubview(drawButton)
        currentX += drawButton.frame.width + buttonSpacing
        
        // Eraser button
        eraserButton = createToolButton(
            title: "Eraser",
            imageName: "eraser",
            action: #selector(selectEraserTool)
        )
        eraserButton.frame.origin = CGPoint(x: currentX, y: 8)
        toolbarView.addSubview(eraserButton)
        currentX += eraserButton.frame.width + buttonSpacing
        
        // Renderer toggle button
        rendererToggleButton = createToolButton(
            title: "Switch Renderer",
            imageName: "rectangle.and.pencil.and.ellipsis",
            action: #selector(toggleRenderer)
        )
        rendererToggleButton.frame.origin = CGPoint(x: currentX, y: 8)
        toolbarView.addSubview(rendererToggleButton)
    }
    
    private func createToolButton(title: String, imageName: String, action: Selector) -> NSButton {
        let button = NSButton()
        button.title = ""
        button.image = NSImage(systemSymbolName: imageName, accessibilityDescription: title)
        button.imagePosition = .imageOnly
        button.isBordered = true
        button.bezelStyle = .regularSquare
        button.target = self
        button.action = action
        button.toolTip = title
        button.sizeToFit()
        
        // Make button square
        let size = max(button.frame.width, button.frame.height, 32)
        button.frame.size = CGSize(width: size, height: size)
        
        return button
    }
    
    private func createColorToolButton(title: String, color: NSColor, action: Selector) -> NSButton {
        let button = NSButton()
        button.title = ""
        button.bezelStyle = .regularSquare
        button.isBordered = false
        button.target = self
        button.action = action
        button.toolTip = title
        button.isEnabled = true
        button.wantsLayer = true
        
        // Create a rounded rectangle appearance like a highlight swatch
        button.layer?.backgroundColor = color.withAlphaComponent(0.8).cgColor
        button.layer?.cornerRadius = 6
        button.layer?.borderWidth = 1
        button.layer?.borderColor = color.withAlphaComponent(0.9).cgColor
        
        // Add subtle shadow for depth
        button.layer?.shadowColor = NSColor.black.cgColor
        button.layer?.shadowOpacity = 0.2
        button.layer?.shadowOffset = CGSize(width: 0, height: 1)
        button.layer?.shadowRadius = 2
        
        // Set fixed size for color buttons
        button.frame.size = CGSize(width: 32, height: 32)
        
        return button
    }
    
    private func setupCloseButton() {
        closeButton = NSButton()
        closeButton.title = ""
        closeButton.image = NSImage(systemSymbolName: "xmark.circle.fill", accessibilityDescription: "Close PDF viewer")
        closeButton.imagePosition = .imageOnly
        closeButton.isBordered = false
        closeButton.target = self
        closeButton.action = #selector(closePDFViewer)
        closeButton.sizeToFit()
        
        // Style the close button
        closeButton.layer?.backgroundColor = NSColor.systemRed.cgColor
        closeButton.layer?.cornerRadius = closeButton.frame.width / 2
        closeButton.contentTintColor = NSColor.white
        
        addSubview(closeButton)
    }
    
    private func setupRenderers() {
        let rendererFrame = NSRect(
            x: 40,
            y: 40,
            width: bounds.width - 80,
            height: bounds.height - 120
        )
        
        // Setup PDFKit renderer
        setupPDFView(frame: rendererFrame)
        
        // Setup Image renderer
        setupImageRenderer(frame: rendererFrame)
    }
    
    private func setupPDFView(frame: NSRect) {
        pdfView = PDFView(frame: frame)
        pdfView.autoresizingMask = [.width, .height]
        pdfView.autoScales = true
        pdfView.displayMode = .singlePageContinuous
        pdfView.displayDirection = .vertical
        pdfView.wantsLayer = true
        pdfView.layer?.cornerRadius = 12
        pdfView.layer?.masksToBounds = true
        pdfView.layer?.borderColor = NSColor.separatorColor.cgColor
        pdfView.layer?.borderWidth = 1
        
        // Load the PDF document
        if let pdfDocument = PDFDocument(url: pdfNote.pdfPath) {
            pdfView.document = pdfDocument
            pdfView.go(to: pdfDocument.page(at: pdfNote.currentPage) ?? pdfDocument.page(at: 0)!)
        }
        
        // Set up notification for selection changes
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(pdfSelectionChanged),
            name: .PDFViewSelectionChanged,
            object: pdfView
        )
        
        addSubview(pdfView)
    }
    
    private func setupImageRenderer(frame: NSRect) {
        imagePDFRenderer = ImagePDFRenderer(pdfNote: pdfNote, frame: frame)
        imagePDFRenderer.autoresizingMask = [.width, .height]
        addSubview(imagePDFRenderer)
    }
    
    private func setupHighlighterPanel() {
        // Create floating highlighter panel
        highlighterPanel = NSView()
        highlighterPanel.wantsLayer = true
        highlighterPanel.layer?.backgroundColor = NSColor.controlBackgroundColor.withAlphaComponent(0.95).cgColor
        highlighterPanel.layer?.cornerRadius = 8
        
        // Add subtle shadow (but keep masksToBounds true for interaction)
        highlighterPanel.layer?.shadowColor = NSColor.black.cgColor
        highlighterPanel.layer?.shadowOpacity = 0.3
        highlighterPanel.layer?.shadowOffset = CGSize(width: -2, height: 0)
        highlighterPanel.layer?.shadowRadius = 6
        highlighterPanel.layer?.masksToBounds = true
        
        addSubview(highlighterPanel)
        
        setupHighlighterButtons()
        
        // Set initial button states to show red as selected
        updateToolButtonStates()
    }
    
    private func setupHighlighterButtons() {
        let buttonSpacing: CGFloat = 8
        var currentY: CGFloat = 12
        
        // Yellow highlight button
        highlightYellowButton = createColorToolButton(
            title: "Yellow Highlight",
            color: .systemYellow,
            action: #selector(selectYellowHighlight)
        )
        highlightYellowButton.frame.origin = CGPoint(x: 12, y: currentY)
        highlighterPanel.addSubview(highlightYellowButton)
        currentY += highlightYellowButton.frame.height + buttonSpacing
        
        // Green highlight button
        highlightGreenButton = createColorToolButton(
            title: "Green Highlight",
            color: .systemGreen,
            action: #selector(selectGreenHighlight)
        )
        highlightGreenButton.frame.origin = CGPoint(x: 12, y: currentY)
        highlighterPanel.addSubview(highlightGreenButton)
        currentY += highlightGreenButton.frame.height + buttonSpacing
        
        // Red highlight button
        highlightRedButton = createColorToolButton(
            title: "Red Highlight",
            color: .systemRed,
            action: #selector(selectRedHighlight)
        )
        highlightRedButton.frame.origin = CGPoint(x: 12, y: currentY)
        highlighterPanel.addSubview(highlightRedButton)
        currentY += highlightRedButton.frame.height + buttonSpacing
        
        // Blue highlight button
        highlightBlueButton = createColorToolButton(
            title: "Blue Highlight",
            color: .systemBlue,
            action: #selector(selectBlueHighlight)
        )
        highlightBlueButton.frame.origin = CGPoint(x: 12, y: currentY)
        highlighterPanel.addSubview(highlightBlueButton)
    }
    
    override func layout() {
        super.layout()
        
        // Position toolbar at top
        let toolbarHeight: CGFloat = 48
        toolbarView.frame = NSRect(
            x: 40,
            y: bounds.height - toolbarHeight - 20,
            width: bounds.width - 80,
            height: toolbarHeight
        )
        
        // Position title in toolbar
        titleLabel.sizeToFit()
        titleLabel.frame.origin = CGPoint(
            x: toolbarView.bounds.width - titleLabel.frame.width - 16,
            y: (toolbarView.bounds.height - titleLabel.frame.height) / 2
        )
        
        // Position close button
        closeButton.frame.origin = CGPoint(
            x: bounds.width - closeButton.frame.width - 20,
            y: bounds.height - closeButton.frame.height - 20
        )
        
        // Update renderer frames
        let rendererFrame = NSRect(
            x: 40,
            y: 40,
            width: bounds.width - 80,
            height: bounds.height - toolbarHeight - 80
        )
        pdfView.frame = rendererFrame
        imagePDFRenderer.frame = rendererFrame
        
        // Position highlighter panel on the right side, inside the PDF content area
        let panelWidth: CGFloat = 56 // Increased width to accommodate button frames + padding
        let panelHeight: CGFloat = 180 // Increased height to ensure all buttons fit
        let panelFrame = NSRect(
            x: rendererFrame.maxX - panelWidth - 20, // Position relative to PDF content area
            y: rendererFrame.minY + 20, // Position at top of PDF content area
            width: panelWidth,
            height: panelHeight
        )
        highlighterPanel.frame = panelFrame
    }
    
    @objc private func selectYellowHighlight() {
        currentTool = .highlightYellow
        updateToolButtonStates()
        
        if rendererMode == .imageRenderer {
            imagePDFRenderer.selectTool(.highlightYellow)
        } else {
            pdfView.currentSelection = nil
        }
    }
    
    @objc private func selectGreenHighlight() {
        currentTool = .highlightGreen
        updateToolButtonStates()
        
        if rendererMode == .imageRenderer {
            imagePDFRenderer.selectTool(.highlightGreen)
        } else {
            pdfView.currentSelection = nil
        }
    }
    
    @objc private func selectRedHighlight() {
        currentTool = .highlightRed
        updateToolButtonStates()
        
        if rendererMode == .imageRenderer {
            imagePDFRenderer.selectTool(.highlightRed)
        } else {
            pdfView.currentSelection = nil
        }
    }
    
    @objc private func selectBlueHighlight() {
        currentTool = .highlightBlue
        updateToolButtonStates()
        
        if rendererMode == .imageRenderer {
            imagePDFRenderer.selectTool(.highlightBlue)
        } else {
            pdfView.currentSelection = nil
        }
    }
    
    @objc private func selectDrawTool() {
        currentTool = .draw
        updateToolButtonStates()
        
        if rendererMode == .imageRenderer {
            imagePDFRenderer.selectTool(.draw)
        }
        print("Draw tool selected")
    }
    
    @objc private func selectEraserTool() {
        currentTool = .eraser
        updateToolButtonStates()
        
        if rendererMode == .imageRenderer {
            imagePDFRenderer.selectTool(.eraser)
        }
        print("Eraser tool selected")
    }
    
    @objc private func toggleRenderer() {
        rendererMode = (rendererMode == .pdfKit) ? .imageRenderer : .pdfKit
        updateRendererVisibility()
        updateToolButtonStates()
        
        // Update the active tool in the new renderer
        if rendererMode == .imageRenderer {
            imagePDFRenderer.selectTool(convertTool(currentTool))
        }
        
        print("Switched to \(rendererMode == .pdfKit ? "PDFKit" : "Image") renderer")
    }
    
    private func updateToolButtonStates() {
        // Reset all highlight button borders and shadows
        highlightYellowButton.layer?.borderWidth = 1
        highlightGreenButton.layer?.borderWidth = 1
        highlightRedButton.layer?.borderWidth = 1
        highlightBlueButton.layer?.borderWidth = 1
        
        // Reset shadow properties
        [highlightYellowButton, highlightGreenButton, highlightRedButton, highlightBlueButton].forEach { button in
            button?.layer?.shadowOpacity = 0.2
            button?.layer?.shadowOffset = CGSize(width: 0, height: 1)
            button?.layer?.shadowRadius = 2
        }
        
        // Reset other button states
        drawButton.layer?.backgroundColor = NSColor.clear.cgColor
        eraserButton.layer?.backgroundColor = NSColor.clear.cgColor
        rendererToggleButton.layer?.backgroundColor = NSColor.clear.cgColor
        
        // Highlight the active button with enhanced visual feedback
        switch currentTool {
        case .highlightYellow:
            highlightYellowButton.layer?.borderWidth = 3
            highlightYellowButton.layer?.borderColor = NSColor.controlAccentColor.cgColor
            highlightYellowButton.layer?.shadowOpacity = 0.4
            highlightYellowButton.layer?.shadowRadius = 4
        case .highlightGreen:
            highlightGreenButton.layer?.borderWidth = 3
            highlightGreenButton.layer?.borderColor = NSColor.controlAccentColor.cgColor
            highlightGreenButton.layer?.shadowOpacity = 0.4
            highlightGreenButton.layer?.shadowRadius = 4
        case .highlightRed:
            highlightRedButton.layer?.borderWidth = 3
            highlightRedButton.layer?.borderColor = NSColor.controlAccentColor.cgColor
            highlightRedButton.layer?.shadowOpacity = 0.4
            highlightRedButton.layer?.shadowRadius = 4
        case .highlightBlue:
            highlightBlueButton.layer?.borderWidth = 3
            highlightBlueButton.layer?.borderColor = NSColor.controlAccentColor.cgColor
            highlightBlueButton.layer?.shadowOpacity = 0.4
            highlightBlueButton.layer?.shadowRadius = 4
        case .draw:
            drawButton.layer?.backgroundColor = NSColor.controlAccentColor.withAlphaComponent(0.3).cgColor
        case .eraser:
            eraserButton.layer?.backgroundColor = NSColor.controlAccentColor.withAlphaComponent(0.3).cgColor
        case .none:
            // Reset all highlight buttons to their original border colors
            highlightYellowButton.layer?.borderColor = NSColor.systemYellow.withAlphaComponent(0.9).cgColor
            highlightGreenButton.layer?.borderColor = NSColor.systemGreen.withAlphaComponent(0.9).cgColor
            highlightRedButton.layer?.borderColor = NSColor.systemRed.withAlphaComponent(0.9).cgColor
            highlightBlueButton.layer?.borderColor = NSColor.systemBlue.withAlphaComponent(0.9).cgColor
        }
        
        // Highlight the renderer toggle if using image renderer
        if rendererMode == .imageRenderer {
            rendererToggleButton.layer?.backgroundColor = NSColor.systemBlue.withAlphaComponent(0.3).cgColor
        }
    }
    
    private func updateRendererVisibility() {
        pdfView.isHidden = (rendererMode != .pdfKit)
        imagePDFRenderer.isHidden = (rendererMode != .imageRenderer)
    }
    
    private func convertTool(_ tool: Tool) -> ImagePDFRenderer.Tool {
        switch tool {
        case .none: return .none
        case .highlightYellow: return .highlightYellow
        case .highlightGreen: return .highlightGreen
        case .highlightRed: return .highlightRed
        case .highlightBlue: return .highlightBlue
        case .draw: return .draw
        case .eraser: return .eraser
        }
    }
    
    @objc private func pdfSelectionChanged() {
        let isHighlightTool = [.highlightYellow, .highlightGreen, .highlightRed, .highlightBlue].contains(currentTool)
        guard isHighlightTool,
              let selection = pdfView.currentSelection,
              let page = selection.pages.first,
              let selectionString = selection.string,
              !selectionString.isEmpty else {
            return
        }
        
        // Only highlight if there's actual text selected
        let selectionBounds = selection.bounds(for: page)
        guard selectionBounds.width > 0 && selectionBounds.height > 0 else {
            return
        }
        
        // Delay the highlighting to allow selection to complete
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            guard let self = self,
                  [.highlightYellow, .highlightGreen, .highlightRed, .highlightBlue].contains(self.currentTool),
                  let currentSelection = self.pdfView.currentSelection,
                  currentSelection.string == selectionString else {
                return
            }
            
            // Check if this area is already highlighted to prevent layering
            let existingAnnotations = page.annotations.filter { annotation in
                annotation.type == "Highlight" && annotation.bounds.intersects(selectionBounds)
            }
            
            if existingAnnotations.isEmpty {
                // Create a highlight annotation only if none exists in this area
                let highlight = PDFAnnotation(bounds: selectionBounds, forType: .highlight, withProperties: nil)
                
                // Set color based on current tool
                switch currentTool {
                case .highlightYellow:
                    highlight.color = NSColor.systemYellow.withAlphaComponent(0.4)
                case .highlightGreen:
                    highlight.color = NSColor.systemGreen.withAlphaComponent(0.4)
                case .highlightRed:
                    highlight.color = NSColor.systemRed.withAlphaComponent(0.4)
                case .highlightBlue:
                    highlight.color = NSColor.systemBlue.withAlphaComponent(0.4)
                default:
                    highlight.color = NSColor.systemYellow.withAlphaComponent(0.4)
                }
                
                // Add the annotation to the page
                page.addAnnotation(highlight)
                print("Text highlighted!")
            }
            
            // Clear the selection after highlighting
            self.pdfView.currentSelection = nil
        }
    }
    
    @objc private func closePDFViewer() {
        delegate?.pdfOverlayDidRequestClose(self)
    }
    
    override func keyDown(with event: NSEvent) {
        // Close on Escape key
        if event.keyCode == 53 { // Escape key
            closePDFViewer()
        } else {
            super.keyDown(with: event)
        }
    }
    
    override var acceptsFirstResponder: Bool { 
        return true 
    }
    
    override func hitTest(_ point: NSPoint) -> NSView? {
        // Always check highlighter panel first - it should always be clickable
        if highlighterPanel.frame.contains(point) {
            let convertedPoint = convert(point, to: highlighterPanel)
            let result = highlighterPanel.hitTest(convertedPoint)
            
            
            // Try direct hit testing on each button as fallback
            if highlightYellowButton.frame.contains(convertedPoint) {
                return highlightYellowButton
            } else if highlightGreenButton.frame.contains(convertedPoint) {
                return highlightGreenButton
            } else if highlightRedButton.frame.contains(convertedPoint) {
                return highlightRedButton
            } else if highlightBlueButton.frame.contains(convertedPoint) {
                return highlightBlueButton
            }
            
            return result
        }
        
        // If using image renderer, let it handle its own events
        if rendererMode == .imageRenderer && imagePDFRenderer.frame.contains(point) {
            return imagePDFRenderer.hitTest(convert(point, to: imagePDFRenderer))
        }
        
        // If we're in draw or eraser mode with PDFKit and the hit is within the PDF area,
        // return ourselves to intercept the mouse events
        if rendererMode == .pdfKit && (currentTool == .draw || currentTool == .eraser) && pdfView.frame.contains(point) {
            return self
        }
        
        return super.hitTest(point)
    }
    
    override func mouseDown(with event: NSEvent) {
        // Only handle PDFKit events - image renderer handles its own
        guard rendererMode == .pdfKit else {
            super.mouseDown(with: event)
            return
        }
        
        let locationInSelf = convert(event.locationInWindow, from: nil)
        
        // Check if click is in PDF view area
        if pdfView.frame.contains(locationInSelf) {
            let locationInPDF = convert(locationInSelf, to: pdfView)
            
            if [.highlightYellow, .highlightGreen, .highlightRed, .highlightBlue].contains(currentTool) {
                // For highlighting, pass the event to the PDF view
                let pdfEvent = NSEvent.mouseEvent(with: event.type,
                                                location: locationInPDF,
                                                modifierFlags: event.modifierFlags,
                                                timestamp: event.timestamp,
                                                windowNumber: event.windowNumber,
                                                context: nil,
                                                eventNumber: event.eventNumber,
                                                clickCount: event.clickCount,
                                                pressure: event.pressure)
                pdfView.mouseDown(with: pdfEvent!)
                return
            } else if currentTool == .draw {
                // For drawing, handle the event ourselves - don't pass to PDFView
                guard let page = pdfView.page(for: locationInPDF, nearest: true) else {
                    return
                }
                
                let locationInPage = pdfView.convert(locationInPDF, to: page)
                
                isDrawing = true
                drawingStartPoint = locationInPage
                lastDrawPoint = locationInPage
                currentPath = NSBezierPath()
                currentPath?.move(to: locationInPage)
                return
            } else if currentTool == .eraser {
                guard let page = pdfView.page(for: locationInPDF, nearest: true) else {
                    return
                }
                
                let locationInPage = pdfView.convert(locationInPDF, to: page)
                
                // Find annotations at this location and remove them
                let annotationsToRemove = page.annotations.filter { annotation in
                    annotation.bounds.contains(locationInPage) && 
                    (annotation.type == "Highlight" || annotation.type == "Ink")
                }
                
                for annotation in annotationsToRemove {
                    page.removeAnnotation(annotation)
                }
                return
            }
        }
        
        // For other cases, pass to super
        super.mouseDown(with: event)
    }
    
    override func mouseDragged(with event: NSEvent) {
        // Only handle PDFKit events - image renderer handles its own
        guard rendererMode == .pdfKit else {
            super.mouseDragged(with: event)
            return
        }
        
        let locationInSelf = convert(event.locationInWindow, from: nil)
        
        // For highlighting, pass drag events to PDF view
        if [.highlightYellow, .highlightGreen, .highlightRed, .highlightBlue].contains(currentTool) && pdfView.frame.contains(locationInSelf) {
            let pdfEvent = NSEvent.mouseEvent(with: event.type,
                                            location: convert(locationInSelf, to: pdfView),
                                            modifierFlags: event.modifierFlags,
                                            timestamp: event.timestamp,
                                            windowNumber: event.windowNumber,
                                            context: nil,
                                            eventNumber: event.eventNumber,
                                            clickCount: event.clickCount,
                                            pressure: event.pressure)
            pdfView.mouseDragged(with: pdfEvent!)
            return
        }
        
        // For drawing - handle ourselves, don't pass to PDFView
        if currentTool == .draw && isDrawing {
            guard let path = currentPath else {
                return
            }
            
            let locationInPDF = convert(locationInSelf, to: pdfView)
            
            guard let page = pdfView.page(for: locationInPDF, nearest: true) else { 
                return 
            }
            
            let locationInPage = pdfView.convert(locationInPDF, to: page)
            
            // Only add point if it's far enough from the last point (performance optimization)
            let distance = sqrt(pow(locationInPage.x - lastDrawPoint.x, 2) + pow(locationInPage.y - lastDrawPoint.y, 2))
            if distance >= minDrawDistance {
                path.line(to: locationInPage)
                lastDrawPoint = locationInPage
            }
            return
        }
        
        // For other cases
        super.mouseDragged(with: event)
    }
    
    override func mouseUp(with event: NSEvent) {
        // Only handle PDFKit events - image renderer handles its own
        guard rendererMode == .pdfKit else {
            super.mouseUp(with: event)
            return
        }
        
        let locationInSelf = convert(event.locationInWindow, from: nil)
        
        // For highlighting, pass mouseUp events to PDF view
        if [.highlightYellow, .highlightGreen, .highlightRed, .highlightBlue].contains(currentTool) && pdfView.frame.contains(locationInSelf) {
            let pdfEvent = NSEvent.mouseEvent(with: event.type,
                                            location: convert(locationInSelf, to: pdfView),
                                            modifierFlags: event.modifierFlags,
                                            timestamp: event.timestamp,
                                            windowNumber: event.windowNumber,
                                            context: nil,
                                            eventNumber: event.eventNumber,
                                            clickCount: event.clickCount,
                                            pressure: event.pressure)
            pdfView.mouseUp(with: pdfEvent!)
            return
        }
        
        // For drawing - handle ourselves
        if currentTool == .draw && isDrawing {
            defer {
                // Always reset drawing state
                isDrawing = false
                currentPath = nil
            }
            
            guard let path = currentPath else {
                return
            }
            
            // Create an ink annotation from the path
            let bounds = path.bounds.insetBy(dx: -2, dy: -2)
            
            // Only create annotation if we actually drew something meaningful
            if !bounds.isEmpty && bounds.width > 3 && bounds.height > 3 {
                let locationInPDF = convert(locationInSelf, to: pdfView)
                
                guard let page = pdfView.page(for: locationInPDF, nearest: true) else { 
                    return 
                }
                
                let ink = PDFAnnotation(bounds: bounds, forType: .ink, withProperties: nil)
                ink.color = NSColor.systemRed
                ink.border = PDFBorder()
                ink.border?.lineWidth = 2.0
                
                // Add the path to the ink annotation
                ink.add(path)
                
                // Add the annotation to the page
                page.addAnnotation(ink)
            }
            return
        }
        
        // For other cases
        super.mouseUp(with: event)
    }
}