import AppKit
import Combine
import PDFKit

// MARK: - Page Navigation System

public protocol NavigationPage: AnyObject {
    var view: NSView { get }
    var title: String { get }
}

public class CanvasPage: NavigationPage {
    public let view: NSView
    public let title = "Canvas"
    
    public init(canvasView: CanvasView) {
        self.view = canvasView
    }
}

public class PDFPage: NavigationPage {
    public let view: NSView
    public let title: String
    
    private let pdfNote: PDFNote
    
    public init(pdfNote: PDFNote) {
        self.pdfNote = pdfNote
        self.title = pdfNote.metadata.title ?? "PDF Document"
        self.view = PDFStandaloneView(pdfNote: pdfNote)
    }
}

// MARK: - Page Navigation Controller

public class PageNavigationController: NSViewController {
    private var pages: [NavigationPage] = []
    private var currentPageIndex: Int = 0
    
    private var contentContainer: NSView!
    private var currentContentView: NSView?
    private var backButton: NSButton!
    
    public weak var delegate: PageNavigationDelegate?
    
    public override func loadView() {
        view = NSView()
        setupNavigationInterface()
    }
    
    private func setupNavigationInterface() {
        // Create content container that fills the entire view
        contentContainer = NSView()
        contentContainer.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(contentContainer)
        
        // Setup constraints for full-screen content
        NSLayoutConstraint.activate([
            contentContainer.topAnchor.constraint(equalTo: view.topAnchor),
            contentContainer.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            contentContainer.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            contentContainer.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
        
        // Create floating back button (will be added only when not on main canvas)
        backButton = NSButton()
        backButton.title = ""
        backButton.image = NSImage(systemSymbolName: "chevron.left.circle.fill", accessibilityDescription: "Back")
        backButton.imagePosition = .imageOnly
        backButton.isBordered = false
        backButton.target = self
        backButton.action = #selector(backButtonTapped)
        backButton.wantsLayer = true
        backButton.layer?.backgroundColor = NSColor.controlBackgroundColor.withAlphaComponent(0.9).cgColor
        backButton.layer?.cornerRadius = 20
        backButton.layer?.shadowColor = NSColor.black.cgColor
        backButton.layer?.shadowOpacity = 0.2
        backButton.layer?.shadowOffset = CGSize(width: 0, height: 2)
        backButton.layer?.shadowRadius = 4
        backButton.translatesAutoresizingMaskIntoConstraints = false
        backButton.isHidden = true // Start hidden
        
        // Add back button on top of content
        view.addSubview(backButton)
        
        NSLayoutConstraint.activate([
            backButton.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            backButton.topAnchor.constraint(equalTo: view.topAnchor, constant: 20),
            backButton.widthAnchor.constraint(equalToConstant: 40),
            backButton.heightAnchor.constraint(equalToConstant: 40)
        ])
    }
    
    public func setInitialPage(_ page: NavigationPage) {
        pages = [page]
        currentPageIndex = 0
        showCurrentPage()
        updateNavigationState()
    }
    
    public func pushPage(_ page: NavigationPage, animated: Bool = true) {
        pages.append(page)
        currentPageIndex = pages.count - 1
        
        if animated {
            simpleAnimateToCurrentPage()
        } else {
            showCurrentPage()
        }
        
        updateNavigationState()
        delegate?.pageNavigation(self, didNavigateToPage: currentPageIndex)
    }
    
    public func popPage(animated: Bool = true) {
        guard canGoBack() else { return }
        
        pages.removeLast()
        currentPageIndex = pages.count - 1
        
        if animated {
            simpleAnimateToCurrentPage()
        } else {
            showCurrentPage()
        }
        
        updateNavigationState()
        delegate?.pageNavigation(self, didNavigateToPage: currentPageIndex)
    }
    
    private func canGoBack() -> Bool {
        return pages.count > 1
    }
    
    @objc private func backButtonTapped() {
        popPage(animated: true)
    }
    
    private func simpleAnimateToCurrentPage() {
        guard currentPageIndex < pages.count else { return }
        
        let newPage = pages[currentPageIndex]
        let newView = newPage.view
        
        // Remove old view immediately
        currentContentView?.removeFromSuperview()
        
        // Set up new view with constraints
        currentContentView = newView
        newView.translatesAutoresizingMaskIntoConstraints = false
        contentContainer.addSubview(newView)
        
        NSLayoutConstraint.activate([
            newView.topAnchor.constraint(equalTo: contentContainer.topAnchor),
            newView.leadingAnchor.constraint(equalTo: contentContainer.leadingAnchor),
            newView.trailingAnchor.constraint(equalTo: contentContainer.trailingAnchor),
            newView.bottomAnchor.constraint(equalTo: contentContainer.bottomAnchor)
        ])
        
        // Simple fade in animation
        newView.alphaValue = 0
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.15  // Quick fade in
            context.timingFunction = CAMediaTimingFunction(name: .easeOut)
            newView.animator().alphaValue = 1
        })
    }
    
    private func showCurrentPage() {
        guard currentPageIndex < pages.count else { return }
        
        clearContent()
        
        let currentPage = pages[currentPageIndex]
        currentContentView = currentPage.view
        guard let contentView = currentContentView else { return }
        
        contentView.translatesAutoresizingMaskIntoConstraints = false
        contentContainer.addSubview(contentView)
        
        NSLayoutConstraint.activate([
            contentView.topAnchor.constraint(equalTo: contentContainer.topAnchor),
            contentView.leadingAnchor.constraint(equalTo: contentContainer.leadingAnchor),
            contentView.trailingAnchor.constraint(equalTo: contentContainer.trailingAnchor),
            contentView.bottomAnchor.constraint(equalTo: contentContainer.bottomAnchor)
        ])
    }
    
    private func clearContent() {
        currentContentView?.removeFromSuperview()
        currentContentView = nil
    }
    
    private func updateNavigationState() {
        guard currentPageIndex < pages.count else { return }
        
        // Show/hide back button based on whether we can go back
        backButton.isHidden = !canGoBack()
        
        // Update back button appearance
        if canGoBack() {
            backButton.contentTintColor = NSColor.controlAccentColor
        }
    }
}

public protocol PageNavigationDelegate: AnyObject {
    func pageNavigation(_ controller: PageNavigationController, didNavigateToPage pageIndex: Int)
}

// MARK: - PDF Standalone View (reused from tab system)

class PDFStandaloneView: NSView {
    private let pdfNote: PDFNote
    private var pdfView: PDFView!
    private var imagePDFRenderer: ImagePDFRenderer!
    private var highlighterPanel: NSView!
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
    private var currentTool: Tool = .highlightRed
    
    // Renderer mode
    private enum RendererMode {
        case pdfKit, imageRenderer
    }
    private var rendererMode: RendererMode = .imageRenderer
    
    init(pdfNote: PDFNote) {
        self.pdfNote = pdfNote
        super.init(frame: .zero)
        setupStandaloneView()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupStandaloneView() {
        wantsLayer = true
        // Use same background color as canvas for consistency
        layer?.backgroundColor = NSColor(red: 0.95, green: 0.95, blue: 0.97, alpha: 1.0).cgColor
        
        // Setup image renderer and floating panel
        setupImageRenderer()
        setupHighlighterPanel()
        
        // Set initial tool in the image renderer
        imagePDFRenderer.selectTool(convertTool(currentTool))
    }
    
    
    private func setupImageRenderer() {
        // Create more generous spacing for a cleaner, more focused look
        let margin: CGFloat = 40
        let rendererFrame = NSRect(
            x: margin,
            y: margin,
            width: bounds.width - (margin * 2),
            height: bounds.height - (margin * 2)
        )
        
        setupImageRenderer(frame: rendererFrame)
    }
    
    
    private func setupImageRenderer(frame: NSRect) {
        imagePDFRenderer = ImagePDFRenderer(pdfNote: pdfNote, frame: frame)
        imagePDFRenderer.autoresizingMask = [.width, .height]
        
        // No additional styling - should appear flat on canvas surface
        addSubview(imagePDFRenderer)
    }
    
    private func setupHighlighterPanel() {
        // Create floating highlighter panel with refined styling
        highlighterPanel = NSView()
        highlighterPanel.wantsLayer = true
        
        // More sophisticated background with blur effect feel
        highlighterPanel.layer?.backgroundColor = NSColor.controlBackgroundColor.withAlphaComponent(0.95).cgColor
        highlighterPanel.layer?.cornerRadius = 12
        highlighterPanel.layer?.masksToBounds = false // Allow shadow
        
        // Enhanced shadow for floating effect
        highlighterPanel.layer?.shadowColor = NSColor.black.cgColor
        highlighterPanel.layer?.shadowOpacity = 0.15
        highlighterPanel.layer?.shadowOffset = CGSize(width: 0, height: 4)
        highlighterPanel.layer?.shadowRadius = 8
        
        // Subtle border
        highlighterPanel.layer?.borderColor = NSColor.black.withAlphaComponent(0.1).cgColor
        highlighterPanel.layer?.borderWidth = 0.5
        
        addSubview(highlighterPanel)
        
        setupHighlighterButtons()
        updateToolButtonStates()
    }
    
    private func setupHighlighterButtons() {
        let buttonSpacing: CGFloat = 8
        var currentY: CGFloat = 12
        
        // Yellow highlight
        highlightYellowButton = createColorToolButton(
            title: "Yellow Highlight",
            color: .systemYellow,
            action: #selector(selectYellowHighlight)
        )
        highlightYellowButton.frame.origin = CGPoint(x: 12, y: currentY)
        highlighterPanel.addSubview(highlightYellowButton)
        currentY += highlightYellowButton.frame.height + buttonSpacing
        
        // Green highlight
        highlightGreenButton = createColorToolButton(
            title: "Green Highlight",
            color: .systemGreen,
            action: #selector(selectGreenHighlight)
        )
        highlightGreenButton.frame.origin = CGPoint(x: 12, y: currentY)
        highlighterPanel.addSubview(highlightGreenButton)
        currentY += highlightGreenButton.frame.height + buttonSpacing
        
        // Red highlight
        highlightRedButton = createColorToolButton(
            title: "Red Highlight",
            color: .systemRed,
            action: #selector(selectRedHighlight)
        )
        highlightRedButton.frame.origin = CGPoint(x: 12, y: currentY)
        highlighterPanel.addSubview(highlightRedButton)
        currentY += highlightRedButton.frame.height + buttonSpacing
        
        // Blue highlight
        highlightBlueButton = createColorToolButton(
            title: "Blue Highlight",
            color: .systemBlue,
            action: #selector(selectBlueHighlight)
        )
        highlightBlueButton.frame.origin = CGPoint(x: 12, y: currentY)
        highlighterPanel.addSubview(highlightBlueButton)
        currentY += highlightBlueButton.frame.height + buttonSpacing
        
        // Add separator line
        currentY += buttonSpacing
        let separator = NSView()
        separator.wantsLayer = true
        separator.layer?.backgroundColor = NSColor.separatorColor.cgColor
        separator.frame = NSRect(x: 8, y: currentY, width: 40, height: 1)
        highlighterPanel.addSubview(separator)
        currentY += 8
        
        // Draw button
        drawButton = createToolButton(
            title: "Draw",
            imageName: "pencil",
            action: #selector(selectDrawTool)
        )
        drawButton.frame.origin = CGPoint(x: 12, y: currentY)
        highlighterPanel.addSubview(drawButton)
        currentY += drawButton.frame.height + buttonSpacing
        
        // Eraser button
        eraserButton = createToolButton(
            title: "Eraser",
            imageName: "eraser.line.dashed",
            action: #selector(selectEraserTool)
        )
        eraserButton.frame.origin = CGPoint(x: 12, y: currentY)
        highlighterPanel.addSubview(eraserButton)
        currentY += eraserButton.frame.height + buttonSpacing
        
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
    
    private func createToolButton(title: String, imageName: String, action: Selector) -> NSButton {
        let button = NSButton()
        button.title = ""
        
        // Try to create system symbol, fallback to title if not available
        if let image = NSImage(systemSymbolName: imageName, accessibilityDescription: title) {
            button.image = image
            button.imagePosition = .imageOnly
        } else {
            // Fallback to single letter if system symbol not available
            let fallbackTitle = String(title.prefix(1))
            button.title = fallbackTitle
            button.imagePosition = .noImage
        }
        
        // Use borderless style and custom layer styling
        button.isBordered = false
        button.bezelStyle = .regularSquare
        button.target = self
        button.action = action
        button.toolTip = title
        button.wantsLayer = true
        
        // Custom styling similar to color buttons
        button.layer?.backgroundColor = NSColor.controlBackgroundColor.withAlphaComponent(0.8).cgColor
        button.layer?.cornerRadius = 6
        button.layer?.borderColor = NSColor.separatorColor.cgColor
        button.layer?.borderWidth = 1
        
        // Add subtle shadow for depth
        button.layer?.shadowColor = NSColor.black.cgColor
        button.layer?.shadowOpacity = 0.1
        button.layer?.shadowOffset = CGSize(width: 0, height: 1)
        button.layer?.shadowRadius = 2
        
        button.sizeToFit()
        let size = max(button.frame.width, button.frame.height, 32)
        button.frame.size = CGSize(width: size, height: size)
        
        return button
    }
    
    override func layout() {
        super.layout()
        
        // Create more generous spacing for a cleaner, more focused look
        let margin: CGFloat = 40
        let rendererFrame = NSRect(
            x: margin,
            y: margin,
            width: bounds.width - (margin * 2),
            height: bounds.height - (margin * 2)
        )
        imagePDFRenderer.frame = rendererFrame
        
        // Position refined floating panel with better spacing
        let panelWidth: CGFloat = 60
        let panelHeight: CGFloat = 280  // Increased to accommodate all buttons properly
        let panelMargin: CGFloat = 24
        let panelFrame = NSRect(
            x: rendererFrame.maxX - panelWidth - panelMargin,
            y: rendererFrame.minY + panelMargin,
            width: panelWidth,
            height: panelHeight
        )
        highlighterPanel.frame = panelFrame
    }
    
    // MARK: - Tool Action Methods
    
    @objc private func selectYellowHighlight() {
        currentTool = .highlightYellow
        updateToolButtonStates()
        imagePDFRenderer.selectTool(.highlightYellow)
    }
    
    @objc private func selectGreenHighlight() {
        currentTool = .highlightGreen
        updateToolButtonStates()
        imagePDFRenderer.selectTool(.highlightGreen)
    }
    
    @objc private func selectRedHighlight() {
        currentTool = .highlightRed
        updateToolButtonStates()
        imagePDFRenderer.selectTool(.highlightRed)
    }
    
    @objc private func selectBlueHighlight() {
        currentTool = .highlightBlue
        updateToolButtonStates()
        imagePDFRenderer.selectTool(.highlightBlue)
    }
    
    @objc private func selectDrawTool() {
        currentTool = .draw
        updateToolButtonStates()
        imagePDFRenderer.selectTool(.draw)
    }
    
    @objc private func selectEraserTool() {
        currentTool = .eraser
        updateToolButtonStates()
        imagePDFRenderer.selectTool(.eraser)
    }
    
    
    private func updateToolButtonStates() {
        // Reset all highlight button borders and shadows
        [highlightYellowButton, highlightGreenButton, highlightRedButton, highlightBlueButton].forEach { button in
            button?.layer?.borderWidth = 1
            button?.layer?.shadowOpacity = 0.2
            button?.layer?.shadowOffset = CGSize(width: 0, height: 1)
            button?.layer?.shadowRadius = 2
        }
        
        // Reset other button states to default styling
        drawButton.layer?.backgroundColor = NSColor.controlBackgroundColor.withAlphaComponent(0.8).cgColor
        drawButton.layer?.borderColor = NSColor.separatorColor.cgColor
        drawButton.layer?.borderWidth = 1
        drawButton.layer?.shadowOpacity = 0.1
        eraserButton.layer?.backgroundColor = NSColor.controlBackgroundColor.withAlphaComponent(0.8).cgColor
        eraserButton.layer?.borderColor = NSColor.separatorColor.cgColor
        eraserButton.layer?.borderWidth = 1
        eraserButton.layer?.shadowOpacity = 0.1
        
        // Highlight the active button with white border
        switch currentTool {
        case .highlightYellow:
            highlightYellowButton.layer?.borderWidth = 3
            highlightYellowButton.layer?.borderColor = NSColor.white.cgColor // White indicator
            highlightYellowButton.layer?.shadowOpacity = 0.4
            highlightYellowButton.layer?.shadowRadius = 4
        case .highlightGreen:
            highlightGreenButton.layer?.borderWidth = 3
            highlightGreenButton.layer?.borderColor = NSColor.white.cgColor // White indicator
            highlightGreenButton.layer?.shadowOpacity = 0.4
            highlightGreenButton.layer?.shadowRadius = 4
        case .highlightRed:
            highlightRedButton.layer?.borderWidth = 3
            highlightRedButton.layer?.borderColor = NSColor.white.cgColor // White indicator
            highlightRedButton.layer?.shadowOpacity = 0.4
            highlightRedButton.layer?.shadowRadius = 4
        case .highlightBlue:
            highlightBlueButton.layer?.borderWidth = 3
            highlightBlueButton.layer?.borderColor = NSColor.white.cgColor // White indicator
            highlightBlueButton.layer?.shadowOpacity = 0.4
            highlightBlueButton.layer?.shadowRadius = 4
        case .draw:
            drawButton.layer?.backgroundColor = NSColor.controlAccentColor.withAlphaComponent(0.3).cgColor
            drawButton.layer?.borderColor = NSColor.white.cgColor
            drawButton.layer?.borderWidth = 2
        case .eraser:
            eraserButton.layer?.backgroundColor = NSColor.controlAccentColor.withAlphaComponent(0.3).cgColor
            eraserButton.layer?.borderColor = NSColor.white.cgColor
            eraserButton.layer?.borderWidth = 2
        case .none:
            [highlightYellowButton, highlightGreenButton, highlightRedButton, highlightBlueButton].forEach { button in
                button?.layer?.borderColor = button?.layer?.backgroundColor
            }
        }
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
}