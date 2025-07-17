import AppKit

class StickyNoteView: BaseNoteView, NoteViewProtocol {
    var note: StickyNote {
        didSet {
            updateAppearance()
        }
    }
    
    private var textView: NSTextView!
    private var scrollView: NSScrollView!
    
    init(note: StickyNote) {
        self.note = note
        super.init(frame: CGRect(origin: note.position, size: note.size))
        setupTextView()
        updateAppearance()
        customizeStickyAppearance()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupTextView() {
        scrollView = NSScrollView(frame: bounds.insetBy(dx: 12, dy: 12))
        scrollView.autoresizingMask = [.width, .height]
        scrollView.borderType = .noBorder
        scrollView.hasVerticalScroller = false
        scrollView.hasHorizontalScroller = false
        scrollView.backgroundColor = .clear
        scrollView.drawsBackground = false
        
        textView = NSTextView(frame: scrollView.bounds)
        textView.autoresizingMask = [.width, .height]
        textView.backgroundColor = .clear
        textView.isRichText = false
        textView.importsGraphics = false
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.textContainer?.containerSize = CGSize(width: scrollView.bounds.width, height: CGFloat.greatestFiniteMagnitude)
        textView.textContainer?.widthTracksTextView = true
        textView.delegate = self
        
        scrollView.documentView = textView
        addSubview(scrollView)
    }
    
    private func customizeStickyAppearance() {
        shadowLayer.shadowOpacity = 0.2
        shadowLayer.shadowRadius = 4
        shadowLayer.shadowOffset = CGSize(width: 0, height: -1)
        
        contentLayer.cornerRadius = 2
    }
    
    func updateAppearance() {
        textView.string = note.content
        textView.font = NSFont(name: "Marker Felt", size: 16) ?? NSFont.systemFont(ofSize: 16)
        textView.textColor = note.stickyColor.textColor
        textView.alignment = .left
        
        contentLayer.backgroundColor = note.stickyColor.color.cgColor
        
        addTapeEffect()
    }
    
    private func addTapeEffect() {
        layer?.sublayers?.filter { $0.name == "tape" }.forEach { $0.removeFromSuperlayer() }
        
        let tapeLayer = CALayer()
        tapeLayer.name = "tape"
        tapeLayer.frame = CGRect(x: bounds.width / 2 - 30, y: -10, width: 60, height: 20)
        tapeLayer.backgroundColor = NSColor(white: 0.95, alpha: 0.8).cgColor
        tapeLayer.cornerRadius = 2
        
        let tapeGradient = CAGradientLayer()
        tapeGradient.frame = tapeLayer.bounds
        tapeGradient.colors = [
            NSColor(white: 1.0, alpha: 0.6).cgColor,
            NSColor(white: 0.9, alpha: 0.6).cgColor
        ]
        tapeGradient.locations = [0.0, 1.0]
        tapeGradient.cornerRadius = 2
        tapeLayer.addSublayer(tapeGradient)
        
        VisualEffects.applyShadow(
            to: tapeLayer,
            opacity: 0.1,
            offset: CGSize(width: 0, height: 1),
            radius: 2
        )
        
        layer?.addSublayer(tapeLayer)
    }
    
    func updateSelection() {
        animateSelection(note.isSelected)
    }
    
    func startDragging(at point: NSPoint) {
    }
    
    func handleResize(edge: ResizeEdge, delta: CGSize) {
    }
    
    override func getNoteItem() -> (any NoteItem)? {
        return note
    }
}

extension StickyNoteView: NSTextViewDelegate {
    func textDidChange(_ notification: Notification) {
        note.content = textView.string
        note.modifiedAt = Date()
    }
}