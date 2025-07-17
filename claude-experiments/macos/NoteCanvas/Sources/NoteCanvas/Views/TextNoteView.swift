import AppKit

class TextNoteView: BaseNoteView, NoteViewProtocol {
    var note: TextNote {
        didSet {
            updateAppearance()
        }
    }
    
    private var textView: NSTextView!
    private var scrollView: NSScrollView!
    
    init(note: TextNote) {
        self.note = note
        super.init(frame: CGRect(origin: note.position, size: note.size))
        setupTextView()
        updateAppearance()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupTextView() {
        scrollView = NSScrollView(frame: bounds.insetBy(dx: 16, dy: 16))
        scrollView.autoresizingMask = [.width, .height]
        scrollView.borderType = .noBorder
        scrollView.hasVerticalScroller = true
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
    
    func updateAppearance() {
        textView.string = note.content
        textView.font = note.font.nsFont
        textView.textColor = note.textColor.nsColor
        textView.alignment = note.alignment.nsTextAlignment
        
        contentLayer.backgroundColor = NSColor.white.cgColor
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

extension TextNoteView: NSTextViewDelegate {
    func textDidChange(_ notification: Notification) {
        note.content = textView.string
        note.modifiedAt = Date()
    }
}