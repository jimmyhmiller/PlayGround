import AppKit

protocol NoteViewProtocol: NSView {
    associatedtype Note: NoteItem
    
    var note: Note { get set }
    var delegate: NoteViewDelegate? { get set }
    
    func updateAppearance()
    func updateSelection()
    func startDragging(at point: NSPoint)
    func handleResize(edge: ResizeEdge, delta: CGSize)
}

protocol NoteViewDelegate: AnyObject {
    func noteViewDidBeginDragging(_ view: NSView, note: any NoteItem)
    func noteViewDidDrag(_ view: NSView, note: any NoteItem, to point: CGPoint)
    func noteViewDidEndDragging(_ view: NSView, note: any NoteItem)
    func noteViewDidResize(_ view: NSView, note: any NoteItem, to size: CGSize)
    func noteViewDidSelect(_ view: NSView, note: any NoteItem, event: NSEvent)
    func noteViewDidRequestContextMenu(_ view: NSView, note: any NoteItem, at point: NSPoint)
}

enum ResizeEdge {
    case topLeft, top, topRight
    case left, right
    case bottomLeft, bottom, bottomRight
}

class BaseNoteView: NSView {
    weak var delegate: NoteViewDelegate?
    
    var shadowLayer: CALayer!
    var contentLayer: CALayer!
    var selectionLayer: CALayer!
    
    private var isDragging = false
    private var dragStartPoint: NSPoint = .zero
    private var dragStartPosition: CGPoint = .zero
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupLayers()
        setupTrackingArea()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupLayers()
        setupTrackingArea()
    }
    
    private func setupLayers() {
        wantsLayer = true
        
        shadowLayer = CALayer()
        shadowLayer.shadowColor = NSColor.black.cgColor
        shadowLayer.shadowOpacity = 0.15
        shadowLayer.shadowOffset = CGSize(width: 0, height: -2)
        shadowLayer.shadowRadius = 8
        layer?.addSublayer(shadowLayer)
        
        contentLayer = CALayer()
        contentLayer.backgroundColor = NSColor.white.cgColor
        contentLayer.cornerRadius = 12
        contentLayer.masksToBounds = true
        layer?.addSublayer(contentLayer)
        
        selectionLayer = CALayer()
        selectionLayer.borderColor = NSColor.controlAccentColor.cgColor
        selectionLayer.borderWidth = 0
        selectionLayer.cornerRadius = 14
        selectionLayer.isHidden = true
        layer?.addSublayer(selectionLayer)
    }
    
    private func setupTrackingArea() {
        let trackingArea = NSTrackingArea(
            rect: bounds,
            options: [.mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect],
            owner: self,
            userInfo: nil
        )
        addTrackingArea(trackingArea)
    }
    
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        trackingAreas.forEach { removeTrackingArea($0) }
        setupTrackingArea()
    }
    
    override func layout() {
        super.layout()
        
        let bounds = self.bounds
        shadowLayer.frame = bounds
        contentLayer.frame = bounds
        selectionLayer.frame = bounds.insetBy(dx: -2, dy: -2)
    }
    
    func animateSelection(_ selected: Bool) {
        CATransaction.begin()
        CATransaction.setAnimationDuration(0.2)
        
        if selected {
            selectionLayer.isHidden = false
            selectionLayer.borderWidth = 3
        } else {
            selectionLayer.borderWidth = 0
            selectionLayer.isHidden = true
        }
        
        CATransaction.commit()
    }
    
    override func mouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        isDragging = true
        dragStartPoint = point
        
        if let noteItem = getNoteItem() {
            dragStartPosition = noteItem.position
            delegate?.noteViewDidSelect(self, note: noteItem, event: event)
            delegate?.noteViewDidBeginDragging(self, note: noteItem)
        }
    }
    
    override func mouseDragged(with event: NSEvent) {
        guard isDragging, let noteItem = getNoteItem() else { return }
        
        let currentPoint = convert(event.locationInWindow, from: nil)
        let deltaX = currentPoint.x - dragStartPoint.x
        let deltaY = currentPoint.y - dragStartPoint.y
        
        let newPosition = CGPoint(
            x: dragStartPosition.x + deltaX,
            y: dragStartPosition.y + deltaY
        )
        
        delegate?.noteViewDidDrag(self, note: noteItem, to: newPosition)
    }
    
    override func mouseUp(with event: NSEvent) {
        guard isDragging, let noteItem = getNoteItem() else { return }
        isDragging = false
        delegate?.noteViewDidEndDragging(self, note: noteItem)
    }
    
    override func rightMouseDown(with event: NSEvent) {
        guard let noteItem = getNoteItem() else { return }
        let point = convert(event.locationInWindow, from: nil)
        delegate?.noteViewDidRequestContextMenu(self, note: noteItem, at: point)
    }
    
    func getNoteItem() -> (any NoteItem)? {
        nil
    }
}