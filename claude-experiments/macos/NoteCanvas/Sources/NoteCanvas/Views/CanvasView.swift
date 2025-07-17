import AppKit
import Combine

public class CanvasView: NSView {
    var canvas: Canvas
    private var noteViews: [UUID: NSView] = [:]
    private var cancellables = Set<AnyCancellable>()
    
    private var contentView: NSView!
    private var selectionView: SelectionView!
    
    private var isSpacePressed = false
    private var isPanning = false
    private var panStartPoint: NSPoint = .zero
    private var panStartOffset: CGPoint = .zero
    
    private var isSelecting = false
    private var selectionStartPoint: NSPoint = .zero
    
    private var isDraggingNote = false
    private var draggedNoteId: UUID?
    private var dragOffset: CGPoint = .zero
    private var selectedNotesStartPositions: [UUID: CGPoint] = [:]
    
    public init(canvas: Canvas) {
        self.canvas = canvas
        super.init(frame: .zero)
        setupView()
        observeCanvas()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupView() {
        wantsLayer = true
        layer?.backgroundColor = NSColor(red: 0.95, green: 0.95, blue: 0.97, alpha: 1.0).cgColor
        
        contentView = NSView(frame: bounds)
        contentView.wantsLayer = true
        addSubview(contentView)
        
        selectionView = SelectionView(frame: .zero)
        addSubview(selectionView)
        
        setupGestureRecognizers()
    }
    
    private func setupGestureRecognizers() {
        let magnification = NSMagnificationGestureRecognizer(target: self, action: #selector(handleMagnification(_:)))
        addGestureRecognizer(magnification)
    }
    
    private func observeCanvas() {
        canvas.$notes
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.updateNoteViews()
            }
            .store(in: &cancellables)
        
        canvas.$selectedNotes
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.updateSelectionStates()
            }
            .store(in: &cancellables)
        
        canvas.$viewportOffset
            .receive(on: DispatchQueue.main)
            .sink { [weak self] offset in
                self?.updateViewport()
            }
            .store(in: &cancellables)
        
        canvas.$zoomLevel
            .receive(on: DispatchQueue.main)
            .sink { [weak self] zoom in
                self?.updateZoom()
            }
            .store(in: &cancellables)
    }
    
    private func updateNoteViews() {
        let currentNoteIds = Set(canvas.notes.map { $0.note.id })
        let viewIds = Set(noteViews.keys)
        
        let toRemove = viewIds.subtracting(currentNoteIds)
        for id in toRemove {
            noteViews[id]?.removeFromSuperview()
            noteViews.removeValue(forKey: id)
        }
        
        let toAdd = currentNoteIds.subtracting(viewIds)
        for id in toAdd {
            if let note = canvas.notes.first(where: { $0.note.id == id }) {
                let view = createView(for: note)
                noteViews[id] = view
                contentView.addSubview(view)
            }
        }
        
        for note in canvas.notes {
            if let view = noteViews[note.note.id] {
                updateView(view, for: note)
            }
        }
        
        sortViewsByZIndex()
    }
    
    private func createView(for anyNote: AnyNote) -> NSView {
        let note = anyNote.note
        
        switch note {
        case let textNote as TextNote:
            let view = TextNoteView(note: textNote)
            view.delegate = self
            return view
        case let imageNote as ImageNote:
            let view = ImageNoteView(note: imageNote)
            view.delegate = self
            return view
        case let stickyNote as StickyNote:
            let view = StickyNoteView(note: stickyNote)
            view.delegate = self
            return view
        default:
            return NSView()
        }
    }
    
    private func updateView(_ view: NSView, for anyNote: AnyNote) {
        let note = anyNote.note
        let frame = CGRect(origin: note.position, size: note.size)
        
        if view.frame != frame {
            if isDraggingNote && canvas.selectedNotes.contains(note.id) {
                // Immediate update during dragging for all selected notes
                view.frame = frame
            } else {
                // Animated update for other changes
                NSAnimationContext.runAnimationGroup({ context in
                    context.duration = 0.2
                    context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                    view.animator().frame = frame
                })
            }
        }
    }
    
    private func sortViewsByZIndex() {
        let sortedNotes = canvas.notes.sorted { $0.note.zIndex < $1.note.zIndex }
        for note in sortedNotes {
            if let view = noteViews[note.note.id] {
                view.removeFromSuperview()
                contentView.addSubview(view)
            }
        }
    }
    
    private func updateSelectionStates() {
        for (id, view) in noteViews {
            if let baseView = view as? BaseNoteView {
                baseView.animateSelection(canvas.selectedNotes.contains(id))
            }
        }
    }
    
    private func updateViewport() {
        contentView.frame.origin = CGPoint(
            x: canvas.viewportOffset.x,
            y: canvas.viewportOffset.y
        )
    }
    
    private func updateZoom() {
        contentView.layer?.transform = CATransform3DMakeScale(canvas.zoomLevel, canvas.zoomLevel, 1)
    }
    
    @objc private func handleMagnification(_ gesture: NSMagnificationGestureRecognizer) {
        switch gesture.state {
        case .changed:
            let newZoom = canvas.zoomLevel * (1 + gesture.magnification)
            canvas.zoomLevel = max(0.25, min(4.0, newZoom))
            gesture.magnification = 0
        default:
            break
        }
    }
    
    public override func keyDown(with event: NSEvent) {
        if event.keyCode == 49 {
            isSpacePressed = true
            NSCursor.openHand.set()
        } else {
            super.keyDown(with: event)
        }
    }
    
    public override func keyUp(with event: NSEvent) {
        if event.keyCode == 49 {
            isSpacePressed = false
            isPanning = false
            NSCursor.arrow.set()
        } else {
            super.keyUp(with: event)
        }
    }
    
    public override func mouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        
        if isSpacePressed {
            isPanning = true
            panStartPoint = point
            panStartOffset = canvas.viewportOffset
            NSCursor.closedHand.set()
            return
        }
        
        let contentPoint = contentView.convert(point, from: self)
        
        // Check if we hit a note using data model hit testing
        for anyNote in canvas.notes.reversed() { // Check from front to back (z-order)
            let note = anyNote.note
            let noteRect = CGRect(origin: note.position, size: note.size)
            if noteRect.contains(contentPoint) {
                // If clicking on a non-selected note, select it
                if !canvas.selectedNotes.contains(note.id) {
                    canvas.selectNote(id: note.id, exclusive: !event.modifierFlags.contains(.shift))
                }
                
                // Start dragging - store initial positions of all selected notes
                isDraggingNote = true
                draggedNoteId = note.id
                dragOffset = CGPoint(
                    x: contentPoint.x - note.position.x,
                    y: contentPoint.y - note.position.y
                )
                
                // Store starting positions for all selected notes
                selectedNotesStartPositions.removeAll()
                for selectedId in canvas.selectedNotes {
                    if let selectedNote = canvas.notes.first(where: { $0.note.id == selectedId }) {
                        selectedNotesStartPositions[selectedId] = selectedNote.note.position
                    }
                }
                
                canvas.bringToFront(id: note.id)
                return
            }
        }
        
        // No note hit, start canvas selection
        canvas.clearSelection()
        isSelecting = true
        selectionStartPoint = point
        selectionView.startSelection(at: point)
    }
    
    public override func mouseDragged(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        
        if isPanning {
            let deltaX = point.x - panStartPoint.x
            let deltaY = point.y - panStartPoint.y
            canvas.viewportOffset = CGPoint(
                x: panStartOffset.x + deltaX,
                y: panStartOffset.y + deltaY
            )
        } else if isDraggingNote, let noteId = draggedNoteId {
            let contentPoint = contentView.convert(point, from: self)
            let newPosition = CGPoint(
                x: contentPoint.x - dragOffset.x,
                y: contentPoint.y - dragOffset.y
            )
            
            // Calculate the delta from the dragged note's starting position
            if let startPosition = selectedNotesStartPositions[noteId] {
                let deltaX = newPosition.x - startPosition.x
                let deltaY = newPosition.y - startPosition.y
                
                // Move all selected notes by the same delta
                for selectedId in canvas.selectedNotes {
                    if let startPos = selectedNotesStartPositions[selectedId] {
                        let newPos = CGPoint(
                            x: startPos.x + deltaX,
                            y: startPos.y + deltaY
                        )
                        canvas.moveNote(id: selectedId, to: newPos)
                    }
                }
            }
        } else if isSelecting {
            selectionView.updateSelection(to: point)
            
            let selectionRect = CGRect(
                x: min(selectionStartPoint.x, point.x),
                y: min(selectionStartPoint.y, point.y),
                width: abs(point.x - selectionStartPoint.x),
                height: abs(point.y - selectionStartPoint.y)
            )
            
            let contentRect = contentView.convert(selectionRect, from: self)
            let selectedNotes = canvas.notesInRect(contentRect)
            
            canvas.selectedNotes = Set(selectedNotes.map { $0.note.id })
        }
    }
    
    public override func mouseUp(with event: NSEvent) {
        if isPanning {
            isPanning = false
            NSCursor.openHand.set()
        } else if isDraggingNote {
            isDraggingNote = false
            draggedNoteId = nil
            dragOffset = .zero
        } else if isSelecting {
            isSelecting = false
            selectionView.endSelection()
        }
    }
    
    public override var acceptsFirstResponder: Bool { true }
}

extension CanvasView: NoteViewDelegate {
    func noteViewDidBeginDragging(_ view: NSView, note: any NoteItem) {
        canvas.bringToFront(id: note.id)
    }
    
    func noteViewDidDrag(_ view: NSView, note: any NoteItem, to point: CGPoint) {
        canvas.moveNote(id: note.id, to: point)
    }
    
    func noteViewDidEndDragging(_ view: NSView, note: any NoteItem) {
    }
    
    func noteViewDidResize(_ view: NSView, note: any NoteItem, to size: CGSize) {
        canvas.resizeNote(id: note.id, to: size)
    }
    
    func noteViewDidSelect(_ view: NSView, note: any NoteItem, event: NSEvent) {
        let exclusive = !event.modifierFlags.contains(.shift)
        canvas.selectNote(id: note.id, exclusive: exclusive)
    }
    
    func noteViewDidRequestContextMenu(_ view: NSView, note: any NoteItem, at point: NSPoint) {
    }
}

class SelectionView: NSView {
    private var selectionLayer: CAShapeLayer!
    private var startPoint: NSPoint = .zero
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupLayer()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupLayer()
    }
    
    private func setupLayer() {
        wantsLayer = true
        
        selectionLayer = CAShapeLayer()
        selectionLayer.fillColor = NSColor.controlAccentColor.withAlphaComponent(0.1).cgColor
        selectionLayer.strokeColor = NSColor.controlAccentColor.cgColor
        selectionLayer.lineWidth = 1
        selectionLayer.lineDashPattern = [5, 3]
        layer?.addSublayer(selectionLayer)
    }
    
    func startSelection(at point: NSPoint) {
        startPoint = point
        isHidden = false
    }
    
    func updateSelection(to point: NSPoint) {
        let rect = CGRect(
            x: min(startPoint.x, point.x),
            y: min(startPoint.y, point.y),
            width: abs(point.x - startPoint.x),
            height: abs(point.y - startPoint.y)
        )
        
        selectionLayer.path = CGPath(rect: rect, transform: nil)
    }
    
    func endSelection() {
        isHidden = true
        selectionLayer.path = nil
    }
}