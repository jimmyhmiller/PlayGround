import Foundation
import AppKit

public class Canvas: ObservableObject {
    @Published public var notes: [AnyNote] = []
    @Published public var selectedNotes: Set<UUID> = []
    @Published public var viewportOffset: CGPoint = .zero
    @Published public var zoomLevel: CGFloat = 1.0
    
    private var nextZIndex = 0
    public let undoManager = UndoManager()
    
    public init() {
        undoManager.levelsOfUndo = 50
    }
    
    public func addNote(_ note: any NoteItem) {
        var mutableNote = note
        mutableNote.zIndex = nextZIndex
        nextZIndex += 1
        let anyNote = AnyNote(mutableNote)
        
        // Register undo action
        undoManager.registerUndo(withTarget: self) { canvas in
            canvas.removeNote(id: mutableNote.id)
        }
        undoManager.setActionName("Add Note")
        
        notes.append(anyNote)
    }
    
    func removeNote(id: UUID) {
        // Find the note before removing it for undo
        guard let noteToRemove = notes.first(where: { $0.note.id == id }) else { return }
        
        // Register undo action
        undoManager.registerUndo(withTarget: self) { canvas in
            canvas.addNote(noteToRemove.note)
        }
        undoManager.setActionName("Delete Note")
        
        notes.removeAll { $0.note.id == id }
        selectedNotes.remove(id)
    }
    
    func selectNote(id: UUID, exclusive: Bool = true) {
        if exclusive {
            selectedNotes.removeAll()
        }
        selectedNotes.insert(id)
        updateSelectionState()
    }
    
    func deselectNote(id: UUID) {
        selectedNotes.remove(id)
        updateSelectionState()
    }
    
    func clearSelection() {
        selectedNotes.removeAll()
        updateSelectionState()
    }
    
    public func deleteSelectedNotes() {
        let notesToDelete = notes.filter { selectedNotes.contains($0.note.id) }
        
        if notesToDelete.isEmpty { return }
        
        // Register undo action for all deleted notes
        undoManager.registerUndo(withTarget: self) { canvas in
            // Restore all deleted notes
            for anyNote in notesToDelete {
                canvas.notes.append(anyNote)
            }
            // Restore selection
            canvas.selectedNotes = Set(notesToDelete.map { $0.note.id })
            canvas.updateSelectionState()
        }
        undoManager.setActionName("Delete \(notesToDelete.count) Note\(notesToDelete.count == 1 ? "" : "s")")
        
        // Remove the notes
        notes.removeAll { selectedNotes.contains($0.note.id) }
        selectedNotes.removeAll()
    }
    
    func moveNote(id: UUID, to position: CGPoint) {
        guard let index = notes.firstIndex(where: { $0.note.id == id }) else { return }
        var note = notes[index].note
        let oldPosition = note.position
        
        // Register undo action
        undoManager.registerUndo(withTarget: self) { canvas in
            canvas.moveNote(id: id, to: oldPosition)
        }
        if !undoManager.isUndoing && !undoManager.isRedoing {
            undoManager.setActionName("Move Note")
        }
        
        note.position = position
        note.modifiedAt = Date()
        notes[index] = AnyNote(note)
    }
    
    func moveNotes(_ updates: [UUID: CGPoint]) {
        for (id, position) in updates {
            guard let index = notes.firstIndex(where: { $0.note.id == id }) else { continue }
            var note = notes[index].note
            note.position = position
            note.modifiedAt = Date()
            notes[index] = AnyNote(note)
        }
    }
    
    func resizeNote(id: UUID, to size: CGSize) {
        guard let index = notes.firstIndex(where: { $0.note.id == id }) else { return }
        var note = notes[index].note
        note.size = size
        note.modifiedAt = Date()
        notes[index] = AnyNote(note)
    }
    
    func bringToFront(id: UUID) {
        guard let index = notes.firstIndex(where: { $0.note.id == id }) else { return }
        var note = notes[index].note
        note.zIndex = nextZIndex
        nextZIndex += 1
        notes[index] = AnyNote(note)
    }
    
    private func updateSelectionState() {
        for i in 0..<notes.count {
            var note = notes[i].note
            note.isSelected = selectedNotes.contains(note.id)
            notes[i] = AnyNote(note)
        }
    }
    
    func notesInRect(_ rect: CGRect) -> [AnyNote] {
        notes.filter { note in
            let noteRect = CGRect(origin: note.note.position, size: note.note.size)
            return noteRect.intersects(rect)
        }
    }
}