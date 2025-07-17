import Foundation
import AppKit

public class Canvas: ObservableObject {
    @Published public var notes: [AnyNote] = []
    @Published public var selectedNotes: Set<UUID> = []
    @Published public var viewportOffset: CGPoint = .zero
    @Published public var zoomLevel: CGFloat = 1.0
    
    private var nextZIndex = 0
    
    public init() {}
    
    public func addNote(_ note: any NoteItem) {
        var mutableNote = note
        mutableNote.zIndex = nextZIndex
        nextZIndex += 1
        notes.append(AnyNote(mutableNote))
    }
    
    func removeNote(id: UUID) {
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
    
    func moveNote(id: UUID, to position: CGPoint) {
        guard let index = notes.firstIndex(where: { $0.note.id == id }) else { return }
        var note = notes[index].note
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