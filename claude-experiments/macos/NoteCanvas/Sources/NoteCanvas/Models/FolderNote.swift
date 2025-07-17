import Foundation
import AppKit

struct FolderNote: NoteItem {
    let id: UUID
    var position: CGPoint
    var size: CGSize
    var zIndex: Int
    var isSelected: Bool
    let createdAt: Date
    var modifiedAt: Date
    var metadata: NoteMetadata
    
    var items: [AnyNote]
    var isExpanded: Bool
    var folderColor: CodableColor
    var gridColumns: Int
    
    init(id: UUID = UUID(),
         position: CGPoint = .zero,
         size: CGSize = CGSize(width: 300, height: 300),
         items: [AnyNote] = [],
         isExpanded: Bool = true,
         folderColor: NSColor = .systemBlue,
         gridColumns: Int = 3) {
        self.id = id
        self.position = position
        self.size = size
        self.zIndex = 0
        self.isSelected = false
        self.createdAt = Date()
        self.modifiedAt = Date()
        self.metadata = NoteMetadata()
        self.items = items
        self.isExpanded = isExpanded
        self.folderColor = CodableColor(color: folderColor)
        self.gridColumns = gridColumns
    }
}

public struct AnyNote: Codable, Hashable {
    private let wrapped: any NoteItem
    
    public init<T: NoteItem>(_ note: T) {
        self.wrapped = note
    }
    
    public var note: any NoteItem {
        wrapped
    }
    
    enum CodingKeys: String, CodingKey {
        case type, data
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(NoteType.self, forKey: .type)
        
        switch type {
        case .text:
            self.wrapped = try container.decode(TextNote.self, forKey: .data)
        case .image:
            self.wrapped = try container.decode(ImageNote.self, forKey: .data)
        case .sticky:
            self.wrapped = try container.decode(StickyNote.self, forKey: .data)
        case .folder:
            self.wrapped = try container.decode(FolderNote.self, forKey: .data)
        case .pdf, .file:
            fatalError("Not implemented yet")
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        switch wrapped {
        case let note as TextNote:
            try container.encode(NoteType.text, forKey: .type)
            try container.encode(note, forKey: .data)
        case let note as ImageNote:
            try container.encode(NoteType.image, forKey: .type)
            try container.encode(note, forKey: .data)
        case let note as StickyNote:
            try container.encode(NoteType.sticky, forKey: .type)
            try container.encode(note, forKey: .data)
        case let note as FolderNote:
            try container.encode(NoteType.folder, forKey: .type)
            try container.encode(note, forKey: .data)
        default:
            fatalError("Unknown note type")
        }
    }
    
    public static func == (lhs: AnyNote, rhs: AnyNote) -> Bool {
        lhs.wrapped.id == rhs.wrapped.id
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(wrapped.id)
    }
}