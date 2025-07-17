import Foundation
import AppKit

public struct StickyNote: NoteItem {
    public let id: UUID
    public var position: CGPoint
    public var size: CGSize
    public var zIndex: Int
    public var isSelected: Bool
    public let createdAt: Date
    public var modifiedAt: Date
    public var metadata: NoteMetadata
    
    public var content: String
    public var stickyColor: StickyColor
    
    public init(id: UUID = UUID(),
         position: CGPoint = .zero,
         size: CGSize = CGSize(width: 180, height: 180),
         content: String = "",
         stickyColor: StickyColor = .yellow) {
        self.id = id
        self.position = position
        self.size = size
        self.zIndex = 0
        self.isSelected = false
        self.createdAt = Date()
        self.modifiedAt = Date()
        self.metadata = NoteMetadata()
        self.content = content
        self.stickyColor = stickyColor
    }
}

public enum StickyColor: String, Codable, CaseIterable {
    case yellow
    case pink
    case green
    case blue
    case orange
    case purple
    
    public var color: NSColor {
        switch self {
        case .yellow: return NSColor(red: 1.0, green: 0.96, blue: 0.56, alpha: 1.0)
        case .pink: return NSColor(red: 1.0, green: 0.76, blue: 0.82, alpha: 1.0)
        case .green: return NSColor(red: 0.73, green: 0.96, blue: 0.65, alpha: 1.0)
        case .blue: return NSColor(red: 0.69, green: 0.88, blue: 1.0, alpha: 1.0)
        case .orange: return NSColor(red: 1.0, green: 0.82, blue: 0.64, alpha: 1.0)
        case .purple: return NSColor(red: 0.87, green: 0.78, blue: 1.0, alpha: 1.0)
        }
    }
    
    var textColor: NSColor {
        NSColor(red: 0.2, green: 0.2, blue: 0.2, alpha: 1.0)
    }
}