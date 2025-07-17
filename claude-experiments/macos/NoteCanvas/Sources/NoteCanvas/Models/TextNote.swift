import Foundation
import AppKit

public struct TextNote: NoteItem {
    public let id: UUID
    public var position: CGPoint
    public var size: CGSize
    public var zIndex: Int
    public var isSelected: Bool
    public let createdAt: Date
    public var modifiedAt: Date
    public var metadata: NoteMetadata
    
    public var content: String
    public var font: CodableFont
    public var textColor: CodableColor
    public var alignment: TextAlignment
    
    public init(id: UUID = UUID(),
         position: CGPoint = .zero,
         size: CGSize = CGSize(width: 200, height: 150),
         content: String = "",
         font: NSFont = .systemFont(ofSize: 14),
         textColor: NSColor = .textColor,
         alignment: TextAlignment = .left) {
        self.id = id
        self.position = position
        self.size = size
        self.zIndex = 0
        self.isSelected = false
        self.createdAt = Date()
        self.modifiedAt = Date()
        self.metadata = NoteMetadata()
        self.content = content
        self.font = CodableFont(font: font)
        self.textColor = CodableColor(color: textColor)
        self.alignment = alignment
    }
}

public struct CodableFont: Codable, Hashable {
    let familyName: String
    let pointSize: CGFloat
    
    public init(font: NSFont) {
        self.familyName = font.familyName ?? NSFont.systemFont(ofSize: 14).familyName!
        self.pointSize = font.pointSize
    }
    
    public var nsFont: NSFont {
        return NSFont(name: familyName, size: pointSize) ?? NSFont.systemFont(ofSize: pointSize)
    }
}

public enum TextAlignment: String, Codable {
    case left, center, right, justified
    
    public var nsTextAlignment: NSTextAlignment {
        switch self {
        case .left: return .left
        case .center: return .center
        case .right: return .right
        case .justified: return .justified
        }
    }
}