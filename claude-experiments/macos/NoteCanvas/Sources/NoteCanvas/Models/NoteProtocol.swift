import Foundation
import AppKit

public protocol NoteItem: Identifiable, Codable, Hashable {
    var id: UUID { get }
    var position: CGPoint { get set }
    var size: CGSize { get set }
    var zIndex: Int { get set }
    var isSelected: Bool { get set }
    var createdAt: Date { get }
    var modifiedAt: Date { get set }
    var metadata: NoteMetadata { get set }
}

public struct NoteMetadata: Codable, Hashable {
    var title: String?
    var tags: Set<String>
    var color: CodableColor?
    var isLocked: Bool
    var isHidden: Bool
    
    public init(title: String? = nil, tags: Set<String> = [], color: NSColor? = nil, isLocked: Bool = false, isHidden: Bool = false) {
        self.title = title
        self.tags = tags
        self.color = color.map(CodableColor.init)
        self.isLocked = isLocked
        self.isHidden = isHidden
    }
}

public struct CodableColor: Codable, Hashable {
    let red: CGFloat
    let green: CGFloat
    let blue: CGFloat
    let alpha: CGFloat
    
    public init(color: NSColor) {
        let converted = color.usingColorSpace(.sRGB) ?? color
        self.red = converted.redComponent
        self.green = converted.greenComponent
        self.blue = converted.blueComponent
        self.alpha = converted.alphaComponent
    }
    
    public var nsColor: NSColor {
        NSColor(red: red, green: green, blue: blue, alpha: alpha)
    }
}

public protocol ContainerNote: NoteItem {
    associatedtype Item: NoteItem
    var items: [Item] { get set }
    var isExpanded: Bool { get set }
}

public enum NoteType: String, Codable {
    case text
    case image
    case pdf
    case file
    case sticky
    case folder
}