import Foundation
import AppKit

struct ImageNote: NoteItem {
    let id: UUID
    var position: CGPoint
    var size: CGSize
    var zIndex: Int
    var isSelected: Bool
    let createdAt: Date
    var modifiedAt: Date
    var metadata: NoteMetadata
    
    var imagePath: URL
    var aspectRatioMode: AspectRatioMode
    var cornerRadius: CGFloat
    
    init(id: UUID = UUID(),
         position: CGPoint = .zero,
         size: CGSize = CGSize(width: 300, height: 200),
         imagePath: URL,
         aspectRatioMode: AspectRatioMode = .fit,
         cornerRadius: CGFloat = 12) {
        self.id = id
        self.position = position
        self.size = size
        self.zIndex = 0
        self.isSelected = false
        self.createdAt = Date()
        self.modifiedAt = Date()
        self.metadata = NoteMetadata()
        self.imagePath = imagePath
        self.aspectRatioMode = aspectRatioMode
        self.cornerRadius = cornerRadius
    }
}

enum AspectRatioMode: String, Codable {
    case fit
    case fill
    case stretch
}