import Foundation
import AppKit
import PDFKit

public struct PDFNote: NoteItem {
    public let id: UUID
    public var position: CGPoint
    public var size: CGSize
    public var zIndex: Int
    public var isSelected: Bool
    public let createdAt: Date
    public var modifiedAt: Date
    public var metadata: NoteMetadata
    
    public var pdfPath: URL
    public var currentPage: Int
    public var cornerRadius: CGFloat
    public var thumbnailSize: CGSize
    
    public init(id: UUID = UUID(),
         position: CGPoint = .zero,
         size: CGSize = CGSize(width: 300, height: 400),
         pdfPath: URL,
         currentPage: Int = 0,
         cornerRadius: CGFloat = 12) {
        self.id = id
        self.position = position
        self.size = size
        self.zIndex = 0
        self.isSelected = false
        self.createdAt = Date()
        self.modifiedAt = Date()
        self.metadata = NoteMetadata()
        self.pdfPath = pdfPath
        self.currentPage = currentPage
        self.cornerRadius = cornerRadius
        
        // Calculate thumbnail size based on PDF first page
        if let pdfDocument = PDFDocument(url: pdfPath),
           let firstPage = pdfDocument.page(at: 0) {
            let pageSize = firstPage.bounds(for: .mediaBox).size
            let aspectRatio = pageSize.width / pageSize.height
            
            if aspectRatio > 1 {
                // Landscape
                self.thumbnailSize = CGSize(width: size.width, height: size.width / aspectRatio)
            } else {
                // Portrait
                self.thumbnailSize = CGSize(width: size.height * aspectRatio, height: size.height)
            }
        } else {
            self.thumbnailSize = size
        }
    }
    
    public func generateThumbnailImage() -> NSImage? {
        guard let pdfDocument = PDFDocument(url: pdfPath),
              let firstPage = pdfDocument.page(at: 0) else {
            return nil
        }
        
        let pageSize = firstPage.bounds(for: .mediaBox).size
        let scale = min(thumbnailSize.width / pageSize.width, thumbnailSize.height / pageSize.height)
        
        let scaledSize = CGSize(
            width: pageSize.width * scale,
            height: pageSize.height * scale
        )
        
        let image = NSImage(size: scaledSize)
        image.lockFocus()
        
        NSGraphicsContext.current?.imageInterpolation = .high
        
        // White background
        NSColor.white.setFill()
        NSRect(origin: .zero, size: scaledSize).fill()
        
        // Render PDF page
        let transform = NSAffineTransform()
        transform.scale(by: scale)
        transform.concat()
        
        firstPage.draw(with: .mediaBox, to: NSGraphicsContext.current!.cgContext)
        
        image.unlockFocus()
        return image
    }
}