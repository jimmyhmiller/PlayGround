import Foundation
import AppKit
import PDFKit

// Note: SerializableAnnotation, SerializableColor, and PDFAnnotations are defined in ImagePDFRenderer.swift

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
        // First try to get a thumbnail from the PDF view if it exists and has annotations
        if let thumbnailFromPDFView = generateThumbnailFromPDFView() {
            return thumbnailFromPDFView
        }
        
        // Fallback to basic PDF rendering
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
    
    private func generateThumbnailFromPDFView() -> NSImage? {
        // Find the ImagePDFRenderer in the view hierarchy if it exists
        guard let window = NSApplication.shared.windows.first,
              let imagePDFRenderer = findImagePDFRenderer(in: window.contentView, for: pdfPath) else {
            return nil
        }
        
        // Get the first page bounds from the renderer
        guard let firstPageImage = captureFirstPageFromRenderer(imagePDFRenderer) else {
            return nil
        }
        
        return firstPageImage
    }
    
    private func findImagePDFRenderer(in view: NSView?, for pdfPath: URL) -> ImagePDFRenderer? {
        guard let view = view else { return nil }
        
        // Check if this view is an ImagePDFRenderer for our PDF
        if let renderer = view as? ImagePDFRenderer,
           renderer.pdfNote.pdfPath.absoluteString == pdfPath.absoluteString {
            return renderer
        }
        
        // Recursively search subviews
        for subview in view.subviews {
            if let found = findImagePDFRenderer(in: subview, for: pdfPath) {
                return found
            }
        }
        
        return nil
    }
    
    private func captureFirstPageFromRenderer(_ renderer: ImagePDFRenderer) -> NSImage? {
        // Get both the document view and drawing canvas to composite them manually
        guard let documentView = renderer.documentView,
              let drawingCanvas = renderer.drawingCanvas else {
            return nil
        }
        
        // Extract the top portion that contains the first page
        let documentFrame = documentView.bounds
        let captureHeight = min(documentFrame.height * 0.4, thumbnailSize.height * 2)
        let captureRect = NSRect(
            x: 0,
            y: documentFrame.height - captureHeight, // Top portion
            width: documentFrame.width,
            height: captureHeight
        )
        
        // Create the composite image with both PDF content and annotations
        let compositeImage = NSImage(size: captureRect.size)
        compositeImage.lockFocus()
        
        // Set up high quality rendering
        NSGraphicsContext.current?.imageInterpolation = .high
        NSGraphicsContext.current?.shouldAntialias = true
        
        // First, capture and draw the PDF content (pageImageView)
        for subview in documentView.subviews {
            if let imageView = subview as? NSImageView, let pdfImage = imageView.image {
                // This is the pageImageView with PDF content
                let imageFrame = imageView.frame
                
                // Check if this image view intersects with our capture rect
                let intersectionRect = imageFrame.intersection(captureRect)
                if !intersectionRect.isEmpty {
                    // Calculate the portion of the image to draw
                    let sourceRect = NSRect(
                        x: intersectionRect.origin.x - imageFrame.origin.x,
                        y: intersectionRect.origin.y - imageFrame.origin.y,
                        width: intersectionRect.width,
                        height: intersectionRect.height
                    )
                    
                    let destRect = NSRect(
                        x: intersectionRect.origin.x - captureRect.origin.x,
                        y: intersectionRect.origin.y - captureRect.origin.y,
                        width: intersectionRect.width,
                        height: intersectionRect.height
                    )
                    
                    pdfImage.draw(in: destRect, from: sourceRect, operation: .copy, fraction: 1.0)
                }
                break // Only need the first (and only) image view
            }
        }
        
        // Then, draw the annotations on top
        let canvasFrame = drawingCanvas.frame
        let canvasIntersection = canvasFrame.intersection(captureRect)
        if !canvasIntersection.isEmpty {
            // Translate the graphics context to draw the annotations at the correct position
            NSGraphicsContext.current?.saveGraphicsState()
            
            let offsetX = canvasIntersection.origin.x - captureRect.origin.x
            let offsetY = canvasIntersection.origin.y - captureRect.origin.y
            NSGraphicsContext.current?.cgContext.translateBy(x: offsetX, y: offsetY)
            
            // Draw the annotations in the intersection area
            drawingCanvas.displayIfNeeded()
            drawingCanvas.draw(canvasIntersection)
            
            NSGraphicsContext.current?.restoreGraphicsState()
        }
        
        compositeImage.unlockFocus()
        
        // Scale the composite image to thumbnail size
        return scaleImageToThumbnailSize(compositeImage)
    }
    
    private func scaleImageToThumbnailSize(_ sourceImage: NSImage) -> NSImage {
        let scaledImage = NSImage(size: thumbnailSize)
        scaledImage.lockFocus()
        
        NSGraphicsContext.current?.imageInterpolation = .high
        
        // Draw with aspect ratio preservation
        let sourceSize = sourceImage.size
        let aspectRatio = sourceSize.width / sourceSize.height
        let thumbnailAspectRatio = thumbnailSize.width / thumbnailSize.height
        
        var drawRect = NSRect(origin: .zero, size: thumbnailSize)
        if aspectRatio > thumbnailAspectRatio {
            // Source is wider, fit to width
            let scaledHeight = thumbnailSize.width / aspectRatio
            drawRect.origin.y = (thumbnailSize.height - scaledHeight) / 2
            drawRect.size.height = scaledHeight
        } else {
            // Source is taller, fit to height
            let scaledWidth = thumbnailSize.height * aspectRatio
            drawRect.origin.x = (thumbnailSize.width - scaledWidth) / 2
            drawRect.size.width = scaledWidth
        }
        
        sourceImage.draw(in: drawRect)
        
        scaledImage.unlockFocus()
        return scaledImage
    }
    
    
    private func loadAnnotationsFromFile() -> [SerializableAnnotation] {
        // Use the same persistence logic from AnnotationPersistenceManager
        let filename = "annotations_\(String(pdfPath.absoluteString.hash)).json"
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let appDir = appSupport.appendingPathComponent("NoteCanvas")
        let annotationsDir = appDir.appendingPathComponent("PDF_Annotations")
        let filePath = annotationsDir.appendingPathComponent(filename)
        
        guard let data = try? Data(contentsOf: filePath),
              let pdfAnnotations = try? JSONDecoder().decode(PDFAnnotations.self, from: data) else {
            return []
        }
        
        return pdfAnnotations.annotations
    }
}