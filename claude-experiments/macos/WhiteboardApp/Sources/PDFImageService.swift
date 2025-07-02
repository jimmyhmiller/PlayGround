import AppKit
import Quartz
import Foundation

class PDFImageService {
    static let shared = PDFImageService()
    private let cacheDirectory: URL
    private let imageCache = NSCache<NSString, NSImage>()
    
    private init() {
        // Create cache directory
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        cacheDirectory = appSupport.appendingPathComponent("WhiteboardApp/PDFImageCache")
        
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        
        // Configure memory cache
        imageCache.countLimit = 50 // Cache up to 50 page images
        imageCache.totalCostLimit = 100 * 1024 * 1024 // 100MB memory limit
    }
    
    struct PDFPageImage {
        let image: NSImage
        let pageIndex: Int
        let originalSize: CGSize
        let scale: CGFloat
    }
    
    // Convert PDF pages to images at 150 DPI
    func convertPDFToImages(
        _ pdfDocument: PDFDocument,
        progress: @escaping (Double) -> Void = { _ in }
    ) async -> [PDFPageImage] {
        let pageCount = pdfDocument.pageCount
        
        // Process pages sequentially to avoid issues
        var results: [PDFPageImage] = []
        
        for pageIndex in 0..<pageCount {
            if let pageImage = await convertPage(pdfDocument, pageIndex: pageIndex) {
                results.append(pageImage)
            }
            
            await MainActor.run {
                progress(Double(pageIndex + 1) / Double(pageCount))
            }
        }
        
        return results
    }
    
    private func convertPage(_ pdfDocument: PDFDocument, pageIndex: Int) async -> PDFPageImage? {
        guard let page = pdfDocument.page(at: pageIndex) else { return nil }
        
        // Check cache first
        let cacheKey = "\(pdfDocument.documentURL?.lastPathComponent ?? "unknown")_page_\(pageIndex)" as NSString
        
        if let cachedImage = imageCache.object(forKey: cacheKey) {
            return PDFPageImage(
                image: cachedImage,
                pageIndex: pageIndex,
                originalSize: page.bounds(for: .mediaBox).size,
                scale: 72.0 / 72.0 // 72 DPI / 72 DPI
            )
        }
        
        // Check disk cache
        let cacheFile = cacheDirectory.appendingPathComponent("\(cacheKey).png")
        if FileManager.default.fileExists(atPath: cacheFile.path),
           let cachedImage = NSImage(contentsOf: cacheFile) {
            imageCache.setObject(cachedImage, forKey: cacheKey)
            return PDFPageImage(
                image: cachedImage,
                pageIndex: pageIndex,
                originalSize: page.bounds(for: .mediaBox).size,
                scale: 72.0 / 72.0
            )
        }
        
        // Generate new image at lower DPI for better performance
        let mediaBox = page.bounds(for: .mediaBox)
        let scale: CGFloat = 72.0 / 72.0 // 72 DPI for fast performance
        let scaledSize = CGSize(
            width: mediaBox.width * scale,
            height: mediaBox.height * scale
        )
        
        let image = NSImage(size: scaledSize)
        
        image.lockFocus()
        defer { image.unlockFocus() }
        
        guard let context = NSGraphicsContext.current?.cgContext else { return nil }
        
        // Fill with white background first
        context.setFillColor(NSColor.white.cgColor)
        context.fill(CGRect(origin: .zero, size: scaledSize))
        
        // Scale and render the PDF page
        context.scaleBy(x: scale, y: scale)
        context.translateBy(x: -mediaBox.origin.x, y: -mediaBox.origin.y)
        page.draw(with: .mediaBox, to: context)
        
        // Cache the image
        imageCache.setObject(image, forKey: cacheKey, cost: Int(scaledSize.width * scaledSize.height * 4))
        
        // Save to disk cache
        if let tiffData = image.tiffRepresentation,
           let bitmap = NSBitmapImageRep(data: tiffData),
           let pngData = bitmap.representation(using: .png, properties: [:]) {
            try? pngData.write(to: cacheFile)
        }
        
        return PDFPageImage(
            image: image,
            pageIndex: pageIndex,
            originalSize: mediaBox.size,
            scale: scale
        )
    }
    
    // Clear cache when memory pressure or user request
    func clearCache() {
        imageCache.removeAllObjects()
        try? FileManager.default.removeItem(at: cacheDirectory)
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
    }
}