import SwiftUI
import Quartz

struct Rectangle: Identifiable {
    let id = UUID()
    var origin: CGPoint
    var size: CGSize
    var color: Color
    var isSelected: Bool = false
    var cornerRadius: CGFloat = 8.0
    
    var frame: CGRect {
        CGRect(origin: origin, size: size)
    }
    
    func contains(point: CGPoint) -> Bool {
        frame.contains(point)
    }
}

struct TextBubble: Identifiable {
    let id = UUID()
    var origin: CGPoint
    var text: String
    var color: Color
    var isSelected: Bool = false
    var fontSize: CGFloat = 16.0
    
    // Calculated size based on text content
    var size: CGSize {
        let font = NSFont.systemFont(ofSize: fontSize)
        let attributes: [NSAttributedString.Key: Any] = [.font: font]
        let textSize = (text as NSString).size(withAttributes: attributes)
        let padding: CGFloat = 16
        return CGSize(width: textSize.width + padding, height: textSize.height + padding)
    }
    
    var frame: CGRect {
        CGRect(origin: origin, size: size)
    }
    
    func contains(point: CGPoint) -> Bool {
        frame.contains(point)
    }
}

struct PDFItem: Identifiable {
    let id = UUID()
    var origin: CGPoint
    var size: CGSize
    var pdfDocument: PDFDocument
    var isSelected: Bool = false
    var cornerRadius: CGFloat = 8.0
    
    var frame: CGRect {
        CGRect(origin: origin, size: size)
    }
    
    func contains(point: CGPoint) -> Bool {
        frame.contains(point)
    }
}