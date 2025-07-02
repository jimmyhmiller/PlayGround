import AppKit
import SwiftUI
import CoreText

struct TextBubbleDrawing {
    static func drawTextBubble(
        bounds: CGRect,
        text: String,
        color: NSColor,
        fontSize: CGFloat = 16.0,
        isSelected: Bool = false,
        isEditing: Bool = false,
        in context: CGContext
    ) {
        let cornerRadius: CGFloat = 8.0
        let path = CGPath(roundedRect: bounds,
                         cornerWidth: cornerRadius,
                         cornerHeight: cornerRadius,
                         transform: nil)
        
        // Fill with semi-transparent background
        context.setFillColor(NSColor.white.withAlphaComponent(0.9).cgColor)
        context.addPath(path)
        context.fillPath()
        
        // Stroke with color
        context.setStrokeColor(color.cgColor)
        context.setLineWidth(2)
        context.setLineDash(phase: 0, lengths: [])
        context.addPath(path)
        context.strokePath()
        
        // Draw text (only if not editing)
        if !isEditing {
            let font = NSFont.systemFont(ofSize: fontSize)
            let attributes: [NSAttributedString.Key: Any] = [
                .font: font,
                .foregroundColor: NSColor.black
            ]
            
            let textSize = (text as NSString).size(withAttributes: attributes)
            let textRect = CGRect(
                x: bounds.origin.x + (bounds.width - textSize.width) / 2,
                y: bounds.origin.y + (bounds.height - textSize.height) / 2,
                width: textSize.width,
                height: textSize.height
            )
            
            (text as NSString).draw(in: textRect, withAttributes: attributes)
        }
        
        // Selection outline
        if isSelected {
            let selectionOffset: CGFloat = 3
            let lineWidth: CGFloat = 2
            let totalOffset = selectionOffset + lineWidth/2
            let selectionBounds = bounds.insetBy(dx: -totalOffset, dy: -totalOffset)
            
            let selectionCornerRadius = cornerRadius + totalOffset
            
            let selectionPath = CGPath(roundedRect: selectionBounds,
                                     cornerWidth: selectionCornerRadius,
                                     cornerHeight: selectionCornerRadius,
                                     transform: nil)
            context.setStrokeColor(NSColor.systemBlue.cgColor)
            context.setLineWidth(lineWidth)
            context.setLineDash(phase: 0, lengths: [])
            context.addPath(selectionPath)
            context.strokePath()
        }
    }
}

struct RectangleDrawing {
    static func drawRectangle(
        bounds: CGRect,
        color: NSColor,
        cornerRadius: CGFloat = 8.0,
        isSelected: Bool = false,
        in context: CGContext
    ) {
        let path = CGPath(roundedRect: bounds,
                         cornerWidth: cornerRadius,
                         cornerHeight: cornerRadius,
                         transform: nil)
        
        // Fill with semi-transparent color
        context.setFillColor(color.withAlphaComponent(0.3).cgColor)
        context.addPath(path)
        context.fillPath()
        
        // Stroke with full color
        context.setStrokeColor(color.cgColor)
        context.setLineWidth(2)
        context.setLineDash(phase: 0, lengths: [])
        context.addPath(path)
        context.strokePath()
        
        // Selection outline
        if isSelected {
            let selectionOffset: CGFloat = 3
            let lineWidth: CGFloat = 2
            let totalOffset = selectionOffset + lineWidth/2
            let selectionBounds = bounds.insetBy(dx: -totalOffset, dy: -totalOffset)
            
            // The corner radius should be the original radius plus the distance from the original edge to the center of the selection line
            let selectionCornerRadius = cornerRadius + totalOffset
            
            let selectionPath = CGPath(roundedRect: selectionBounds,
                                     cornerWidth: selectionCornerRadius,
                                     cornerHeight: selectionCornerRadius,
                                     transform: nil)
            context.setStrokeColor(NSColor.systemBlue.cgColor)
            context.setLineWidth(lineWidth)
            context.setLineDash(phase: 0, lengths: [])
            context.addPath(selectionPath)
            context.strokePath()
        }
    }
}