import Foundation
import CoreGraphics
import Metal
import simd

/// Advanced text layout helper
public struct TextLayout {
    /// Font manager for metrics
    private let fontManager: FontAtlasManager
    
    /// A laid out line of text
    public struct Line {
        public let text: String
        public let width: CGFloat
        public let glyphs: [(character: Character, glyph: RenderedGlyph)]
        
        public init(text: String, width: CGFloat, glyphs: [(Character, RenderedGlyph)]) {
            self.text = text
            self.width = width
            self.glyphs = glyphs
        }
    }
    
    /// Text wrapping mode
    public enum WrapMode {
        case none           // No wrapping
        case character      // Wrap at any character
        case word          // Wrap at word boundaries
    }
    
    /// Initialize with a font manager
    public init(fontManager: FontAtlasManager) {
        self.fontManager = fontManager
    }
    
    /// Layout text within a given width
    /// - Parameters:
    ///   - text: Text to layout
    ///   - maxWidth: Maximum width for each line
    ///   - wrapMode: How to wrap text
    /// - Returns: Array of laid out lines
    public func layoutText(_ text: String, maxWidth: CGFloat, wrapMode: WrapMode = .word) -> [Line] {
        switch wrapMode {
        case .none:
            return layoutWithoutWrapping(text)
        case .character:
            return layoutWithCharacterWrapping(text, maxWidth: maxWidth)
        case .word:
            return layoutWithWordWrapping(text, maxWidth: maxWidth)
        }
    }
    
    /// Calculate the bounding box for text
    /// - Parameters:
    ///   - text: Text to measure
    ///   - maxWidth: Maximum width (nil for no limit)
    ///   - wrapMode: Wrapping mode
    /// - Returns: Size needed to render the text
    public func textBounds(_ text: String, maxWidth: CGFloat? = nil, wrapMode: WrapMode = .word) -> CGSize {
        let lines: [Line]
        if let maxWidth = maxWidth {
            lines = layoutText(text, maxWidth: maxWidth, wrapMode: wrapMode)
        } else {
            lines = layoutWithoutWrapping(text)
        }
        
        guard !lines.isEmpty else { return .zero }
        
        let width = lines.map { $0.width }.max() ?? 0
        let height = CGFloat(lines.count) * fontManager.metrics.cellHeight
        
        return CGSize(width: width, height: height)
    }
    
    /// Layout text with custom line breaks
    /// - Parameter text: Text with newline characters
    /// - Returns: Array of lines
    public func layoutLines(_ text: String) -> [Line] {
        let textLines = text.components(separatedBy: .newlines)
        return textLines.compactMap { line in
            let glyphs = collectGlyphs(for: line)
            guard !glyphs.isEmpty else { return nil }
            let width = calculateWidth(for: glyphs)
            return Line(text: line, width: width, glyphs: glyphs)
        }
    }
    
    // MARK: - Private Methods
    
    private func layoutWithoutWrapping(_ text: String) -> [Line] {
        // Split by newlines and layout each line
        return layoutLines(text)
    }
    
    private func layoutWithCharacterWrapping(_ text: String, maxWidth: CGFloat) -> [Line] {
        var lines: [Line] = []
        var currentLine = ""
        var currentGlyphs: [(Character, RenderedGlyph)] = []
        var currentWidth: CGFloat = 0
        
        for character in text {
            if character.isNewline {
                if !currentLine.isEmpty {
                    lines.append(Line(text: currentLine, width: currentWidth, glyphs: currentGlyphs))
                }
                currentLine = ""
                currentGlyphs = []
                currentWidth = 0
                continue
            }
            
            if let glyph = fontManager.renderCharacter(character) {
                let glyphWidth = CGFloat(glyph.advanceX)
                
                // Check if adding this character would exceed max width
                if currentWidth + glyphWidth > maxWidth && !currentLine.isEmpty {
                    lines.append(Line(text: currentLine, width: currentWidth, glyphs: currentGlyphs))
                    currentLine = String(character)
                    currentGlyphs = [(character, glyph)]
                    currentWidth = glyphWidth
                } else {
                    currentLine.append(character)
                    currentGlyphs.append((character, glyph))
                    currentWidth += glyphWidth
                }
            }
        }
        
        // Add remaining line
        if !currentLine.isEmpty {
            lines.append(Line(text: currentLine, width: currentWidth, glyphs: currentGlyphs))
        }
        
        return lines
    }
    
    private func layoutWithWordWrapping(_ text: String, maxWidth: CGFloat) -> [Line] {
        var lines: [Line] = []
        let paragraphs = text.components(separatedBy: .newlines)
        
        for paragraph in paragraphs {
            if paragraph.isEmpty {
                // Preserve empty lines
                lines.append(Line(text: "", width: 0, glyphs: []))
                continue
            }
            
            let words = paragraph.split(separator: " ", omittingEmptySubsequences: false)
                .map { String($0) }
            
            var currentLine = ""
            var currentGlyphs: [(Character, RenderedGlyph)] = []
            var currentWidth: CGFloat = 0
            
            for (index, word) in words.enumerated() {
                let isLastWord = index == words.count - 1
                let wordToMeasure = isLastWord ? word : word + " "
                let wordGlyphs = collectGlyphs(for: wordToMeasure)
                let wordWidth = calculateWidth(for: wordGlyphs)
                
                // Check if word fits on current line
                if currentWidth + wordWidth > maxWidth && !currentLine.isEmpty {
                    // Remove trailing space from current line if present
                    if currentLine.hasSuffix(" ") {
                        currentLine.removeLast()
                        currentGlyphs.removeLast()
                        currentWidth -= CGFloat(currentGlyphs.last?.1.advanceX ?? 0)
                    }
                    
                    lines.append(Line(text: currentLine, width: currentWidth, glyphs: currentGlyphs))
                    currentLine = wordToMeasure
                    currentGlyphs = wordGlyphs
                    currentWidth = wordWidth
                } else {
                    currentLine += wordToMeasure
                    currentGlyphs.append(contentsOf: wordGlyphs)
                    currentWidth += wordWidth
                }
            }
            
            // Add remaining line
            if !currentLine.isEmpty {
                // Remove trailing space if present
                if currentLine.hasSuffix(" ") {
                    currentLine.removeLast()
                    currentGlyphs.removeLast()
                    currentWidth -= CGFloat(currentGlyphs.last?.1.advanceX ?? 0)
                }
                lines.append(Line(text: currentLine, width: currentWidth, glyphs: currentGlyphs))
            }
        }
        
        return lines
    }
    
    private func collectGlyphs(for text: String) -> [(Character, RenderedGlyph)] {
        var glyphs: [(Character, RenderedGlyph)] = []
        for character in text {
            if let glyph = fontManager.renderCharacter(character) {
                glyphs.append((character, glyph))
            }
        }
        return glyphs
    }
    
    private func calculateWidth(for glyphs: [(Character, RenderedGlyph)]) -> CGFloat {
        return glyphs.reduce(0) { $0 + CGFloat($1.1.advanceX) }
    }
}

// MARK: - TextRenderer Extension

extension TextRenderer {
    /// Draw wrapped text within a rectangle
    /// - Parameters:
    ///   - text: Text to draw
    ///   - rect: Rectangle to draw within
    ///   - wrapMode: Text wrapping mode
    ///   - alignment: Text alignment
    ///   - lineSpacing: Additional line spacing
    ///   - color: Text color
    ///   - renderEncoder: Metal render encoder
    public func drawWrappedText(
        _ text: String,
        in rect: CGRect,
        wrapMode: TextLayout.WrapMode = .word,
        alignment: TextAlignment = .left,
        lineSpacing: CGFloat = 0,
        color: simd_float4 = simd_float4(1, 1, 1, 1),
        using renderEncoder: MTLRenderCommandEncoder
    ) {
        let layout = TextLayout(fontManager: fontManager)
        let lines = layout.layoutText(text, maxWidth: rect.width, wrapMode: wrapMode)
        
        let lineHeight = fontManager.metrics.cellHeight + lineSpacing
        var currentY = rect.minY + fontManager.metrics.ascent
        
        for line in lines {
            let x: CGFloat
            
            switch alignment {
            case .left:
                x = rect.minX
            case .center:
                x = rect.minX + (rect.width - line.width) / 2
            case .right:
                x = rect.minX + rect.width - line.width
            }
            
            drawText(line.text, at: CGPoint(x: x, y: currentY), color: color, using: renderEncoder)
            currentY += lineHeight
            
            // Stop if we've exceeded the rect height
            if currentY - fontManager.metrics.ascent > rect.maxY {
                break
            }
        }
    }
}