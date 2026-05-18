import AppKit

enum MarkdownHighlighter {
    private static let baseSize: CGFloat = 14
    private static let hiddenSize: CGFloat = 0.01

    static func apply(to storage: NSTextStorage,
                      rawMode: Bool,
                      activeLineRange: NSRange?) {
        let full = NSRange(location: 0, length: storage.length)

        if rawMode {
            let mono = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)
            storage.setAttributes([
                .font: mono,
                .foregroundColor: NSColor.labelColor,
                .paragraphStyle: defaultParagraph()
            ], range: full)
            return
        }

        // Base body attributes.
        storage.setAttributes([
            .font: NSFont.systemFont(ofSize: baseSize),
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: defaultParagraph()
        ], range: full)

        let text = storage.string as NSString
        let active = activeLineRange ?? NSRange(location: NSNotFound, length: 0)

        // Marker on the active line: muted but visible. Off the active line: collapsed to ~zero width + clear.
        let marker: (NSRange) -> Void = { range in
            if active.location != NSNotFound,
               NSIntersectionRange(range, active).length > 0 {
                storage.addAttributes([
                    .foregroundColor: NSColor.tertiaryLabelColor
                ], range: range)
            } else {
                storage.addAttributes([
                    .font: NSFont.systemFont(ofSize: hiddenSize),
                    .foregroundColor: NSColor.clear
                ], range: range)
            }
        }

        // Always-muted marker (lists, blockquotes, rules) — single-char structural cues stay visible.
        let mutedAlways: (NSRange) -> Void = { range in
            storage.addAttributes([
                .foregroundColor: NSColor.tertiaryLabelColor
            ], range: range)
        }

        styleHeadings(text: text, storage: storage, marker: marker)
        styleBlockquotes(text: text, storage: storage, marker: mutedAlways)
        styleLists(text: text, storage: storage, marker: mutedAlways)
        styleHorizontalRules(text: text, storage: storage, marker: mutedAlways)
        styleInlineCode(text: text, storage: storage, marker: marker)
        styleBold(text: text, storage: storage, marker: marker)
        styleItalic(text: text, storage: storage, marker: marker)
        styleLinks(text: text, storage: storage, marker: marker)
    }

    // MARK: - Block elements

    private static func styleHeadings(text: NSString,
                                      storage: NSTextStorage,
                                      marker: (NSRange) -> Void) {
        regex(#"^(#{1,6})([ \t]+)(.+)$"#, options: [.anchorsMatchLines])
            .enumerateMatches(in: text as String, range: fullRange(text)) { match, _, _ in
                guard let m = match else { return }
                let hashes = m.range(at: 1)
                let space = m.range(at: 2)
                let content = m.range(at: 3)
                let size: CGFloat
                switch hashes.length {
                case 1: size = 22
                case 2: size = 19
                case 3: size = 17
                case 4: size = 15
                default: size = 14
                }
                let font = NSFont.systemFont(ofSize: size, weight: .semibold)
                storage.addAttributes([
                    .font: font,
                    .foregroundColor: NSColor.labelColor
                ], range: content)
                marker(NSUnionRange(hashes, space))
            }
    }

    private static func styleBlockquotes(text: NSString,
                                         storage: NSTextStorage,
                                         marker: (NSRange) -> Void) {
        regex(#"^(>[ \t]?)(.*)$"#, options: [.anchorsMatchLines])
            .enumerateMatches(in: text as String, range: fullRange(text)) { match, _, _ in
                guard let m = match else { return }
                let markerR = m.range(at: 1)
                let content = m.range(at: 2)
                storage.addAttributes([
                    .foregroundColor: NSColor.secondaryLabelColor,
                    .obliqueness: 0.07
                ], range: content)
                marker(markerR)
            }
    }

    private static func styleLists(text: NSString,
                                   storage: NSTextStorage,
                                   marker: (NSRange) -> Void) {
        regex(#"^([ \t]*)([-*+]|\d+\.)[ \t]+"#, options: [.anchorsMatchLines])
            .enumerateMatches(in: text as String, range: fullRange(text)) { match, _, _ in
                guard let m = match else { return }
                let markerR = m.range(at: 2)
                storage.addAttributes([
                    .font: NSFont.systemFont(ofSize: baseSize, weight: .semibold)
                ], range: markerR)
                marker(markerR)
            }
    }

    private static func styleHorizontalRules(text: NSString,
                                             storage: NSTextStorage,
                                             marker: (NSRange) -> Void) {
        regex(#"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$"#, options: [.anchorsMatchLines])
            .enumerateMatches(in: text as String, range: fullRange(text)) { match, _, _ in
                guard let m = match else { return }
                marker(m.range)
            }
    }

    // MARK: - Inline elements

    private static func styleInlineCode(text: NSString,
                                        storage: NSTextStorage,
                                        marker: (NSRange) -> Void) {
        regex(#"`([^`\n]+)`"#)
            .enumerateMatches(in: text as String, range: fullRange(text)) { match, _, _ in
                guard let m = match else { return }
                let content = m.range(at: 1)
                let mono = NSFont.monospacedSystemFont(ofSize: baseSize - 1, weight: .regular)
                storage.addAttributes([
                    .font: mono,
                    .foregroundColor: NSColor.systemPink,
                    .backgroundColor: NSColor.quaternaryLabelColor.withAlphaComponent(0.35)
                ], range: content)
                let openTick = NSRange(location: m.range.location, length: 1)
                let closeTick = NSRange(location: m.range.upperBound - 1, length: 1)
                marker(openTick)
                marker(closeTick)
            }
    }

    private static func styleBold(text: NSString,
                                  storage: NSTextStorage,
                                  marker: (NSRange) -> Void) {
        regex(#"(\*\*|__)(?=\S)(.+?)(?<=\S)\1"#)
            .enumerateMatches(in: text as String, range: fullRange(text)) { match, _, _ in
                guard let m = match else { return }
                let openMarker = NSRange(location: m.range.location, length: 2)
                let closeMarker = NSRange(location: m.range.upperBound - 2, length: 2)
                let content = NSRange(
                    location: openMarker.upperBound,
                    length: m.range.length - 4
                )
                let bold = NSFont.systemFont(ofSize: baseSize, weight: .bold)
                storage.addAttributes([.font: bold], range: content)
                marker(openMarker)
                marker(closeMarker)
            }
    }

    private static func styleItalic(text: NSString,
                                    storage: NSTextStorage,
                                    marker: (NSRange) -> Void) {
        regex(#"(?<![\*\w])(\*|_)(?=\S)([^\*_\n]+?)(?<=\S)\1(?![\*\w])"#)
            .enumerateMatches(in: text as String, range: fullRange(text)) { match, _, _ in
                guard let m = match else { return }
                let openMarker = NSRange(location: m.range.location, length: 1)
                let closeMarker = NSRange(location: m.range.upperBound - 1, length: 1)
                let content = NSRange(
                    location: openMarker.upperBound,
                    length: m.range.length - 2
                )
                let italic = italicSystemFont(ofSize: baseSize)
                storage.addAttributes([.font: italic], range: content)
                marker(openMarker)
                marker(closeMarker)
            }
    }

    private static func styleLinks(text: NSString,
                                   storage: NSTextStorage,
                                   marker: (NSRange) -> Void) {
        regex(#"\[([^\]\n]+)\]\(([^)\n]+)\)"#)
            .enumerateMatches(in: text as String, range: fullRange(text)) { match, _, _ in
                guard let m = match else { return }
                let label = m.range(at: 1)
                let url = m.range(at: 2)
                let openBracket = NSRange(location: m.range.location, length: 1)
                let closeBracket = NSRange(location: label.upperBound, length: 1)
                let openParen = NSRange(location: closeBracket.upperBound, length: 1)
                let closeParen = NSRange(location: m.range.upperBound - 1, length: 1)

                storage.addAttributes([
                    .foregroundColor: NSColor.linkColor,
                    .underlineStyle: NSUnderlineStyle.single.rawValue
                ], range: label)

                marker(openBracket)
                marker(closeBracket)
                marker(openParen)
                marker(closeParen)
                marker(url)
            }
    }

    // MARK: - Helpers

    private static func regex(_ pattern: String,
                              options: NSRegularExpression.Options = []) -> NSRegularExpression {
        do {
            return try NSRegularExpression(pattern: pattern, options: options)
        } catch {
            fatalError("Invalid markdown regex \(pattern): \(error)")
        }
    }

    private static func fullRange(_ text: NSString) -> NSRange {
        NSRange(location: 0, length: text.length)
    }

    private static func defaultParagraph() -> NSParagraphStyle {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3
        style.paragraphSpacing = 2
        return style
    }

    private static func italicSystemFont(ofSize size: CGFloat) -> NSFont {
        let base = NSFont.systemFont(ofSize: size)
        let descriptor = base.fontDescriptor.withSymbolicTraits(.italic)
        return NSFont(descriptor: descriptor, size: size) ?? base
    }
}
