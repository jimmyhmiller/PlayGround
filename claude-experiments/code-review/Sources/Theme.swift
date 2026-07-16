import SwiftUI

extension Color {
    init(hex: UInt32, alpha: Double = 1) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255.0,
            green: Double((hex >> 8) & 0xFF) / 255.0,
            blue: Double(hex & 0xFF) / 255.0,
            opacity: alpha
        )
    }
}

enum Th {
    static let bg = Color(hex: 0x1c1c1f)
    static let bgHeader = Color(hex: 0x202023)
    static let sidebar = Color(hex: 0x1e1e22).opacity(0.98)
    static let titlebar = Color(hex: 0x26262a).opacity(0.95)
    static let panel = Color(hex: 0x232326)
    static let card = Color(hex: 0x2a2a2e)
    static let cardDark = Color(hex: 0x141416)
    static let editorField = Color(hex: 0x1a1a1c)

    static let text = Color(hex: 0xe8e8ea)
    static let text2 = Color(hex: 0xc4c4c8)
    static let text3 = Color(hex: 0x9a9aa0)
    static let dim = Color(hex: 0x8a8a90)
    static let dimmer = Color(hex: 0x6e6e76)
    static let faint = Color(hex: 0x5a5a60)
    static let codeText = Color(hex: 0xd4d4d8)

    static let accent = Color(hex: 0x0A84FF)
    static let ai = Color(hex: 0xAF52DE)
    static let green = Color(hex: 0x3fb950)
    static let greenSoft = Color(hex: 0x5ed17a)
    static let red = Color(hex: 0xf85149)
    static let redSoft = Color(hex: 0xff8f8a)
    static let yellow = Color(hex: 0xfebc2e)
    static let purple = Color(hex: 0xa371f7)
    static let orange = Color(hex: 0xf0a35e)
    static let blue = Color(hex: 0x6cb6ff)
    static let hunkText = Color(hex: 0x8ea2e0)

    static let border = Color.white.opacity(0.07)
    static let borderStrong = Color.white.opacity(0.12)

    static let addBg = Color(hex: 0x3fb950, alpha: 0.15)
    static let addGutter = Color(hex: 0x3fb950, alpha: 0.22)
    static let delBg = Color(hex: 0xf85149, alpha: 0.15)
    static let delGutter = Color(hex: 0xf85149, alpha: 0.22)
    static let ctxGutter = Color.white.opacity(0.02)
    static let hunkBg = Color(hex: 0x5878dc, alpha: 0.13)
    static let hunkBorder = Color(hex: 0x5878dc, alpha: 0.22)

    static func statusColors(_ status: String) -> (Color, Color) {
        switch status {
        case "A", "?": return (greenSoft, Color(hex: 0x3fb950, alpha: 0.16))
        case "D": return (redSoft, Color(hex: 0xf85149, alpha: 0.16))
        case "R": return (purple, Color(hex: 0xa371f7, alpha: 0.18))
        default: return (orange, Color(hex: 0xf08c32, alpha: 0.16))
        }
    }

    static func categoryColors(_ cat: String) -> (Color, Color) {
        switch cat.lowercased() {
        case "bug", "blocker", "security", "issue":
            return (redSoft, Color(hex: 0xf85149, alpha: 0.16))
        case "nit":
            return (Color(hex: 0xa8a8b0), Color.white.opacity(0.09))
        case "praise":
            return (greenSoft, Color(hex: 0x3fb950, alpha: 0.16))
        case "question":
            return (blue, Color(hex: 0x388bfd, alpha: 0.16))
        case "consider", "suggestion", "perf", "performance":
            return (orange, Color(hex: 0xf08c32, alpha: 0.16))
        case "style":
            return (Color(hex: 0xc8a0ff), Color(hex: 0xa371f7, alpha: 0.18))
        default:
            return (Color(hex: 0xa8a8b0), Color.white.opacity(0.09))
        }
    }
}
