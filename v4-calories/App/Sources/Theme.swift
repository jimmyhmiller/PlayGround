import SwiftUI

/// Design tokens lifted directly from the Cumulative Tracker design.
enum Theme {
    static let bg = Color(hex: 0x0A0A0B)
    static let sheetBg = Color(hex: 0x0E0E10)
    static let key = Color(hex: 0x191A1D)
    static let green = Color(hex: 0x7DD3A8)
    static let amber = Color(hex: 0xF0A878)
    static let onGreen = Color(hex: 0x08130D)

    static let text = Color(hex: 0xF3F3F5)
    static func textDim(_ a: Double) -> Color { Color(white: 0.92, opacity: a) }

    static let hair = Color.white.opacity(0.07)
    static let hairLight = Color.white.opacity(0.05)
}

extension Color {
    init(hex: UInt32, alpha: Double = 1) {
        self.init(.sRGB,
                  red: Double((hex >> 16) & 0xFF) / 255,
                  green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255,
                  opacity: alpha)
    }
}

extension Font {
    /// JetBrains Mono in the design → SF Mono (monospaced system) here.
    static func mono(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
}

/// Number formatting matching the design (grouped thousands, signed values with a real minus).
enum Fmt {
    static func int(_ n: Double) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.maximumFractionDigits = 0
        return f.string(from: NSNumber(value: n.rounded())) ?? "\(Int(n.rounded()))"
    }

    /// Signed with a typographic minus (−), e.g. "−1,240" or "+820".
    static func signedInt(_ n: Double) -> String {
        let s = int(abs(n))
        if n > 0.5 { return "+" + s }
        if n < -0.5 { return "\u{2212}" + s }
        return s
    }

    static func signed(_ n: Double, _ decimals: Int = 1) -> String {
        let s = String(format: "%.\(decimals)f", abs(n))
        if n > 0 { return "+" + s }
        if n < 0 { return "\u{2212}" + s }
        return s
    }

    static func f(_ n: Double, _ decimals: Int = 1) -> String {
        String(format: "%.\(decimals)f", n)
    }
}
