import SwiftUI

enum ThemeType: String, CaseIterable {
    case alchemist
    case construct
    case pulse
}

struct ThemeColors {
    let bgApp: Color
    let textPrimary: Color
    let textSecondary: Color
    let accent: Color
    let widgetBg: Color
    let borderColor: Color
}

struct ThemeData {
    let title: String
    let subtitle: String
    let stat: String
    let items: [String]
    let colors: ThemeColors
    let cornerRadius: CGFloat
    let borderWidth: CGFloat
    let fontStyle: FontStyle

    enum FontStyle {
        case serif
        case mono
        case system
    }

    static func forType(_ type: ThemeType) -> ThemeData {
        switch type {
        case .alchemist:
            return ThemeData(
                title: "The Alchemist",
                subtitle: "ORGANIC NEURAL PROCESSING",
                stat: "99.4%",
                items: ["Grimoire.py", "Potion.config", "Elixir.js", "Root.css"],
                colors: ThemeColors(
                    bgApp: Color(hex: "14120e"),
                    textPrimary: Color(hex: "e6d2aa"),
                    textSecondary: Color(hex: "8c7e5e"),
                    accent: Color(hex: "d4af37"),
                    widgetBg: Color(hex: "d4af37").opacity(0.03),
                    borderColor: Color(hex: "332a18")
                ),
                cornerRadius: 20,
                borderWidth: 1,
                fontStyle: .serif
            )
        case .construct:
            return ThemeData(
                title: "CONSTRUCT_V4",
                subtitle: "MAINFRAME INFRASTRUCTURE",
                stat: "12ms",
                items: ["SYS_BOOT.bat", "KERNEL_32.dll", "NET_bridge.go", "DOCKER_img"],
                colors: ThemeColors(
                    bgApp: Color(hex: "05080a"),
                    textPrimary: Color(hex: "aaffff"),
                    textSecondary: Color(hex: "406060"),
                    accent: Color(hex: "00f0ff"),
                    widgetBg: Color(hex: "00f0ff").opacity(0.02),
                    borderColor: Color(hex: "003333")
                ),
                cornerRadius: 4,
                borderWidth: 1,
                fontStyle: .mono
            )
        case .pulse:
            return ThemeData(
                title: "Neon Pulse",
                subtitle: "FLUID STATE MANAGER",
                stat: "84bpm",
                items: ["Heartbeat.tsx", "Flow.ts", "Rhythm.css", "Wave.svg"],
                colors: ThemeColors(
                    bgApp: Color(hex: "0e0508"),
                    textPrimary: Color(hex: "ffccd5"),
                    textSecondary: Color(hex: "804050"),
                    accent: Color(hex: "ff2a68"),
                    widgetBg: Color(hex: "ff2a68").opacity(0.05),
                    borderColor: Color.white.opacity(0.05)
                ),
                cornerRadius: 30,
                borderWidth: 0,
                fontStyle: .system
            )
        }
    }
}

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let r, g, b: UInt64
        switch hex.count {
        case 6: // RGB
            (r, g, b) = ((int >> 16) & 0xFF, (int >> 8) & 0xFF, int & 0xFF)
        default:
            (r, g, b) = (0, 0, 0)
        }
        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: 1
        )
    }
}
