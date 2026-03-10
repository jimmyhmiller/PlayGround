import SwiftUI

#if os(macOS)
import AppKit
public typealias PlatformColor = NSColor
public typealias PlatformView = NSView
#else
import UIKit
public typealias PlatformColor = UIColor
public typealias PlatformView = UIView
#endif

// MARK: - Cross-platform system colors

extension Color {
    static var systemGroupedBackground: Color {
        #if os(macOS)
        Color(nsColor: NSColor.windowBackgroundColor)
        #else
        Color(.systemGroupedBackground)
        #endif
    }

    static var systemGray5: Color {
        #if os(macOS)
        Color(nsColor: NSColor.controlBackgroundColor)
        #else
        Color(.systemGray5)
        #endif
    }

    static var systemGray6: Color {
        #if os(macOS)
        Color(nsColor: NSColor.controlBackgroundColor)
        #else
        Color(.systemGray6)
        #endif
    }
}

// MARK: - Highlight color helper

extension PlatformColor {
    static func highlightColor(from color: Color) -> PlatformColor {
        let alpha: CGFloat = 0.15
        switch color {
        case .yellow:
            return PlatformColor(red: 1.0, green: 1.0, blue: 0.0, alpha: alpha)
        case .green:
            return PlatformColor(red: 0.0, green: 1.0, blue: 0.0, alpha: alpha)
        case .red:
            return PlatformColor(red: 1.0, green: 0.35, blue: 0.35, alpha: alpha)
        case .orange:
            return PlatformColor(red: 1.0, green: 0.65, blue: 0.0, alpha: alpha)
        case .blue:
            return PlatformColor(red: 0.0, green: 0.5, blue: 1.0, alpha: alpha)
        case .purple:
            return PlatformColor(red: 0.6, green: 0.3, blue: 0.9, alpha: alpha)
        default:
            if let cgColor = color.cgColor,
               let components = cgColor.components,
               components.count >= 3 {
                return PlatformColor(red: components[0], green: components[1], blue: components[2], alpha: alpha)
            }
            return PlatformColor(red: 1.0, green: 1.0, blue: 0.0, alpha: alpha)
        }
    }
}

// MARK: - Cross-platform color extraction

extension Color {
    func rgbaComponents() -> (red: Double, green: Double, blue: Double, alpha: Double) {
        #if os(macOS)
        let nsColor = NSColor(self)
        let converted = nsColor.usingColorSpace(NSColorSpace.sRGB) ?? nsColor
        return (
            red: Double(converted.redComponent),
            green: Double(converted.greenComponent),
            blue: Double(converted.blueComponent),
            alpha: Double(converted.alphaComponent)
        )
        #else
        let uiColor = UIColor(self)
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        uiColor.getRed(&r, green: &g, blue: &b, alpha: &a)
        return (red: Double(r), green: Double(g), blue: Double(b), alpha: Double(a))
        #endif
    }
}
