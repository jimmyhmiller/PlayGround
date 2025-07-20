import AppKit
import CoreGraphics

enum DesignSystem {
    
    // MARK: - Colors
    enum Colors {
        static let background = NSColor(red: 0.05, green: 0.05, blue: 0.07, alpha: 1.0)
        static let surfacePrimary = NSColor(red: 0.08, green: 0.08, blue: 0.12, alpha: 1.0)
        static let surfaceSecondary = NSColor(red: 0.12, green: 0.12, blue: 0.18, alpha: 1.0)
        static let surfaceElevated = NSColor(red: 0.15, green: 0.15, blue: 0.22, alpha: 1.0)
        
        static let accent = NSColor(red: 0.3, green: 0.6, blue: 1.0, alpha: 1.0)
        static let accentHover = NSColor(red: 0.25, green: 0.55, blue: 0.95, alpha: 1.0)
        static let accentSecondary = NSColor(red: 0.4, green: 0.4, blue: 1.0, alpha: 0.1)
        
        static let success = NSColor(red: 0.2, green: 0.8, blue: 0.4, alpha: 1.0)
        static let warning = NSColor(red: 1.0, green: 0.7, blue: 0.0, alpha: 1.0)
        static let error = NSColor(red: 1.0, green: 0.3, blue: 0.3, alpha: 1.0)
        
        static let textPrimary = NSColor(white: 0.95, alpha: 1.0)
        static let textSecondary = NSColor(white: 0.7, alpha: 1.0)
        static let textTertiary = NSColor(white: 0.5, alpha: 1.0)
        
        static let border = NSColor(white: 0.2, alpha: 1.0)
        static let borderLight = NSColor(white: 0.15, alpha: 1.0)
        
        static let glassFill = NSColor(white: 1.0, alpha: 0.05)
        static let glassStroke = NSColor(white: 1.0, alpha: 0.1)
    }
    
    // MARK: - Typography
    enum Typography {
        static let largeTitle = NSFont.systemFont(ofSize: 28, weight: .bold)
        static let title1 = NSFont.systemFont(ofSize: 22, weight: .semibold)
        static let title2 = NSFont.systemFont(ofSize: 18, weight: .semibold)
        static let title3 = NSFont.systemFont(ofSize: 16, weight: .medium)
        static let headline = NSFont.systemFont(ofSize: 14, weight: .semibold)
        static let body = NSFont.systemFont(ofSize: 13, weight: .regular)
        static let bodyMedium = NSFont.systemFont(ofSize: 13, weight: .medium)
        static let caption = NSFont.systemFont(ofSize: 11, weight: .regular)
        static let captionMedium = NSFont.systemFont(ofSize: 11, weight: .medium)
    }
    
    // MARK: - Spacing
    enum Spacing {
        static let xs: CGFloat = 4
        static let sm: CGFloat = 8
        static let md: CGFloat = 12
        static let lg: CGFloat = 16
        static let xl: CGFloat = 20
        static let xxl: CGFloat = 24
        static let xxxl: CGFloat = 32
    }
    
    // MARK: - Corner Radius
    enum CornerRadius {
        static let sm: CGFloat = 6
        static let md: CGFloat = 8
        static let lg: CGFloat = 12
        static let xl: CGFloat = 16
        static let xxl: CGFloat = 20
    }
    
    // MARK: - Shadows
    enum Shadows {
        static func cardShadow() -> NSShadow {
            let shadow = NSShadow()
            shadow.shadowColor = NSColor.black.withAlphaComponent(0.3)
            shadow.shadowOffset = NSSize(width: 0, height: -2)
            shadow.shadowBlurRadius = 8
            return shadow
        }
        
        static func elevatedShadow() -> NSShadow {
            let shadow = NSShadow()
            shadow.shadowColor = NSColor.black.withAlphaComponent(0.4)
            shadow.shadowOffset = NSSize(width: 0, height: -4)
            shadow.shadowBlurRadius = 16
            return shadow
        }
        
        static func subtleShadow() -> NSShadow {
            let shadow = NSShadow()
            shadow.shadowColor = NSColor.black.withAlphaComponent(0.2)
            shadow.shadowOffset = NSSize(width: 0, height: -1)
            shadow.shadowBlurRadius = 4
            return shadow
        }
    }
}

// MARK: - Extensions
extension NSView {
    func addGlassmorphismEffect() {
        wantsLayer = true
        layer?.backgroundColor = DesignSystem.Colors.glassFill.cgColor
        layer?.borderColor = DesignSystem.Colors.glassStroke.cgColor
        layer?.borderWidth = 1
        layer?.cornerRadius = DesignSystem.CornerRadius.lg
        
        // Add backdrop filter effect simulation
        let backdropView = NSView()
        backdropView.wantsLayer = true
        backdropView.layer?.backgroundColor = NSColor.black.withAlphaComponent(0.1).cgColor
        addSubview(backdropView, positioned: .below, relativeTo: nil)
        backdropView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            backdropView.topAnchor.constraint(equalTo: topAnchor),
            backdropView.leadingAnchor.constraint(equalTo: leadingAnchor),
            backdropView.trailingAnchor.constraint(equalTo: trailingAnchor),
            backdropView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
    }
    
    func addCardStyling() {
        wantsLayer = true
        layer?.backgroundColor = DesignSystem.Colors.surfaceElevated.cgColor
        layer?.cornerRadius = DesignSystem.CornerRadius.lg
        layer?.borderColor = DesignSystem.Colors.border.cgColor
        layer?.borderWidth = 1
        shadow = DesignSystem.Shadows.cardShadow()
    }
    
    func addHoverEffect() {
        let trackingArea = NSTrackingArea(
            rect: bounds,
            options: [.mouseEnteredAndExited, .activeAlways],
            owner: self,
            userInfo: nil
        )
        addTrackingArea(trackingArea)
    }
}