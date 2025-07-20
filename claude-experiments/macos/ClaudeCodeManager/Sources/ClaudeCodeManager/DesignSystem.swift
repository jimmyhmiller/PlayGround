import AppKit
import CoreGraphics

enum DesignSystem {
    
    // MARK: - Colors
    enum Colors {
        // Adaptive colors that work with system appearance
        static let background = NSColor.name("AppBackground") ?? NSColor.windowBackgroundColor
        static let surfacePrimary = NSColor.name("SurfacePrimary") ?? NSColor.controlBackgroundColor
        static let surfaceSecondary = NSColor.name("SurfaceSecondary") ?? NSColor.controlColor
        static let surfaceElevated = NSColor.name("SurfaceElevated") ?? NSColor.quaternaryLabelColor.withAlphaComponent(0.1)
        
        // Modern accent colors with vibrancy
        static let accent = NSColor.systemBlue
        static let accentHover = NSColor.systemBlue.withSystemEffect(.pressed)
        static let accentSecondary = NSColor.systemBlue.withAlphaComponent(0.15)
        static let accentTertiary = NSColor.systemBlue.withAlphaComponent(0.08)
        
        // Status colors
        static let success = NSColor.systemGreen
        static let warning = NSColor.systemOrange
        static let error = NSColor.systemRed
        static let info = NSColor.systemPurple
        
        // Text colors that adapt to appearance
        static let textPrimary = NSColor.labelColor
        static let textSecondary = NSColor.secondaryLabelColor
        static let textTertiary = NSColor.tertiaryLabelColor
        static let textPlaceholder = NSColor.placeholderTextColor
        
        // Border and separator colors
        static let border = NSColor.separatorColor
        static let borderSubtle = NSColor.separatorColor.withAlphaComponent(0.5)
        static let borderStrong = NSColor.separatorColor.withAlphaComponent(0.8)
        
        // Glass morphism effects
        static let glassFill = NSColor.controlBackgroundColor.withAlphaComponent(0.8)
        static let glassStroke = NSColor.separatorColor.withAlphaComponent(0.3)
        
        // Interactive states
        static let hoverBackground = NSColor.controlAccentColor.withAlphaComponent(0.05)
        static let pressedBackground = NSColor.controlAccentColor.withAlphaComponent(0.1)
        static let selectedBackground = NSColor.selectedContentBackgroundColor
        
        // Card and surface shadows
        static let shadowColor = NSColor.black.withAlphaComponent(0.15)
        static let shadowColorStrong = NSColor.black.withAlphaComponent(0.25)
    }
    
    // MARK: - Typography
    enum Typography {
        // Improved font scale with better hierarchy
        static let largeTitle = NSFont.systemFont(ofSize: 28, weight: .bold)
        static let title1 = NSFont.systemFont(ofSize: 22, weight: .semibold)
        static let title2 = NSFont.systemFont(ofSize: 18, weight: .semibold)
        static let title3 = NSFont.systemFont(ofSize: 16, weight: .medium)
        static let headline = NSFont.systemFont(ofSize: 15, weight: .semibold)
        static let headlineEmphasized = NSFont.systemFont(ofSize: 15, weight: .bold)
        static let body = NSFont.systemFont(ofSize: 13, weight: .regular)
        static let bodyEmphasized = NSFont.systemFont(ofSize: 13, weight: .medium)
        static let subheadline = NSFont.systemFont(ofSize: 12, weight: .regular)
        static let subheadlineEmphasized = NSFont.systemFont(ofSize: 12, weight: .medium)
        static let footnote = NSFont.systemFont(ofSize: 11, weight: .regular)
        static let footnoteEmphasized = NSFont.systemFont(ofSize: 11, weight: .medium)
        static let caption = NSFont.systemFont(ofSize: 10, weight: .regular)
        static let captionEmphasized = NSFont.systemFont(ofSize: 10, weight: .medium)
        
        // Monospace fonts for code
        static let codeBody = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)
        static let codeSmall = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
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
        static let xxxxl: CGFloat = 40
        
        // Component-specific spacing
        static let cardPadding: CGFloat = 20
        static let sectionSpacing: CGFloat = 24
        static let itemSpacing: CGFloat = 8
    }
    
    // MARK: - Corner Radius
    enum CornerRadius {
        static let xs: CGFloat = 4
        static let sm: CGFloat = 6
        static let md: CGFloat = 8
        static let lg: CGFloat = 12
        static let xl: CGFloat = 16
        static let xxl: CGFloat = 20
        static let round: CGFloat = 999  // For pill-shaped elements
        
        // Component-specific radii
        static let button: CGFloat = 8
        static let card: CGFloat = 12
        static let modal: CGFloat = 16
    }
    
    // MARK: - Shadows
    enum Shadows {
        static func subtleShadow() -> NSShadow {
            let shadow = NSShadow()
            shadow.shadowColor = Colors.shadowColor
            shadow.shadowOffset = NSSize(width: 0, height: 1)
            shadow.shadowBlurRadius = 3
            return shadow
        }
        
        static func cardShadow() -> NSShadow {
            let shadow = NSShadow()
            shadow.shadowColor = Colors.shadowColor
            shadow.shadowOffset = NSSize(width: 0, height: 2)
            shadow.shadowBlurRadius = 8
            return shadow
        }
        
        static func elevatedShadow() -> NSShadow {
            let shadow = NSShadow()
            shadow.shadowColor = Colors.shadowColorStrong
            shadow.shadowOffset = NSSize(width: 0, height: 4)
            shadow.shadowBlurRadius = 16
            return shadow
        }
        
        static func modalShadow() -> NSShadow {
            let shadow = NSShadow()
            shadow.shadowColor = Colors.shadowColorStrong
            shadow.shadowOffset = NSSize(width: 0, height: 8)
            shadow.shadowBlurRadius = 32
            return shadow
        }
    }
}

// MARK: - Extensions
extension NSView {
    func addModernCardStyling(elevated: Bool = false) {
        wantsLayer = true
        layer?.backgroundColor = DesignSystem.Colors.surfacePrimary.cgColor
        layer?.cornerRadius = DesignSystem.CornerRadius.card
        layer?.borderColor = DesignSystem.Colors.borderSubtle.cgColor
        layer?.borderWidth = 0.5
        
        if elevated {
            shadow = DesignSystem.Shadows.elevatedShadow()
        } else {
            shadow = DesignSystem.Shadows.cardShadow()
        }
    }
    
    func addSubtleCardStyling() {
        wantsLayer = true
        layer?.backgroundColor = DesignSystem.Colors.surfaceSecondary.cgColor
        layer?.cornerRadius = DesignSystem.CornerRadius.md
        layer?.borderColor = DesignSystem.Colors.borderSubtle.cgColor
        layer?.borderWidth = 0.5
        shadow = DesignSystem.Shadows.subtleShadow()
    }
    
    func addGlassmorphismEffect() {
        wantsLayer = true
        layer?.backgroundColor = DesignSystem.Colors.glassFill.cgColor
        layer?.borderColor = DesignSystem.Colors.glassStroke.cgColor
        layer?.borderWidth = 1
        layer?.cornerRadius = DesignSystem.CornerRadius.lg
        shadow = DesignSystem.Shadows.subtleShadow()
    }
    
    func addHoverEffect(target: Any?, action: Selector?) {
        let trackingArea = NSTrackingArea(
            rect: bounds,
            options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect],
            owner: self,
            userInfo: nil
        )
        addTrackingArea(trackingArea)
    }
    
    func animateHover(isHovered: Bool) {
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.2
            context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            
            if isHovered {
                layer?.backgroundColor = DesignSystem.Colors.hoverBackground.cgColor
                layer?.transform = CATransform3DMakeScale(1.02, 1.02, 1)
            } else {
                layer?.backgroundColor = DesignSystem.Colors.surfacePrimary.cgColor
                layer?.transform = CATransform3DIdentity
            }
        })
    }
    
    func addPressedEffect() {
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.1
            layer?.transform = CATransform3DMakeScale(0.98, 0.98, 1)
        }) {
            NSAnimationContext.runAnimationGroup({ context in
                context.duration = 0.1
                self.layer?.transform = CATransform3DIdentity
            })
        }
    }
}