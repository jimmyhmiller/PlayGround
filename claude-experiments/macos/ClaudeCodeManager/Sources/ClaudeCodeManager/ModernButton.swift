import AppKit

enum ModernButtonStyle {
    case primary
    case secondary
    case accent
    case ghost
    case danger
}

class ModernButton: NSView {
    var onPressed: (() -> Void)?
    
    private let style: ModernButtonStyle
    private let titleLabel = NSTextField(labelWithString: "")
    private let iconLabel = NSTextField(labelWithString: "")
    private var isHovered = false
    private var isPressed = false
    
    init(title: String = "", style: ModernButtonStyle = .primary) {
        self.style = style
        super.init(frame: .zero)
        setupViews()
        setTitle(title)
        setupStyling()
        addHoverTracking()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        wantsLayer = true
        
        titleLabel.isEditable = false
        titleLabel.isBordered = false
        titleLabel.backgroundColor = .clear
        titleLabel.alignment = .center
        titleLabel.font = DesignSystem.Typography.captionMedium
        
        iconLabel.isEditable = false
        iconLabel.isBordered = false
        iconLabel.backgroundColor = .clear
        iconLabel.alignment = .center
        iconLabel.font = NSFont.systemFont(ofSize: 12, weight: .medium)
        
        let stackView = NSStackView(views: [iconLabel, titleLabel])
        stackView.orientation = .horizontal
        stackView.spacing = DesignSystem.Spacing.xs
        stackView.alignment = .centerY
        
        addSubview(stackView)
        stackView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            stackView.centerXAnchor.constraint(equalTo: centerXAnchor),
            stackView.centerYAnchor.constraint(equalTo: centerYAnchor)
        ])
    }
    
    private func setupStyling() {
        layer?.cornerRadius = DesignSystem.CornerRadius.sm
        updateAppearance()
    }
    
    private func addHoverTracking() {
        let trackingArea = NSTrackingArea(
            rect: bounds,
            options: [.mouseEnteredAndExited, .activeAlways],
            owner: self,
            userInfo: nil
        )
        addTrackingArea(trackingArea)
    }
    
    func setTitle(_ title: String) {
        titleLabel.stringValue = title
        titleLabel.isHidden = title.isEmpty
    }
    
    func setIcon(_ systemName: String, size: CGFloat = 12) {
        // Convert SF Symbol names to Unicode equivalents
        let iconMap: [String: String] = [
            "plus": "+",
            "play.fill": "▶",
            "stop.fill": "⏹",
            "ellipsis": "⋯",
            "trash": "🗑",
            "folder": "📁",
            "checkmark": "✓",
            "xmark": "✕"
        ]
        
        iconLabel.stringValue = iconMap[systemName] ?? systemName
        iconLabel.font = NSFont.systemFont(ofSize: size, weight: .medium)
        iconLabel.isHidden = systemName.isEmpty
    }
    
    override func mouseEntered(with event: NSEvent) {
        isHovered = true
        updateAppearance()
    }
    
    override func mouseExited(with event: NSEvent) {
        isHovered = false
        updateAppearance()
    }
    
    override func mouseDown(with event: NSEvent) {
        isPressed = true
        updateAppearance()
    }
    
    override func mouseUp(with event: NSEvent) {
        isPressed = false
        updateAppearance()
        
        if bounds.contains(event.locationInWindow) {
            onPressed?()
        }
    }
    
    private func updateAppearance() {
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.15
            context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            
            let (backgroundColor, textColor, borderColor) = getColors()
            
            layer?.backgroundColor = backgroundColor.cgColor
            layer?.borderColor = borderColor?.cgColor
            layer?.borderWidth = borderColor != nil ? 1 : 0
            
            titleLabel.textColor = textColor
            iconLabel.textColor = textColor
            
            // Scale effect when pressed
            if isPressed {
                layer?.transform = CATransform3DMakeScale(0.95, 0.95, 1)
            } else {
                layer?.transform = CATransform3DIdentity
            }
        })
    }
    
    private func getColors() -> (backgroundColor: NSColor, textColor: NSColor, borderColor: NSColor?) {
        switch style {
        case .primary:
            if isPressed {
                return (DesignSystem.Colors.accent.withAlphaComponent(0.8), DesignSystem.Colors.textPrimary, nil)
            } else if isHovered {
                return (DesignSystem.Colors.accentHover, DesignSystem.Colors.textPrimary, nil)
            } else {
                return (DesignSystem.Colors.accent, DesignSystem.Colors.textPrimary, nil)
            }
            
        case .secondary:
            if isPressed {
                return (DesignSystem.Colors.surfaceElevated.withAlphaComponent(0.8), DesignSystem.Colors.textPrimary, DesignSystem.Colors.border)
            } else if isHovered {
                return (DesignSystem.Colors.surfaceElevated, DesignSystem.Colors.textPrimary, DesignSystem.Colors.accent)
            } else {
                return (DesignSystem.Colors.surfaceSecondary, DesignSystem.Colors.textSecondary, DesignSystem.Colors.border)
            }
            
        case .accent:
            if isPressed {
                return (DesignSystem.Colors.accentSecondary.withAlphaComponent(0.8), DesignSystem.Colors.accent, DesignSystem.Colors.accent)
            } else if isHovered {
                return (DesignSystem.Colors.accentSecondary, DesignSystem.Colors.accent, DesignSystem.Colors.accent)
            } else {
                return (.clear, DesignSystem.Colors.accent, DesignSystem.Colors.accent)
            }
            
        case .ghost:
            if isPressed {
                return (DesignSystem.Colors.surfaceSecondary.withAlphaComponent(0.5), DesignSystem.Colors.textSecondary, nil)
            } else if isHovered {
                return (DesignSystem.Colors.surfaceSecondary.withAlphaComponent(0.3), DesignSystem.Colors.textPrimary, nil)
            } else {
                return (.clear, DesignSystem.Colors.textTertiary, nil)
            }
            
        case .danger:
            if isPressed {
                return (DesignSystem.Colors.error.withAlphaComponent(0.8), DesignSystem.Colors.textPrimary, nil)
            } else if isHovered {
                return (DesignSystem.Colors.error.withAlphaComponent(0.9), DesignSystem.Colors.textPrimary, nil)
            } else {
                return (DesignSystem.Colors.error, DesignSystem.Colors.textPrimary, nil)
            }
        }
    }
}