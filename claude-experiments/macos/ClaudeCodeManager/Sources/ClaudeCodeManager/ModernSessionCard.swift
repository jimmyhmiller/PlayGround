import AppKit

protocol ModernSessionCardDelegate: AnyObject {
    func cardDidSelect(_ card: ModernSessionCard)
    func cardDidRequestStart(_ card: ModernSessionCard)
    func cardDidRequestStop(_ card: ModernSessionCard)
    func cardDidRequestRemove(_ card: ModernSessionCard)
}

class ModernSessionCard: NSView {
    weak var delegate: ModernSessionCardDelegate?
    let session: WorkspaceSession
    
    private let nameLabel = NSTextField(labelWithString: "")
    private let pathLabel = NSTextField(labelWithString: "")
    private let statusIndicator = StatusIndicatorView()
    private let actionButton = ModernButton(title: "", style: .secondary)
    private let contextButton = ModernButton(title: "", style: .ghost)
    
    private var isSelected = false
    private var isHovered = false
    
    init(session: WorkspaceSession) {
        self.session = session
        super.init(frame: .zero)
        setupViews()
        configure()
        addHoverTracking()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        wantsLayer = true
        addModernCardStyling()
        
        nameLabel.font = DesignSystem.Typography.bodyEmphasized
        nameLabel.textColor = DesignSystem.Colors.textPrimary
        nameLabel.lineBreakMode = .byTruncatingTail
        
        pathLabel.font = DesignSystem.Typography.caption
        pathLabel.textColor = DesignSystem.Colors.textSecondary
        pathLabel.lineBreakMode = .byTruncatingMiddle
        
        actionButton.setIcon("play.fill", size: 12)
        actionButton.onPressed = { [weak self] in
            guard let self = self else { return }
            if self.session.status == .active {
                self.delegate?.cardDidRequestStop(self)
            } else {
                self.delegate?.cardDidRequestStart(self)
            }
        }
        
        contextButton.setIcon("ellipsis", size: 12)
        contextButton.onPressed = { [weak self] in
            self?.showContextMenu()
        }
        
        addSubview(statusIndicator)
        addSubview(nameLabel)
        addSubview(pathLabel)
        addSubview(actionButton)
        addSubview(contextButton)
        
        statusIndicator.translatesAutoresizingMaskIntoConstraints = false
        nameLabel.translatesAutoresizingMaskIntoConstraints = false
        pathLabel.translatesAutoresizingMaskIntoConstraints = false
        actionButton.translatesAutoresizingMaskIntoConstraints = false
        contextButton.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            statusIndicator.leadingAnchor.constraint(equalTo: leadingAnchor, constant: DesignSystem.Spacing.lg),
            statusIndicator.centerYAnchor.constraint(equalTo: centerYAnchor),
            statusIndicator.widthAnchor.constraint(equalToConstant: 12),
            statusIndicator.heightAnchor.constraint(equalToConstant: 12),
            
            nameLabel.leadingAnchor.constraint(equalTo: statusIndicator.trailingAnchor, constant: DesignSystem.Spacing.md),
            nameLabel.topAnchor.constraint(equalTo: topAnchor, constant: DesignSystem.Spacing.lg),
            nameLabel.trailingAnchor.constraint(equalTo: contextButton.leadingAnchor, constant: -DesignSystem.Spacing.md),
            
            pathLabel.leadingAnchor.constraint(equalTo: nameLabel.leadingAnchor),
            pathLabel.topAnchor.constraint(equalTo: nameLabel.bottomAnchor, constant: DesignSystem.Spacing.xs),
            pathLabel.trailingAnchor.constraint(equalTo: nameLabel.trailingAnchor),
            
            contextButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -DesignSystem.Spacing.md),
            contextButton.topAnchor.constraint(equalTo: topAnchor, constant: DesignSystem.Spacing.md),
            contextButton.widthAnchor.constraint(equalToConstant: 24),
            contextButton.heightAnchor.constraint(equalToConstant: 24),
            
            actionButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -DesignSystem.Spacing.lg),
            actionButton.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -DesignSystem.Spacing.md),
            actionButton.widthAnchor.constraint(equalToConstant: 60),
            actionButton.heightAnchor.constraint(equalToConstant: 24)
        ])
    }
    
    private func configure() {
        nameLabel.stringValue = session.name
        pathLabel.stringValue = session.path
        statusIndicator.setStatus(session.status)
        
        let isActive = session.status == .active
        actionButton.setTitle(isActive ? "Stop" : "Start")
        actionButton.setIcon(isActive ? "stop.fill" : "play.fill", size: 12)
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
    
    override func mouseEntered(with event: NSEvent) {
        isHovered = true
        updateAppearance()
    }
    
    override func mouseExited(with event: NSEvent) {
        isHovered = false
        updateAppearance()
    }
    
    override func mouseDown(with event: NSEvent) {
        delegate?.cardDidSelect(self)
    }
    
    func setSelected(_ selected: Bool) {
        isSelected = selected
        updateAppearance()
    }
    
    private func updateAppearance() {
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.2
            context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            
            if isSelected {
                layer?.backgroundColor = DesignSystem.Colors.accent.withAlphaComponent(0.1).cgColor
                layer?.borderColor = DesignSystem.Colors.accent.cgColor
                layer?.borderWidth = 2
            } else if isHovered {
                layer?.backgroundColor = DesignSystem.Colors.surfaceElevated.withAlphaComponent(0.8).cgColor
                layer?.borderColor = DesignSystem.Colors.border.cgColor
                layer?.borderWidth = 1
            } else {
                layer?.backgroundColor = DesignSystem.Colors.surfaceElevated.cgColor
                layer?.borderColor = DesignSystem.Colors.borderSubtle.cgColor
                layer?.borderWidth = 1
            }
        })
    }
    
    private func showContextMenu() {
        let menu = NSMenu()
        
        let removeItem = NSMenuItem(title: "Remove Workspace", action: #selector(removeSession), keyEquivalent: "")
        removeItem.target = self
        menu.addItem(removeItem)
        
        let revealItem = NSMenuItem(title: "Reveal in Finder", action: #selector(revealInFinder), keyEquivalent: "")
        revealItem.target = self
        menu.addItem(revealItem)
        
        menu.popUp(positioning: nil, at: NSPoint(x: bounds.width - 10, y: bounds.height), in: self)
    }
    
    @objc private func removeSession() {
        delegate?.cardDidRequestRemove(self)
    }
    
    @objc private func revealInFinder() {
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: session.path)
    }
}

// MARK: - Status Indicator View
class StatusIndicatorView: NSView {
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer?.cornerRadius = 6
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    func setStatus(_ status: SessionStatus) {
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.3
            
            switch status {
            case .active:
                layer?.backgroundColor = DesignSystem.Colors.success.cgColor
            case .inactive:
                layer?.backgroundColor = DesignSystem.Colors.textTertiary.cgColor
            case .error:
                layer?.backgroundColor = DesignSystem.Colors.error.cgColor
            }
        })
        
        // Add subtle pulsing animation for active status
        if status == .active {
            let pulseAnimation = CABasicAnimation(keyPath: "opacity")
            pulseAnimation.fromValue = 1.0
            pulseAnimation.toValue = 0.6
            pulseAnimation.duration = 1.0
            pulseAnimation.repeatCount = .infinity
            pulseAnimation.autoreverses = true
            layer?.add(pulseAnimation, forKey: "pulse")
        } else {
            layer?.removeAnimation(forKey: "pulse")
        }
    }
}