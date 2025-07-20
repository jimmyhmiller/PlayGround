import AppKit
import Combine

class ModernCleanSidebarViewController: NSViewController {
    weak var delegate: SidebarViewControllerDelegate?
    
    let sessionManager: SessionManager
    private var scrollView: NSScrollView!
    private var documentView: NSView!
    private var headerSection: SidebarHeaderSection!
    var workspacesSection: SidebarWorkspacesSection!
    private var cancellables = Set<AnyCancellable>()
    
    var isVisible = true {
        didSet {
            updateVisibility()
        }
    }
    
    init(sessionManager: SessionManager) {
        self.sessionManager = sessionManager
        super.init(nibName: nil, bundle: nil)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func loadView() {
        view = NSView()
        view.wantsLayer = true
        
        // Use proper sidebar background that adapts to system appearance
        if #available(macOS 10.14, *) {
            view.layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
        } else {
            view.layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
        }
        
        setupScrollView()
        setupSections()
        setupConstraints()
        setupBindings()
    }
    
    private func setupScrollView() {
        documentView = NSView()
        documentView.wantsLayer = true
        
        scrollView = NSScrollView()
        scrollView.documentView = documentView
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder
        scrollView.backgroundColor = .clear
        scrollView.drawsBackground = false
        scrollView.scrollerStyle = .overlay
        
        // Ensure proper scrolling behavior
        scrollView.verticalScrollElasticity = .allowed
        scrollView.horizontalScrollElasticity = .none
        
        view.addSubview(scrollView)
    }
    
    private func setupSections() {
        // Header with app name and controls
        headerSection = SidebarHeaderSection()
        headerSection.onMenuPressed = { [weak self] in
            self?.showMainMenu()
        }
        
        // Workspaces section
        workspacesSection = SidebarWorkspacesSection(sessionManager: sessionManager)
        workspacesSection.delegate = self
        
        documentView.addSubview(headerSection)
        documentView.addSubview(workspacesSection)
    }
    
    private func setupConstraints() {
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        documentView.translatesAutoresizingMaskIntoConstraints = false
        headerSection.translatesAutoresizingMaskIntoConstraints = false
        workspacesSection.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            // ScrollView fills the entire view
            scrollView.topAnchor.constraint(equalTo: view.topAnchor),
            scrollView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            // DocumentView width matches scrollView width, height is flexible
            documentView.widthAnchor.constraint(equalTo: scrollView.widthAnchor),
            documentView.topAnchor.constraint(equalTo: scrollView.topAnchor),
            documentView.leadingAnchor.constraint(equalTo: scrollView.leadingAnchor),
            documentView.trailingAnchor.constraint(equalTo: scrollView.trailingAnchor),
            
            // Header section - fixed at top with proper spacing
            headerSection.topAnchor.constraint(equalTo: documentView.topAnchor, constant: 16),
            headerSection.leadingAnchor.constraint(equalTo: documentView.leadingAnchor, constant: 16),
            headerSection.trailingAnchor.constraint(equalTo: documentView.trailingAnchor, constant: -16),
            headerSection.heightAnchor.constraint(equalToConstant: 54),
            
            // Workspaces section - takes remaining space
            workspacesSection.topAnchor.constraint(equalTo: headerSection.bottomAnchor, constant: 16),
            workspacesSection.leadingAnchor.constraint(equalTo: documentView.leadingAnchor, constant: 8),
            workspacesSection.trailingAnchor.constraint(equalTo: documentView.trailingAnchor, constant: -8),
            workspacesSection.bottomAnchor.constraint(equalTo: documentView.bottomAnchor, constant: -16)
        ])
    }
    
    private func setupBindings() {
        sessionManager.$sessions
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.workspacesSection.refreshWorkspaces()
            }
            .store(in: &cancellables)
    }
    
    private func updateVisibility() {
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.3
            context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            view.animator().isHidden = !isVisible
        })
    }
    
    private func showMainMenu() {
        let menu = NSMenu()
        
        let prefsItem = NSMenuItem(title: "Preferences...", action: #selector(showPreferences), keyEquivalent: ",")
        prefsItem.target = self
        menu.addItem(prefsItem)
        
        menu.addItem(NSMenuItem.separator())
        
        let aboutItem = NSMenuItem(title: "About Claude Code Manager", action: #selector(showAbout), keyEquivalent: "")
        aboutItem.target = self
        menu.addItem(aboutItem)
        
        if let button = headerSection.menuButton {
            menu.popUp(positioning: nil, at: NSPoint(x: 0, y: button.bounds.height), in: button)
        }
    }
    
    @objc private func showPreferences() {
        // Show preferences window
    }
    
    @objc private func showAbout() {
        NSApp.orderFrontStandardAboutPanel(nil)
    }
    
}

// MARK: - Sidebar Header Section
class SidebarHeaderSection: NSView {
    var onMenuPressed: (() -> Void)?
    var menuButton: NSButton?
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupViews()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        // App icon with proper sidebar styling
        let appIcon = NSImageView()
        appIcon.image = NSImage(systemSymbolName: "terminal.fill", accessibilityDescription: "Claude Code Manager")
        appIcon.contentTintColor = .controlAccentColor
        appIcon.imageScaling = .scaleProportionallyUpOrDown
        
        let titleLabel = NSTextField(labelWithString: "Claude Code")
        titleLabel.font = NSFont.systemFont(ofSize: 15, weight: .semibold)
        titleLabel.textColor = .labelColor
        titleLabel.isEditable = false
        titleLabel.isSelectable = false
        titleLabel.backgroundColor = .clear
        titleLabel.isBordered = false
        
        let subtitleLabel = NSTextField(labelWithString: "Session Manager")
        subtitleLabel.font = NSFont.systemFont(ofSize: 11, weight: .medium)
        subtitleLabel.textColor = .secondaryLabelColor
        subtitleLabel.isEditable = false
        subtitleLabel.isSelectable = false
        subtitleLabel.backgroundColor = .clear
        subtitleLabel.isBordered = false
        
        // More button with proper macOS styling
        menuButton = NSButton()
        menuButton?.image = NSImage(systemSymbolName: "ellipsis.circle", accessibilityDescription: "More Options")
        menuButton?.bezelStyle = .regularSquare
        menuButton?.isBordered = false
        menuButton?.imageScaling = .scaleProportionallyDown
        menuButton?.target = self
        menuButton?.action = #selector(menuPressed)
        
        // Add subtle hover effect
        if let button = menuButton {
            button.wantsLayer = true
            button.layer?.cornerRadius = 10
        }
        
        let textStack = NSStackView(views: [titleLabel, subtitleLabel])
        textStack.orientation = .vertical
        textStack.alignment = .leading
        textStack.spacing = 1
        textStack.distribution = .gravityAreas
        
        addSubview(appIcon)
        addSubview(textStack)
        if let menuButton = menuButton {
            addSubview(menuButton)
        }
        
        appIcon.translatesAutoresizingMaskIntoConstraints = false
        textStack.translatesAutoresizingMaskIntoConstraints = false
        menuButton?.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            appIcon.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 4),
            appIcon.centerYAnchor.constraint(equalTo: centerYAnchor),
            appIcon.widthAnchor.constraint(equalToConstant: 28),
            appIcon.heightAnchor.constraint(equalToConstant: 28),
            
            textStack.leadingAnchor.constraint(equalTo: appIcon.trailingAnchor, constant: 10),
            textStack.centerYAnchor.constraint(equalTo: centerYAnchor),
            textStack.trailingAnchor.constraint(lessThanOrEqualTo: menuButton?.leadingAnchor ?? trailingAnchor, constant: -8),
            
            menuButton?.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -4) ?? NSLayoutConstraint(),
            menuButton?.centerYAnchor.constraint(equalTo: centerYAnchor) ?? NSLayoutConstraint(),
            menuButton?.widthAnchor.constraint(equalToConstant: 20) ?? NSLayoutConstraint(),
            menuButton?.heightAnchor.constraint(equalToConstant: 20) ?? NSLayoutConstraint()
        ])
    }
    
    @objc private func menuPressed() {
        onMenuPressed?()
    }
}

// MARK: - Sidebar Workspaces Section
protocol SidebarWorkspacesSectionDelegate: AnyObject {
    func workspaceDidSelect(_ workspace: WorkspaceSession?)
}

class SidebarWorkspacesSection: NSView {
    weak var delegate: SidebarWorkspacesSectionDelegate?
    
    private let sessionManager: SessionManager
    private var workspaceCards: [WorkspaceCard] = []
    private var emptyStateView: WorkspaceEmptyStateView?
    
    init(sessionManager: SessionManager) {
        self.sessionManager = sessionManager
        super.init(frame: .zero)
        setupViews()
        refreshWorkspaces()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        let titleLabel = NSTextField(labelWithString: "WORKSPACES")
        titleLabel.font = NSFont.systemFont(ofSize: 11, weight: .medium)
        titleLabel.textColor = .secondaryLabelColor
        titleLabel.isEditable = false
        titleLabel.isSelectable = false
        titleLabel.backgroundColor = .clear
        titleLabel.isBordered = false
        
        // Add workspace button in the header
        let addButton = NSButton()
        addButton.image = NSImage(systemSymbolName: "plus", accessibilityDescription: "Add Workspace")
        addButton.bezelStyle = .regularSquare
        addButton.isBordered = false
        addButton.imageScaling = .scaleProportionallyDown
        addButton.target = self
        addButton.action = #selector(addWorkspacePressed)
        addButton.alphaValue = 0.7
        
        addSubview(titleLabel)
        addSubview(addButton)
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        addButton.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            titleLabel.topAnchor.constraint(equalTo: topAnchor),
            titleLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 12),
            
            addButton.centerYAnchor.constraint(equalTo: titleLabel.centerYAnchor),
            addButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -12),
            addButton.widthAnchor.constraint(equalToConstant: 16),
            addButton.heightAnchor.constraint(equalToConstant: 16)
        ])
    }
    
    @objc private func addWorkspacePressed() {
        addWorkspace()
    }
    
    func refreshWorkspaces() {
        // Remove existing cards
        workspaceCards.forEach { $0.removeFromSuperview() }
        workspaceCards.removeAll()
        emptyStateView?.removeFromSuperview()
        emptyStateView = nil
        
        if sessionManager.sessions.isEmpty {
            showEmptyState()
        } else {
            createWorkspaceCards()
        }
        
        layoutCards()
    }
    
    private func showEmptyState() {
        emptyStateView = WorkspaceEmptyStateView()
        emptyStateView?.onAddPressed = { [weak self] in
            self?.addWorkspace()
        }
        
        if let emptyView = emptyStateView {
            addSubview(emptyView)
            emptyView.translatesAutoresizingMaskIntoConstraints = false
            
            NSLayoutConstraint.activate([
                emptyView.topAnchor.constraint(equalTo: topAnchor, constant: 24),
                emptyView.leadingAnchor.constraint(equalTo: leadingAnchor),
                emptyView.trailingAnchor.constraint(equalTo: trailingAnchor),
                emptyView.heightAnchor.constraint(equalToConstant: 60),
                emptyView.bottomAnchor.constraint(equalTo: bottomAnchor)
            ])
        }
    }
    
    private func createWorkspaceCards() {
        for session in sessionManager.sessions {
            let card = WorkspaceCard(session: session)
            card.delegate = self
            workspaceCards.append(card)
            addSubview(card)
        }
    }
    
    private func layoutCards() {
        guard !workspaceCards.isEmpty else { return }
        
        var previousCard: NSView?
        let cardSpacing: CGFloat = 1 // Tight spacing like macOS sidebars
        
        for card in workspaceCards {
            card.translatesAutoresizingMaskIntoConstraints = false
            
            let topConstraint = previousCard == nil
                ? card.topAnchor.constraint(equalTo: topAnchor, constant: 24) // Leave space for header
                : card.topAnchor.constraint(equalTo: previousCard!.bottomAnchor, constant: cardSpacing)
            
            NSLayoutConstraint.activate([
                topConstraint,
                card.leadingAnchor.constraint(equalTo: leadingAnchor),
                card.trailingAnchor.constraint(equalTo: trailingAnchor),
                card.heightAnchor.constraint(equalToConstant: 28) // Tighter row height
            ])
            
            previousCard = card
        }
        
        if let lastCard = previousCard {
            NSLayoutConstraint.activate([
                bottomAnchor.constraint(equalTo: lastCard.bottomAnchor, constant: 8)
            ])
        }
    }
    
    func addWorkspace() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Select"
        panel.message = "Choose a workspace directory"
        
        panel.begin { [weak self] response in
            guard response == .OK, let url = panel.url else { return }
            
            let name = url.lastPathComponent
            let path = url.path
            
            self?.sessionManager.addSession(name: name, path: path)
        }
    }
}

// MARK: - Workspace Card
protocol WorkspaceCardDelegate: AnyObject {
    func cardDidSelect(_ card: WorkspaceCard)
    func cardDidRequestToggle(_ card: WorkspaceCard)
}

class WorkspaceCard: NSView {
    weak var delegate: WorkspaceCardDelegate?
    let session: WorkspaceSession
    
    private var isSelected = false
    private var isHovered = false
    
    init(session: WorkspaceSession) {
        self.session = session
        super.init(frame: .zero)
        setupViews()
        addHoverTracking()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        wantsLayer = true
        layer?.cornerRadius = 6
        layer?.backgroundColor = NSColor.clear.cgColor
        
        // Folder icon like macOS Finder sidebar
        let folderIcon = NSImageView()
        folderIcon.image = NSImage(systemSymbolName: "folder.fill", accessibilityDescription: "Workspace")
        folderIcon.contentTintColor = .systemBlue
        folderIcon.imageScaling = .scaleProportionallyUpOrDown
        
        // Status indicator as a subtle badge
        let statusDot = NSView()
        statusDot.wantsLayer = true
        statusDot.layer?.cornerRadius = 3
        statusDot.layer?.backgroundColor = session.status.color.cgColor
        
        let nameLabel = NSTextField(labelWithString: session.name)
        nameLabel.font = NSFont.systemFont(ofSize: 13, weight: .medium)
        nameLabel.textColor = .labelColor
        nameLabel.lineBreakMode = .byTruncatingTail
        nameLabel.isEditable = false
        nameLabel.isSelectable = false
        nameLabel.backgroundColor = .clear
        nameLabel.isBordered = false
        
        // Subtle action button
        let actionButton = NSButton()
        let isActive = session.status == .active
        actionButton.image = NSImage(systemSymbolName: isActive ? "stop.circle" : "play.circle", 
                                   accessibilityDescription: isActive ? "Stop" : "Start")
        actionButton.bezelStyle = .regularSquare
        actionButton.isBordered = false
        actionButton.imageScaling = .scaleProportionallyDown
        actionButton.target = self
        actionButton.action = #selector(toggleAction)
        actionButton.alphaValue = 0.7
        
        addSubview(folderIcon)
        addSubview(statusDot)
        addSubview(nameLabel)
        addSubview(actionButton)
        
        folderIcon.translatesAutoresizingMaskIntoConstraints = false
        statusDot.translatesAutoresizingMaskIntoConstraints = false
        nameLabel.translatesAutoresizingMaskIntoConstraints = false
        actionButton.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            folderIcon.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 12),
            folderIcon.centerYAnchor.constraint(equalTo: centerYAnchor),
            folderIcon.widthAnchor.constraint(equalToConstant: 16),
            folderIcon.heightAnchor.constraint(equalToConstant: 16),
            
            statusDot.leadingAnchor.constraint(equalTo: folderIcon.trailingAnchor, constant: -2),
            statusDot.topAnchor.constraint(equalTo: folderIcon.topAnchor, constant: -2),
            statusDot.widthAnchor.constraint(equalToConstant: 6),
            statusDot.heightAnchor.constraint(equalToConstant: 6),
            
            nameLabel.leadingAnchor.constraint(equalTo: folderIcon.trailingAnchor, constant: 8),
            nameLabel.centerYAnchor.constraint(equalTo: centerYAnchor),
            nameLabel.trailingAnchor.constraint(equalTo: actionButton.leadingAnchor, constant: -8),
            
            actionButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -8),
            actionButton.centerYAnchor.constraint(equalTo: centerYAnchor),
            actionButton.widthAnchor.constraint(equalToConstant: 16),
            actionButton.heightAnchor.constraint(equalToConstant: 16)
        ])
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
            context.duration = 0.15
            context.timingFunction = CAMediaTimingFunction(name: .easeOut)
            
            if isSelected {
                layer?.backgroundColor = NSColor.selectedContentBackgroundColor.cgColor
            } else if isHovered {
                layer?.backgroundColor = NSColor.controlAccentColor.withAlphaComponent(0.1).cgColor
            } else {
                layer?.backgroundColor = NSColor.clear.cgColor
            }
        })
    }
    
    @objc private func toggleAction() {
        delegate?.cardDidRequestToggle(self)
    }
}

// MARK: - Extensions
extension SidebarWorkspacesSection: WorkspaceCardDelegate {
    func cardDidSelect(_ card: WorkspaceCard) {
        workspaceCards.forEach { $0.setSelected(false) }
        card.setSelected(true)
        
        sessionManager.selectSession(card.session)
        delegate?.workspaceDidSelect(card.session)
    }
    
    func cardDidRequestToggle(_ card: WorkspaceCard) {
        if card.session.status == .active {
            sessionManager.stopSession(card.session)
        } else {
            sessionManager.startSession(card.session)
        }
    }
}

extension ModernCleanSidebarViewController: SidebarWorkspacesSectionDelegate {
    func workspaceDidSelect(_ workspace: WorkspaceSession?) {
        delegate?.didSelectSession(workspace)
    }
}


// MARK: - Workspace Empty State
class WorkspaceEmptyStateView: NSView {
    var onAddPressed: (() -> Void)?
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupViews()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        let messageLabel = NSTextField(labelWithString: "No workspaces")
        messageLabel.font = NSFont.systemFont(ofSize: 13, weight: .medium)
        messageLabel.textColor = .tertiaryLabelColor
        messageLabel.alignment = .center
        messageLabel.isEditable = false
        messageLabel.isSelectable = false
        messageLabel.backgroundColor = .clear
        messageLabel.isBordered = false
        
        let subtitleLabel = NSTextField(labelWithString: "Add a project to get started")
        subtitleLabel.font = NSFont.systemFont(ofSize: 11, weight: .regular)
        subtitleLabel.textColor = .quaternaryLabelColor
        subtitleLabel.alignment = .center
        subtitleLabel.isEditable = false
        subtitleLabel.isSelectable = false
        subtitleLabel.backgroundColor = .clear
        subtitleLabel.isBordered = false
        
        let stackView = NSStackView(views: [messageLabel, subtitleLabel])
        stackView.orientation = .vertical
        stackView.spacing = 2
        stackView.alignment = .centerX
        
        addSubview(stackView)
        stackView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            stackView.centerXAnchor.constraint(equalTo: centerXAnchor),
            stackView.centerYAnchor.constraint(equalTo: centerYAnchor)
        ])
    }
    
    @objc private func addPressed() {
        onAddPressed?()
    }
}