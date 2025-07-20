import AppKit
import Combine

class ModernSidebarViewController: NSViewController {
    weak var delegate: SidebarViewControllerDelegate?
    
    let sessionManager: SessionManager
    private var scrollView: NSScrollView!
    private var containerView: NSView!
    private var headerView: ModernHeaderView!
    private var sessionCards: [ModernSessionCard] = []
    private var cancellables = Set<AnyCancellable>()
    
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
        view.layer?.backgroundColor = DesignSystem.Colors.surfacePrimary.cgColor
        
        setupScrollView()
        setupHeader()
        setupBindings()
        refreshSessionCards()
    }
    
    private func setupScrollView() {
        containerView = NSView()
        
        scrollView = NSScrollView()
        scrollView.documentView = containerView
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder
        scrollView.backgroundColor = .clear
        scrollView.drawsBackground = false
        
        view.addSubview(scrollView)
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            scrollView.topAnchor.constraint(equalTo: view.topAnchor),
            scrollView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }
    
    private func setupHeader() {
        headerView = ModernHeaderView()
        headerView.onAddPressed = { [weak self] in
            self?.addSession()
        }
        
        containerView.addSubview(headerView)
        headerView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            headerView.topAnchor.constraint(equalTo: containerView.topAnchor, constant: DesignSystem.Spacing.lg),
            headerView.leadingAnchor.constraint(equalTo: containerView.leadingAnchor, constant: DesignSystem.Spacing.lg),
            headerView.trailingAnchor.constraint(equalTo: containerView.trailingAnchor, constant: -DesignSystem.Spacing.lg),
            headerView.heightAnchor.constraint(equalToConstant: 60)
        ])
    }
    
    private func setupBindings() {
        sessionManager.$sessions
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.refreshSessionCards()
            }
            .store(in: &cancellables)
    }
    
    private func refreshSessionCards() {
        // Remove existing cards
        sessionCards.forEach { $0.removeFromSuperview() }
        sessionCards.removeAll()
        
        // Create new cards
        var previousCard: NSView = headerView
        
        for (_, session) in sessionManager.sessions.enumerated() {
            let card = ModernSessionCard(session: session)
            card.delegate = self
            sessionCards.append(card)
            containerView.addSubview(card)
            
            card.translatesAutoresizingMaskIntoConstraints = false
            NSLayoutConstraint.activate([
                card.topAnchor.constraint(equalTo: previousCard.bottomAnchor, constant: DesignSystem.Spacing.md),
                card.leadingAnchor.constraint(equalTo: containerView.leadingAnchor, constant: DesignSystem.Spacing.lg),
                card.trailingAnchor.constraint(equalTo: containerView.trailingAnchor, constant: -DesignSystem.Spacing.lg),
                card.heightAnchor.constraint(equalToConstant: 80)
            ])
            
            previousCard = card
        }
        
        // Add empty state if no sessions
        if sessionManager.sessions.isEmpty {
            let emptyStateView = ModernEmptyStateView()
            emptyStateView.onAddPressed = { [weak self] in
                self?.addSession()
            }
            containerView.addSubview(emptyStateView)
            emptyStateView.translatesAutoresizingMaskIntoConstraints = false
            NSLayoutConstraint.activate([
                emptyStateView.topAnchor.constraint(equalTo: headerView.bottomAnchor, constant: DesignSystem.Spacing.xxxl),
                emptyStateView.leadingAnchor.constraint(equalTo: containerView.leadingAnchor, constant: DesignSystem.Spacing.lg),
                emptyStateView.trailingAnchor.constraint(equalTo: containerView.trailingAnchor, constant: -DesignSystem.Spacing.lg),
                emptyStateView.heightAnchor.constraint(equalToConstant: 160)
            ])
            previousCard = emptyStateView
        }
        
        // Update container size
        NSLayoutConstraint.activate([
            containerView.widthAnchor.constraint(equalTo: scrollView.widthAnchor),
            containerView.bottomAnchor.constraint(equalTo: previousCard.bottomAnchor, constant: DesignSystem.Spacing.xxxl)
        ])
    }
    
    private func addSession() {
        print("Add session button pressed")
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Select Workspace Directory"
        panel.message = "Choose a folder that contains your project files"
        
        panel.begin { [weak self] response in
            if response == .OK, let url = panel.url {
                let name = url.lastPathComponent
                let path = url.path
                print("Adding workspace: \(name) at \(path)")
                self?.sessionManager.addSession(name: name, path: path)
            } else {
                print("User cancelled folder selection")
            }
        }
    }
}

extension ModernSidebarViewController: ModernSessionCardDelegate {
    func cardDidSelect(_ card: ModernSessionCard) {
        // Update selection state
        sessionCards.forEach { $0.setSelected(false) }
        card.setSelected(true)
        
        sessionManager.selectSession(card.session)
        delegate?.didSelectSession(card.session)
    }
    
    func cardDidRequestStart(_ card: ModernSessionCard) {
        sessionManager.startSession(card.session)
    }
    
    func cardDidRequestStop(_ card: ModernSessionCard) {
        sessionManager.stopSession(card.session)
    }
    
    func cardDidRequestRemove(_ card: ModernSessionCard) {
        sessionManager.removeSession(card.session)
    }
}

// MARK: - Modern Header View
class ModernHeaderView: NSView {
    var onAddPressed: (() -> Void)?
    
    private let titleLabel = NSTextField(labelWithString: "Workspaces")
    private let addButton = ModernButton(title: "", style: .accent)
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupViews()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        titleLabel.font = DesignSystem.Typography.title2
        titleLabel.textColor = DesignSystem.Colors.textPrimary
        
        addButton.setIcon("plus", size: 14)
        addButton.onPressed = { [weak self] in
            self?.onAddPressed?()
        }
        
        addSubview(titleLabel)
        addSubview(addButton)
        
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        addButton.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            titleLabel.leadingAnchor.constraint(equalTo: leadingAnchor),
            titleLabel.centerYAnchor.constraint(equalTo: centerYAnchor),
            
            addButton.trailingAnchor.constraint(equalTo: trailingAnchor),
            addButton.centerYAnchor.constraint(equalTo: centerYAnchor),
            addButton.widthAnchor.constraint(equalToConstant: 32),
            addButton.heightAnchor.constraint(equalToConstant: 32)
        ])
    }
}

// MARK: - Modern Empty State View
class ModernEmptyStateView: NSView {
    var onAddPressed: (() -> Void)?
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupViews()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        wantsLayer = true
        layer?.backgroundColor = DesignSystem.Colors.surfaceSecondary.withAlphaComponent(0.3).cgColor
        layer?.cornerRadius = DesignSystem.CornerRadius.lg
        layer?.borderColor = DesignSystem.Colors.borderLight.cgColor
        layer?.borderWidth = 1
        
        let iconLabel = NSTextField(labelWithString: "📁")
        iconLabel.font = NSFont.systemFont(ofSize: 40)
        iconLabel.alignment = .center
        
        let messageLabel = NSTextField(labelWithString: "No workspaces yet")
        messageLabel.font = DesignSystem.Typography.title3
        messageLabel.textColor = DesignSystem.Colors.textPrimary
        messageLabel.alignment = .center
        
        let subtitleLabel = NSTextField(labelWithString: "Add your first project folder to get started")
        subtitleLabel.font = DesignSystem.Typography.body
        subtitleLabel.textColor = DesignSystem.Colors.textSecondary
        subtitleLabel.alignment = .center
        
        let addButton = ModernButton(title: "Add Workspace", style: .accent)
        addButton.onPressed = { [weak self] in
            self?.onAddPressed?()
        }
        
        let stackView = NSStackView(views: [iconLabel, messageLabel, subtitleLabel, addButton])
        stackView.orientation = .vertical
        stackView.spacing = DesignSystem.Spacing.md
        stackView.alignment = .centerX
        
        addSubview(stackView)
        stackView.translatesAutoresizingMaskIntoConstraints = false
        addButton.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            stackView.centerXAnchor.constraint(equalTo: centerXAnchor),
            stackView.centerYAnchor.constraint(equalTo: centerYAnchor),
            addButton.widthAnchor.constraint(equalToConstant: 140),
            addButton.heightAnchor.constraint(equalToConstant: 32)
        ])
    }
}