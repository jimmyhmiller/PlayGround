import AppKit
import Combine

class ModernContentViewController: NSViewController, SidebarViewControllerDelegate, SimpleSidebarDelegate {
    private let sessionManager: SessionManager
    private var currentSession: WorkspaceSession?
    private var cancellables = Set<AnyCancellable>()
    
    private var heroSection: HeroSectionView!
    private var todoSection: ModernTodoSectionView!
    private var emptyStateView: ModernContentEmptyStateView!
    
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
        view.layer?.backgroundColor = DesignSystem.Colors.background.cgColor
        
        setupViews()
        setupConstraints()
        setupBindings()
        showEmptyState()
    }
    
    
    private func setupViews() {
        emptyStateView = ModernContentEmptyStateView()
        heroSection = HeroSectionView()
        todoSection = ModernTodoSectionView()
        
        view.addSubview(emptyStateView)
        view.addSubview(heroSection)
        view.addSubview(todoSection)
        
        heroSection.isHidden = true
        todoSection.isHidden = true
    }
    
    private func setupConstraints() {
        emptyStateView.translatesAutoresizingMaskIntoConstraints = false
        heroSection.translatesAutoresizingMaskIntoConstraints = false
        todoSection.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            emptyStateView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            emptyStateView.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            emptyStateView.widthAnchor.constraint(equalToConstant: 300),
            emptyStateView.heightAnchor.constraint(equalToConstant: 200),
            
            heroSection.topAnchor.constraint(equalTo: view.topAnchor, constant: 24),
            heroSection.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 24),
            heroSection.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -24),
            heroSection.heightAnchor.constraint(equalToConstant: 120),
            
            todoSection.topAnchor.constraint(equalTo: heroSection.bottomAnchor, constant: 20),
            todoSection.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 24),
            todoSection.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -24),
            todoSection.bottomAnchor.constraint(lessThanOrEqualTo: view.bottomAnchor, constant: -24)
        ])
    }
    
    private func setupBindings() {
        sessionManager.$selectedSession
            .receive(on: DispatchQueue.main)
            .sink { [weak self] session in
                self?.didSelectSession(session)
            }
            .store(in: &cancellables)
    }
    
    func didSelectSession(_ session: WorkspaceSession?) {
        currentSession = session
        
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.3
            context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            
            if let session = session {
                showSessionContent(session)
            } else {
                showEmptyState()
            }
        })
    }
    
    private func showEmptyState() {
        emptyStateView.isHidden = false
        heroSection.isHidden = true
        todoSection.isHidden = true
    }
    
    private func showSessionContent(_ session: WorkspaceSession) {
        emptyStateView.isHidden = true
        heroSection.isHidden = false
        todoSection.isHidden = false
        
        heroSection.configure(with: session)
        todoSection.configure(with: session, sessionManager: sessionManager)
    }
}


// MARK: - Hero Section View
class HeroSectionView: NSView {
    private let containerView = NSView()
    private let nameLabel = NSTextField(labelWithString: "")
    private let pathLabel = NSTextField(labelWithString: "")
    private let statusCard = StatusCardView()
    private let actionButton = ModernButton(title: "Open Terminal", style: .accent)
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupViews()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        addModernCardStyling()
        
        nameLabel.font = DesignSystem.Typography.title2
        nameLabel.textColor = DesignSystem.Colors.textPrimary
        nameLabel.isEditable = false
        nameLabel.isSelectable = false
        nameLabel.backgroundColor = .clear
        nameLabel.isBordered = false
        
        pathLabel.font = DesignSystem.Typography.subheadline
        pathLabel.textColor = DesignSystem.Colors.textSecondary
        pathLabel.lineBreakMode = .byTruncatingMiddle
        pathLabel.isEditable = false
        pathLabel.isSelectable = false
        pathLabel.backgroundColor = .clear
        pathLabel.isBordered = false
        
        actionButton.onPressed = { [weak self] in
            self?.openTerminal()
        }
        
        let leftStack = NSStackView(views: [nameLabel, pathLabel])
        leftStack.orientation = .vertical
        leftStack.spacing = 4
        leftStack.alignment = .leading
        
        let rightStack = NSStackView(views: [statusCard, actionButton])
        rightStack.orientation = .vertical
        rightStack.spacing = 12
        rightStack.alignment = .trailing
        
        containerView.addSubview(leftStack)
        containerView.addSubview(rightStack)
        addSubview(containerView)
        
        containerView.translatesAutoresizingMaskIntoConstraints = false
        leftStack.translatesAutoresizingMaskIntoConstraints = false
        rightStack.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            containerView.topAnchor.constraint(equalTo: topAnchor, constant: DesignSystem.Spacing.cardPadding),
            containerView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: DesignSystem.Spacing.cardPadding),
            containerView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -DesignSystem.Spacing.cardPadding),
            containerView.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -DesignSystem.Spacing.cardPadding),
            
            leftStack.leadingAnchor.constraint(equalTo: containerView.leadingAnchor),
            leftStack.centerYAnchor.constraint(equalTo: containerView.centerYAnchor),
            leftStack.trailingAnchor.constraint(lessThanOrEqualTo: rightStack.leadingAnchor, constant: -20),
            
            rightStack.trailingAnchor.constraint(equalTo: containerView.trailingAnchor),
            rightStack.centerYAnchor.constraint(equalTo: containerView.centerYAnchor)
        ])
    }
    
    func configure(with session: WorkspaceSession) {
        nameLabel.stringValue = session.name
        pathLabel.stringValue = session.path
        statusCard.setStatus(session.status)
    }
    
    private func openTerminal() {
        // Implementation to open terminal in workspace directory
        let script = """
        tell application "Terminal"
            activate
            do script "cd '\(pathLabel.stringValue)'"
        end tell
        """
        
        if let appleScript = NSAppleScript(source: script) {
            appleScript.executeAndReturnError(nil)
        }
    }
}

// MARK: - Status Card View
class StatusCardView: NSView {
    private let statusLabel = NSTextField(labelWithString: "")
    private let indicatorView = NSView()
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupViews()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        addSubtleCardStyling()
        
        statusLabel.font = DesignSystem.Typography.captionEmphasized
        statusLabel.textColor = DesignSystem.Colors.textSecondary
        statusLabel.isEditable = false
        statusLabel.isSelectable = false
        statusLabel.backgroundColor = .clear
        statusLabel.isBordered = false
        
        indicatorView.wantsLayer = true
        indicatorView.layer?.cornerRadius = 4
        
        addSubview(indicatorView)
        addSubview(statusLabel)
        
        indicatorView.translatesAutoresizingMaskIntoConstraints = false
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            widthAnchor.constraint(equalToConstant: 100),
            heightAnchor.constraint(equalToConstant: 32),
            
            indicatorView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 10),
            indicatorView.centerYAnchor.constraint(equalTo: centerYAnchor),
            indicatorView.widthAnchor.constraint(equalToConstant: 6),
            indicatorView.heightAnchor.constraint(equalToConstant: 6),
            
            statusLabel.leadingAnchor.constraint(equalTo: indicatorView.trailingAnchor, constant: 6),
            statusLabel.centerYAnchor.constraint(equalTo: centerYAnchor),
            statusLabel.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -10)
        ])
    }
    
    func setStatus(_ status: SessionStatus) {
        statusLabel.stringValue = status.rawValue
        indicatorView.layer?.backgroundColor = status.color.cgColor
    }
}

// MARK: - Modern Content Empty State View
class ModernContentEmptyStateView: NSView {
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupViews()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        addModernCardStyling()
        
        let titleLabel = NSTextField(labelWithString: "Select a workspace")
        titleLabel.font = DesignSystem.Typography.title2
        titleLabel.textColor = DesignSystem.Colors.textPrimary
        titleLabel.alignment = .center
        titleLabel.isEditable = false
        titleLabel.isSelectable = false
        titleLabel.backgroundColor = .clear
        titleLabel.isBordered = false
        
        let subtitleLabel = NSTextField(labelWithString: "Choose a workspace from the sidebar to start managing your Claude Code sessions")
        subtitleLabel.font = DesignSystem.Typography.body
        subtitleLabel.textColor = DesignSystem.Colors.textSecondary
        subtitleLabel.alignment = .center
        subtitleLabel.lineBreakMode = .byWordWrapping
        subtitleLabel.maximumNumberOfLines = 0
        subtitleLabel.isEditable = false
        subtitleLabel.isSelectable = false
        subtitleLabel.backgroundColor = .clear
        subtitleLabel.isBordered = false
        
        let stackView = NSStackView(views: [titleLabel, subtitleLabel])
        stackView.orientation = .vertical
        stackView.spacing = 8
        stackView.alignment = .centerX
        
        addSubview(stackView)
        stackView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            stackView.centerXAnchor.constraint(equalTo: centerXAnchor),
            stackView.centerYAnchor.constraint(equalTo: centerYAnchor),
            stackView.leadingAnchor.constraint(greaterThanOrEqualTo: leadingAnchor, constant: 20),
            stackView.trailingAnchor.constraint(lessThanOrEqualTo: trailingAnchor, constant: -20)
        ])
    }
}