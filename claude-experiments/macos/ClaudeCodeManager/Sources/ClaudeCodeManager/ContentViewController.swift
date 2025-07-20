import AppKit
import Combine

class ContentViewController: NSViewController, SidebarViewControllerDelegate {
    private let sessionManager: SessionManager
    private var currentSession: WorkspaceSession?
    private var cancellables = Set<AnyCancellable>()
    
    private var sessionInfoView: SessionInfoView!
    private var todoListView: TodoListView!
    private var emptyStateView: EmptyStateView!
    
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
        view.layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
        
        setupViews()
        setupConstraints()
        setupBindings()
        showEmptyState()
    }
    
    private func setupViews() {
        emptyStateView = EmptyStateView()
        sessionInfoView = SessionInfoView()
        todoListView = TodoListView()
        
        view.addSubview(emptyStateView)
        view.addSubview(sessionInfoView)
        view.addSubview(todoListView)
        
        sessionInfoView.isHidden = true
        todoListView.isHidden = true
    }
    
    private func setupConstraints() {
        emptyStateView.translatesAutoresizingMaskIntoConstraints = false
        sessionInfoView.translatesAutoresizingMaskIntoConstraints = false
        todoListView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            emptyStateView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            emptyStateView.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            
            sessionInfoView.topAnchor.constraint(equalTo: view.topAnchor),
            sessionInfoView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            sessionInfoView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            sessionInfoView.heightAnchor.constraint(equalToConstant: 120),
            
            todoListView.topAnchor.constraint(equalTo: sessionInfoView.bottomAnchor),
            todoListView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            todoListView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            todoListView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
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
        
        if let session = session {
            showSessionContent(session)
        } else {
            showEmptyState()
        }
    }
    
    private func showEmptyState() {
        emptyStateView.isHidden = false
        sessionInfoView.isHidden = true
        todoListView.isHidden = true
    }
    
    private func showSessionContent(_ session: WorkspaceSession) {
        emptyStateView.isHidden = true
        sessionInfoView.isHidden = false
        todoListView.isHidden = false
        
        sessionInfoView.configure(with: session)
        todoListView.configure(with: session, sessionManager: sessionManager)
    }
}

class EmptyStateView: NSView {
    private let messageLabel = NSTextField(labelWithString: "Select a workspace to get started")
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupViews()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        messageLabel.font = NSFont.systemFont(ofSize: 16)
        messageLabel.textColor = .secondaryLabelColor
        messageLabel.alignment = .center
        
        addSubview(messageLabel)
        
        messageLabel.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            messageLabel.centerXAnchor.constraint(equalTo: centerXAnchor),
            messageLabel.centerYAnchor.constraint(equalTo: centerYAnchor)
        ])
    }
}

class SessionInfoView: NSView {
    private let nameLabel = NSTextField(labelWithString: "")
    private let pathLabel = NSTextField(labelWithString: "")
    private let statusLabel = NSTextField(labelWithString: "")
    private let lastUsedLabel = NSTextField(labelWithString: "")
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupViews()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        wantsLayer = true
        layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
        
        nameLabel.font = NSFont.systemFont(ofSize: 18, weight: .semibold)
        pathLabel.font = NSFont.systemFont(ofSize: 12)
        pathLabel.textColor = .secondaryLabelColor
        statusLabel.font = NSFont.systemFont(ofSize: 12, weight: .medium)
        lastUsedLabel.font = NSFont.systemFont(ofSize: 12)
        lastUsedLabel.textColor = .secondaryLabelColor
        
        let stackView = NSStackView(views: [nameLabel, pathLabel, statusLabel, lastUsedLabel])
        stackView.orientation = .vertical
        stackView.alignment = .leading
        stackView.spacing = 4
        
        addSubview(stackView)
        
        stackView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            stackView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 16),
            stackView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -16),
            stackView.centerYAnchor.constraint(equalTo: centerYAnchor)
        ])
    }
    
    func configure(with session: WorkspaceSession) {
        nameLabel.stringValue = session.name
        pathLabel.stringValue = session.path
        statusLabel.stringValue = "Status: \(session.status.rawValue)"
        statusLabel.textColor = session.status.color
        
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        lastUsedLabel.stringValue = "Last used: \(formatter.string(from: session.lastUsed))"
    }
}