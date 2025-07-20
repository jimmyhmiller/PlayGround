import AppKit
import Combine

class StandardSidebarViewController: NSViewController {
    weak var delegate: SidebarViewControllerDelegate?
    
    let sessionManager: SessionManager
    private var tableView: NSTableView!
    private var scrollView: NSScrollView!
    private var headerView: NSView!
    private var buttonContainer: NSView!
    private var addButton: NSButton!
    private var removeButton: NSButton!
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
        view.layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
        
        setupHeaderView()
        setupTableView()
        setupButtonContainer()
        setupConstraints()
        setupBindings()
    }
    
    private func setupHeaderView() {
        headerView = NSView()
        
        let titleLabel = NSTextField(labelWithString: "Workspaces")
        titleLabel.font = NSFont.systemFont(ofSize: 13, weight: .semibold)
        titleLabel.textColor = .labelColor
        
        headerView.addSubview(titleLabel)
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            titleLabel.leadingAnchor.constraint(equalTo: headerView.leadingAnchor, constant: 8),
            titleLabel.centerYAnchor.constraint(equalTo: headerView.centerYAnchor)
        ])
        
        view.addSubview(headerView)
    }
    
    private func setupTableView() {
        tableView = NSTableView()
        tableView.style = .sourceList
        tableView.headerView = nil
        tableView.rowSizeStyle = .medium
        tableView.allowsEmptySelection = true
        tableView.delegate = self
        tableView.dataSource = self
        tableView.menu = createContextMenu()
        
        let column = NSTableColumn(identifier: NSUserInterfaceItemIdentifier("SessionColumn"))
        column.title = "Sessions"
        tableView.addTableColumn(column)
        
        scrollView = NSScrollView()
        scrollView.documentView = tableView
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder
        
        view.addSubview(scrollView)
    }
    
    private func setupButtonContainer() {
        buttonContainer = NSView()
        
        // Standard macOS + and - buttons like in Finder
        addButton = NSButton()
        addButton.image = NSImage(systemSymbolName: "plus", accessibilityDescription: "Add Workspace")
        addButton.bezelStyle = .texturedRounded
        addButton.isBordered = true
        addButton.target = self
        addButton.action = #selector(addWorkspace)
        addButton.toolTip = "Add Workspace"
        
        removeButton = NSButton()
        removeButton.image = NSImage(systemSymbolName: "minus", accessibilityDescription: "Remove Workspace")
        removeButton.bezelStyle = .texturedRounded
        removeButton.isBordered = true
        removeButton.target = self
        removeButton.action = #selector(removeWorkspace)
        removeButton.toolTip = "Remove Workspace"
        removeButton.isEnabled = false
        
        buttonContainer.addSubview(addButton)
        buttonContainer.addSubview(removeButton)
        
        addButton.translatesAutoresizingMaskIntoConstraints = false
        removeButton.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            addButton.leadingAnchor.constraint(equalTo: buttonContainer.leadingAnchor, constant: 4),
            addButton.centerYAnchor.constraint(equalTo: buttonContainer.centerYAnchor),
            addButton.widthAnchor.constraint(equalToConstant: 22),
            addButton.heightAnchor.constraint(equalToConstant: 22),
            
            removeButton.leadingAnchor.constraint(equalTo: addButton.trailingAnchor, constant: 2),
            removeButton.centerYAnchor.constraint(equalTo: buttonContainer.centerYAnchor),
            removeButton.widthAnchor.constraint(equalToConstant: 22),
            removeButton.heightAnchor.constraint(equalToConstant: 22)
        ])
        
        view.addSubview(buttonContainer)
    }
    
    private func setupConstraints() {
        headerView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        buttonContainer.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            // Header at top
            headerView.topAnchor.constraint(equalTo: view.topAnchor),
            headerView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            headerView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            headerView.heightAnchor.constraint(equalToConstant: 32),
            
            // Table view in middle
            scrollView.topAnchor.constraint(equalTo: headerView.bottomAnchor),
            scrollView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: buttonContainer.topAnchor),
            
            // Buttons at bottom
            buttonContainer.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            buttonContainer.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            buttonContainer.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            buttonContainer.heightAnchor.constraint(equalToConstant: 32)
        ])
    }
    
    private func setupBindings() {
        sessionManager.$sessions
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.tableView.reloadData()
            }
            .store(in: &cancellables)
    }
    
    private func createContextMenu() -> NSMenu {
        let menu = NSMenu()
        
        let addItem = NSMenuItem(title: "Add Workspace...", action: #selector(addWorkspace), keyEquivalent: "")
        addItem.target = self
        menu.addItem(addItem)
        
        menu.addItem(NSMenuItem.separator())
        
        let removeItem = NSMenuItem(title: "Remove Workspace", action: #selector(removeWorkspace), keyEquivalent: "")
        removeItem.target = self
        menu.addItem(removeItem)
        
        let revealItem = NSMenuItem(title: "Reveal in Finder", action: #selector(revealInFinder), keyEquivalent: "")
        revealItem.target = self
        menu.addItem(revealItem)
        
        return menu
    }
    
    @objc private func addWorkspace() {
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
    
    @objc private func removeWorkspace() {
        let selectedRow = tableView.selectedRow
        guard selectedRow >= 0, selectedRow < sessionManager.sessions.count else { return }
        
        let session = sessionManager.sessions[selectedRow]
        sessionManager.removeSession(session)
    }
    
    @objc private func revealInFinder() {
        let selectedRow = tableView.selectedRow
        guard selectedRow >= 0, selectedRow < sessionManager.sessions.count else { return }
        
        let session = sessionManager.sessions[selectedRow]
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: session.path)
    }
}

// MARK: - Table View Data Source
extension StandardSidebarViewController: NSTableViewDataSource {
    func numberOfRows(in tableView: NSTableView) -> Int {
        return sessionManager.sessions.count
    }
}

// MARK: - Table View Delegate
extension StandardSidebarViewController: NSTableViewDelegate {
    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView? {
        let session = sessionManager.sessions[row]
        
        let cellView = StandardSessionCellView()
        cellView.configure(with: session, sessionManager: sessionManager)
        return cellView
    }
    
    func tableViewSelectionDidChange(_ notification: Notification) {
        let selectedRow = tableView.selectedRow
        removeButton.isEnabled = selectedRow >= 0
        
        if selectedRow >= 0 && selectedRow < sessionManager.sessions.count {
            let session = sessionManager.sessions[selectedRow]
            sessionManager.selectSession(session)
            delegate?.didSelectSession(session)
        } else {
            delegate?.didSelectSession(nil)
        }
    }
}

// MARK: - Standard Session Cell View
class StandardSessionCellView: NSTableCellView {
    private let nameLabel = NSTextField(labelWithString: "")
    private let pathLabel = NSTextField(labelWithString: "")
    private let statusIndicator = NSView()
    private let actionButton = NSButton()
    
    private var session: WorkspaceSession?
    private var sessionManager: SessionManager?
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupViews()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        nameLabel.font = NSFont.systemFont(ofSize: 13, weight: .medium)
        nameLabel.textColor = .labelColor
        nameLabel.lineBreakMode = .byTruncatingTail
        
        pathLabel.font = NSFont.systemFont(ofSize: 11)
        pathLabel.textColor = .secondaryLabelColor
        pathLabel.lineBreakMode = .byTruncatingMiddle
        
        statusIndicator.wantsLayer = true
        statusIndicator.layer?.cornerRadius = 4
        
        actionButton.bezelStyle = .rounded
        actionButton.font = NSFont.systemFont(ofSize: 10)
        actionButton.target = self
        actionButton.action = #selector(toggleSession)
        
        addSubview(statusIndicator)
        addSubview(nameLabel)
        addSubview(pathLabel)
        addSubview(actionButton)
        
        statusIndicator.translatesAutoresizingMaskIntoConstraints = false
        nameLabel.translatesAutoresizingMaskIntoConstraints = false
        pathLabel.translatesAutoresizingMaskIntoConstraints = false
        actionButton.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            statusIndicator.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            statusIndicator.centerYAnchor.constraint(equalTo: centerYAnchor),
            statusIndicator.widthAnchor.constraint(equalToConstant: 8),
            statusIndicator.heightAnchor.constraint(equalToConstant: 8),
            
            nameLabel.leadingAnchor.constraint(equalTo: statusIndicator.trailingAnchor, constant: 8),
            nameLabel.topAnchor.constraint(equalTo: topAnchor, constant: 6),
            nameLabel.trailingAnchor.constraint(equalTo: actionButton.leadingAnchor, constant: -8),
            
            pathLabel.leadingAnchor.constraint(equalTo: nameLabel.leadingAnchor),
            pathLabel.topAnchor.constraint(equalTo: nameLabel.bottomAnchor, constant: 2),
            pathLabel.trailingAnchor.constraint(equalTo: nameLabel.trailingAnchor),
            pathLabel.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -6),
            
            actionButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -8),
            actionButton.centerYAnchor.constraint(equalTo: centerYAnchor),
            actionButton.widthAnchor.constraint(equalToConstant: 50),
            actionButton.heightAnchor.constraint(equalToConstant: 20)
        ])
    }
    
    func configure(with session: WorkspaceSession, sessionManager: SessionManager) {
        self.session = session
        self.sessionManager = sessionManager
        
        nameLabel.stringValue = session.name
        pathLabel.stringValue = session.path
        statusIndicator.layer?.backgroundColor = session.status.color.cgColor
        
        let isActive = session.status == .active
        actionButton.title = isActive ? "Stop" : "Start"
    }
    
    @objc private func toggleSession() {
        guard let session = session, let sessionManager = sessionManager else { return }
        
        if session.status == .active {
            sessionManager.stopSession(session)
        } else {
            sessionManager.startSession(session)
        }
    }
}