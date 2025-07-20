import AppKit
import Combine

protocol SidebarViewControllerDelegate: AnyObject {
    func didSelectSession(_ session: WorkspaceSession?)
}

class SidebarViewController: NSViewController {
    weak var delegate: SidebarViewControllerDelegate?
    
    let sessionManager: SessionManager
    private var tableView: NSTableView!
    private var scrollView: NSScrollView!
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
        
        setupTableView()
        setupButtons()
        setupConstraints()
        setupBindings()
    }
    
    private func setupTableView() {
        tableView = NSTableView()
        tableView.headerView = nil
        tableView.rowSizeStyle = .medium
        tableView.style = .sourceList
        tableView.allowsEmptySelection = true
        tableView.delegate = self
        tableView.dataSource = self
        
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
    
    private func setupButtons() {
        addButton = NSButton()
        addButton.title = "+"
        addButton.bezelStyle = .circular
        addButton.target = self
        addButton.action = #selector(addSession)
        
        removeButton = NSButton()
        removeButton.title = "-"
        removeButton.bezelStyle = .circular
        removeButton.target = self
        removeButton.action = #selector(removeSession)
        removeButton.isEnabled = false
        
        view.addSubview(addButton)
        view.addSubview(removeButton)
    }
    
    private func setupConstraints() {
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        addButton.translatesAutoresizingMaskIntoConstraints = false
        removeButton.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            scrollView.topAnchor.constraint(equalTo: view.topAnchor),
            scrollView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: addButton.topAnchor, constant: -8),
            
            addButton.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 8),
            addButton.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -8),
            addButton.widthAnchor.constraint(equalToConstant: 24),
            addButton.heightAnchor.constraint(equalToConstant: 24),
            
            removeButton.leadingAnchor.constraint(equalTo: addButton.trailingAnchor, constant: 4),
            removeButton.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -8),
            removeButton.widthAnchor.constraint(equalToConstant: 24),
            removeButton.heightAnchor.constraint(equalToConstant: 24)
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
    
    @objc private func addSession() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Select Workspace"
        
        panel.begin { [weak self] response in
            guard response == .OK, let url = panel.url else { return }
            
            let name = url.lastPathComponent
            let path = url.path
            
            self?.sessionManager.addSession(name: name, path: path)
        }
    }
    
    @objc private func removeSession() {
        let selectedRow = tableView.selectedRow
        guard selectedRow >= 0, selectedRow < sessionManager.sessions.count else { return }
        
        let session = sessionManager.sessions[selectedRow]
        sessionManager.removeSession(session)
        
        removeButton.isEnabled = false
    }
}

extension SidebarViewController: NSTableViewDataSource {
    func numberOfRows(in tableView: NSTableView) -> Int {
        return sessionManager.sessions.count
    }
}

extension SidebarViewController: NSTableViewDelegate {
    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView? {
        let session = sessionManager.sessions[row]
        
        let cellView = SessionCellView()
        cellView.configure(with: session)
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

class SessionCellView: NSTableCellView {
    private let nameLabel = NSTextField(labelWithString: "")
    private let pathLabel = NSTextField(labelWithString: "")
    private let statusIndicator = NSView()
    private let startStopButton = NSButton()
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupViews()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        nameLabel.font = NSFont.systemFont(ofSize: 13, weight: .medium)
        pathLabel.font = NSFont.systemFont(ofSize: 11)
        pathLabel.textColor = .secondaryLabelColor
        
        statusIndicator.wantsLayer = true
        statusIndicator.layer?.cornerRadius = 4
        
        startStopButton.bezelStyle = .rounded
        startStopButton.font = NSFont.systemFont(ofSize: 10)
        
        addSubview(nameLabel)
        addSubview(pathLabel)
        addSubview(statusIndicator)
        addSubview(startStopButton)
        
        nameLabel.translatesAutoresizingMaskIntoConstraints = false
        pathLabel.translatesAutoresizingMaskIntoConstraints = false
        statusIndicator.translatesAutoresizingMaskIntoConstraints = false
        startStopButton.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            statusIndicator.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            statusIndicator.centerYAnchor.constraint(equalTo: centerYAnchor),
            statusIndicator.widthAnchor.constraint(equalToConstant: 8),
            statusIndicator.heightAnchor.constraint(equalToConstant: 8),
            
            nameLabel.leadingAnchor.constraint(equalTo: statusIndicator.trailingAnchor, constant: 8),
            nameLabel.topAnchor.constraint(equalTo: topAnchor, constant: 4),
            nameLabel.trailingAnchor.constraint(equalTo: startStopButton.leadingAnchor, constant: -8),
            
            pathLabel.leadingAnchor.constraint(equalTo: nameLabel.leadingAnchor),
            pathLabel.topAnchor.constraint(equalTo: nameLabel.bottomAnchor, constant: 2),
            pathLabel.trailingAnchor.constraint(equalTo: nameLabel.trailingAnchor),
            pathLabel.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -4),
            
            startStopButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -8),
            startStopButton.centerYAnchor.constraint(equalTo: centerYAnchor),
            startStopButton.widthAnchor.constraint(equalToConstant: 50),
            startStopButton.heightAnchor.constraint(equalToConstant: 20)
        ])
    }
    
    func configure(with session: WorkspaceSession) {
        nameLabel.stringValue = session.name
        pathLabel.stringValue = session.path
        statusIndicator.layer?.backgroundColor = session.status.color.cgColor
        
        let isActive = session.status == .active
        startStopButton.title = isActive ? "Stop" : "Start"
        startStopButton.target = self
        startStopButton.action = isActive ? #selector(stopSession) : #selector(startSession)
        
        self.session = session
    }
    
    private var session: WorkspaceSession?
    
    @objc private func startSession() {
        guard let session = session else { return }
        // Get session manager from the view hierarchy
        if let sidebarVC = findParentViewController() as? SidebarViewController {
            sidebarVC.sessionManager.startSession(session)
        }
    }
    
    @objc private func stopSession() {
        guard let session = session else { return }
        if let sidebarVC = findParentViewController() as? SidebarViewController {
            sidebarVC.sessionManager.stopSession(session)
        }
    }
    
    private func findParentViewController() -> NSViewController? {
        var responder: NSResponder? = self
        while responder != nil {
            if let viewController = responder as? NSViewController {
                return viewController
            }
            responder = responder?.nextResponder
        }
        return nil
    }
}