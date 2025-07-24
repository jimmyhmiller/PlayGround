import AppKit
import Combine

protocol SimpleSidebarDelegate: AnyObject {
    func didSelectSession(_ session: WorkspaceSession?)
}

class SimpleSidebarViewController: NSViewController {
    weak var delegate: SimpleSidebarDelegate?
    let sessionManager: SessionManager
    private var cancellables = Set<AnyCancellable>()
    
    private var tableView: NSTableView!
    private var scrollView: NSScrollView!
    private var headerView: NSView!
    
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
        
        setupHeader()
        setupTableView()
        setupConstraints()
        setupBindings()
    }
    
    private func setupHeader() {
        let headerView = NSView()
        headerView.wantsLayer = true
        
        let titleLabel = NSTextField(labelWithString: "WORKSPACES")
        titleLabel.font = DesignSystem.Typography.captionEmphasized
        titleLabel.textColor = NSColor.tertiaryLabelColor
        titleLabel.isEditable = false
        titleLabel.isSelectable = false
        titleLabel.backgroundColor = .clear
        titleLabel.isBordered = false
        
        let addButton = NSButton()
        addButton.image = NSImage(systemSymbolName: "plus", accessibilityDescription: "Add Workspace")
        addButton.bezelStyle = .regularSquare
        addButton.isBordered = false
        addButton.imageScaling = .scaleProportionallyDown
        addButton.target = self
        addButton.action = #selector(addWorkspacePressed)
        
        // Style the add button simply
        addButton.contentTintColor = NSColor.controlAccentColor
        
        headerView.addSubview(titleLabel)
        headerView.addSubview(addButton)
        view.addSubview(headerView)
        
        headerView.translatesAutoresizingMaskIntoConstraints = false
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        addButton.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            headerView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            headerView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            headerView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            headerView.heightAnchor.constraint(equalToConstant: 24),
            
            titleLabel.leadingAnchor.constraint(equalTo: headerView.leadingAnchor, constant: 12),
            titleLabel.centerYAnchor.constraint(equalTo: headerView.centerYAnchor),
            
            addButton.trailingAnchor.constraint(equalTo: headerView.trailingAnchor, constant: -12),
            addButton.centerYAnchor.constraint(equalTo: headerView.centerYAnchor),
            addButton.widthAnchor.constraint(equalToConstant: 16),
            addButton.heightAnchor.constraint(equalToConstant: 16)
        ])
        
        self.headerView = headerView
    }
    
    @objc private func addWorkspacePressed() {
        addWorkspace()
    }
    
    private func setupTableView() {
        tableView = NSTableView()
        tableView.delegate = self
        tableView.dataSource = self
        tableView.headerView = nil
        tableView.rowSizeStyle = .small
        tableView.intercellSpacing = NSSize(width: 0, height: 1)
        tableView.backgroundColor = .clear
        
        if #available(macOS 11.0, *) {
            tableView.style = .sourceList
        } else {
            tableView.selectionHighlightStyle = .sourceList
        }
        
        let column = NSTableColumn(identifier: NSUserInterfaceItemIdentifier("WorkspaceColumn"))
        column.title = "Workspaces"
        tableView.addTableColumn(column)
        
        scrollView = NSScrollView()
        scrollView.documentView = tableView
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = true
        scrollView.scrollerStyle = .overlay
        scrollView.borderType = .noBorder
        scrollView.backgroundColor = .clear
        
        view.addSubview(scrollView)
    }
    
    private func setupConstraints() {
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            scrollView.topAnchor.constraint(equalTo: headerView.bottomAnchor, constant: 8),
            scrollView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
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

extension SimpleSidebarViewController: NSTableViewDataSource {
    func numberOfRows(in tableView: NSTableView) -> Int {
        return sessionManager.sessions.count
    }
}

extension SimpleSidebarViewController: NSTableViewDelegate {
    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView? {
        let session = sessionManager.sessions[row]
        
        let cellView = NSTableCellView()
        
        // Folder icon
        let iconView = NSImageView()
        iconView.image = NSImage(systemSymbolName: "folder.fill", accessibilityDescription: "Workspace")
        iconView.contentTintColor = .systemBlue
        iconView.imageScaling = .scaleProportionallyUpOrDown
        
        // Status indicator
        let statusDot = NSView()
        statusDot.wantsLayer = true
        statusDot.layer?.cornerRadius = 3
        statusDot.layer?.backgroundColor = session.status.color.cgColor
        
        // Workspace name
        let textField = NSTextField(labelWithString: session.name)
        textField.font = DesignSystem.Typography.bodyEmphasized
        textField.textColor = NSColor.labelColor
        textField.isEditable = false
        textField.isSelectable = false
        textField.backgroundColor = .clear
        textField.isBordered = false
        
        cellView.addSubview(iconView)
        cellView.addSubview(statusDot)
        cellView.addSubview(textField)
        
        iconView.translatesAutoresizingMaskIntoConstraints = false
        statusDot.translatesAutoresizingMaskIntoConstraints = false
        textField.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            iconView.leadingAnchor.constraint(equalTo: cellView.leadingAnchor, constant: 12),
            iconView.centerYAnchor.constraint(equalTo: cellView.centerYAnchor),
            iconView.widthAnchor.constraint(equalToConstant: 16),
            iconView.heightAnchor.constraint(equalToConstant: 16),
            
            statusDot.leadingAnchor.constraint(equalTo: iconView.trailingAnchor, constant: -2),
            statusDot.topAnchor.constraint(equalTo: iconView.topAnchor, constant: -2),
            statusDot.widthAnchor.constraint(equalToConstant: 6),
            statusDot.heightAnchor.constraint(equalToConstant: 6),
            
            textField.leadingAnchor.constraint(equalTo: iconView.trailingAnchor, constant: 8),
            textField.centerYAnchor.constraint(equalTo: cellView.centerYAnchor),
            textField.trailingAnchor.constraint(equalTo: cellView.trailingAnchor, constant: -8)
        ])
        
        return cellView
    }
    
    func tableView(_ tableView: NSTableView, shouldSelectRow row: Int) -> Bool {
        let session = sessionManager.sessions[row]
        sessionManager.selectSession(session)
        delegate?.didSelectSession(session)
        return true
    }
}

// Conform to the same protocol as the other sidebar
extension SimpleSidebarViewController: SidebarViewControllerDelegate {
    func didSelectSession(_ session: WorkspaceSession?) {
        delegate?.didSelectSession(session)
    }
}