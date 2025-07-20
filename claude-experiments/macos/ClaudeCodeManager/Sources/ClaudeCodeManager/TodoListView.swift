import AppKit

class TodoListView: NSView {
    private var tableView: NSTableView!
    private var scrollView: NSScrollView!
    private var addButton: NSButton!
    private var removeButton: NSButton!
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
        wantsLayer = true
        layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
        
        let titleLabel = NSTextField(labelWithString: "Todo List")
        titleLabel.font = NSFont.systemFont(ofSize: 16, weight: .semibold)
        
        setupTableView()
        setupButtons()
        
        let headerView = NSView()
        headerView.addSubview(titleLabel)
        headerView.addSubview(addButton)
        headerView.addSubview(removeButton)
        
        addSubview(headerView)
        addSubview(scrollView)
        
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        headerView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        addButton.translatesAutoresizingMaskIntoConstraints = false
        removeButton.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            headerView.topAnchor.constraint(equalTo: topAnchor),
            headerView.leadingAnchor.constraint(equalTo: leadingAnchor),
            headerView.trailingAnchor.constraint(equalTo: trailingAnchor),
            headerView.heightAnchor.constraint(equalToConstant: 40),
            
            titleLabel.leadingAnchor.constraint(equalTo: headerView.leadingAnchor, constant: 16),
            titleLabel.centerYAnchor.constraint(equalTo: headerView.centerYAnchor),
            
            removeButton.trailingAnchor.constraint(equalTo: headerView.trailingAnchor, constant: -16),
            removeButton.centerYAnchor.constraint(equalTo: headerView.centerYAnchor),
            removeButton.widthAnchor.constraint(equalToConstant: 24),
            removeButton.heightAnchor.constraint(equalToConstant: 24),
            
            addButton.trailingAnchor.constraint(equalTo: removeButton.leadingAnchor, constant: -4),
            addButton.centerYAnchor.constraint(equalTo: headerView.centerYAnchor),
            addButton.widthAnchor.constraint(equalToConstant: 24),
            addButton.heightAnchor.constraint(equalToConstant: 24),
            
            scrollView.topAnchor.constraint(equalTo: headerView.bottomAnchor),
            scrollView.leadingAnchor.constraint(equalTo: leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
    }
    
    private func setupTableView() {
        tableView = NSTableView()
        tableView.headerView = nil
        tableView.rowSizeStyle = .medium
        tableView.allowsEmptySelection = true
        tableView.delegate = self
        tableView.dataSource = self
        
        let contentColumn = NSTableColumn(identifier: NSUserInterfaceItemIdentifier("ContentColumn"))
        contentColumn.title = "Task"
        contentColumn.width = 300
        tableView.addTableColumn(contentColumn)
        
        let statusColumn = NSTableColumn(identifier: NSUserInterfaceItemIdentifier("StatusColumn"))
        statusColumn.title = "Status"
        statusColumn.width = 100
        tableView.addTableColumn(statusColumn)
        
        let priorityColumn = NSTableColumn(identifier: NSUserInterfaceItemIdentifier("PriorityColumn"))
        priorityColumn.title = "Priority"
        priorityColumn.width = 80
        tableView.addTableColumn(priorityColumn)
        
        scrollView = NSScrollView()
        scrollView.documentView = tableView
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
    }
    
    private func setupButtons() {
        addButton = NSButton()
        addButton.title = "+"
        addButton.bezelStyle = .circular
        addButton.target = self
        addButton.action = #selector(addTodo)
        
        removeButton = NSButton()
        removeButton.title = "-"
        removeButton.bezelStyle = .circular
        removeButton.target = self
        removeButton.action = #selector(removeTodo)
        removeButton.isEnabled = false
    }
    
    func configure(with session: WorkspaceSession, sessionManager: SessionManager) {
        self.session = session
        self.sessionManager = sessionManager
        tableView.reloadData()
    }
    
    @objc private func addTodo() {
        let alert = NSAlert()
        alert.messageText = "New Todo Item"
        alert.informativeText = "Enter a description for the new todo item:"
        
        let textField = NSTextField(frame: NSRect(x: 0, y: 0, width: 300, height: 24))
        textField.placeholderString = "Todo description..."
        alert.accessoryView = textField
        
        alert.addButton(withTitle: "Add")
        alert.addButton(withTitle: "Cancel")
        
        alert.window.initialFirstResponder = textField
        
        let response = alert.runModal()
        if response == .alertFirstButtonReturn && !textField.stringValue.isEmpty {
            guard var session = session else { return }
            
            let newTodo = TodoItem(content: textField.stringValue)
            session.todos.append(newTodo)
            
            sessionManager?.updateSession(session)
            self.session = session
            tableView.reloadData()
        }
    }
    
    @objc private func removeTodo() {
        let selectedRow = tableView.selectedRow
        guard selectedRow >= 0, var session = session, selectedRow < session.todos.count else { return }
        
        session.todos.remove(at: selectedRow)
        sessionManager?.updateSession(session)
        self.session = session
        tableView.reloadData()
        removeButton.isEnabled = false
    }
}

extension TodoListView: NSTableViewDataSource {
    func numberOfRows(in tableView: NSTableView) -> Int {
        return session?.todos.count ?? 0
    }
}

extension TodoListView: NSTableViewDelegate {
    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView? {
        guard let session = session, row < session.todos.count else { return nil }
        
        let todo = session.todos[row]
        let identifier = tableColumn?.identifier
        
        if identifier == NSUserInterfaceItemIdentifier("ContentColumn") {
            let cellView = NSTableCellView()
            let textField = NSTextField(labelWithString: todo.content)
            textField.font = NSFont.systemFont(ofSize: 12)
            cellView.addSubview(textField)
            textField.translatesAutoresizingMaskIntoConstraints = false
            NSLayoutConstraint.activate([
                textField.leadingAnchor.constraint(equalTo: cellView.leadingAnchor, constant: 4),
                textField.trailingAnchor.constraint(equalTo: cellView.trailingAnchor, constant: -4),
                textField.centerYAnchor.constraint(equalTo: cellView.centerYAnchor)
            ])
            return cellView
            
        } else if identifier == NSUserInterfaceItemIdentifier("StatusColumn") {
            let cellView = NSTableCellView()
            let button = NSPopUpButton()
            
            for status in TodoStatus.allCases {
                button.addItem(withTitle: status.rawValue)
            }
            
            button.selectItem(withTitle: todo.status.rawValue)
            button.target = self
            button.action = #selector(statusChanged(_:))
            button.tag = row
            
            cellView.addSubview(button)
            button.translatesAutoresizingMaskIntoConstraints = false
            NSLayoutConstraint.activate([
                button.leadingAnchor.constraint(equalTo: cellView.leadingAnchor),
                button.trailingAnchor.constraint(equalTo: cellView.trailingAnchor),
                button.centerYAnchor.constraint(equalTo: cellView.centerYAnchor)
            ])
            return cellView
            
        } else if identifier == NSUserInterfaceItemIdentifier("PriorityColumn") {
            let cellView = NSTableCellView()
            let button = NSPopUpButton()
            
            for priority in TodoPriority.allCases {
                button.addItem(withTitle: priority.rawValue)
            }
            
            button.selectItem(withTitle: todo.priority.rawValue)
            button.target = self
            button.action = #selector(priorityChanged(_:))
            button.tag = row
            
            cellView.addSubview(button)
            button.translatesAutoresizingMaskIntoConstraints = false
            NSLayoutConstraint.activate([
                button.leadingAnchor.constraint(equalTo: cellView.leadingAnchor),
                button.trailingAnchor.constraint(equalTo: cellView.trailingAnchor),
                button.centerYAnchor.constraint(equalTo: cellView.centerYAnchor)
            ])
            return cellView
        }
        
        return nil
    }
    
    func tableViewSelectionDidChange(_ notification: Notification) {
        removeButton.isEnabled = tableView.selectedRow >= 0
    }
    
    @objc private func statusChanged(_ sender: NSPopUpButton) {
        let row = sender.tag
        guard var session = session, row < session.todos.count else { return }
        
        let statusTitle = sender.titleOfSelectedItem ?? ""
        if let newStatus = TodoStatus.allCases.first(where: { $0.rawValue == statusTitle }) {
            session.todos[row].status = newStatus
            sessionManager?.updateSession(session)
            self.session = session
        }
    }
    
    @objc private func priorityChanged(_ sender: NSPopUpButton) {
        let row = sender.tag
        guard var session = session, row < session.todos.count else { return }
        
        let priorityTitle = sender.titleOfSelectedItem ?? ""
        if let newPriority = TodoPriority.allCases.first(where: { $0.rawValue == priorityTitle }) {
            session.todos[row].priority = newPriority
            sessionManager?.updateSession(session)
            self.session = session
        }
    }
}