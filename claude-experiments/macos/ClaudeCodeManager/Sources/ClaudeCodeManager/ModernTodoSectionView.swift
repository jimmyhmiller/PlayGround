import AppKit

class ModernTodoSectionView: NSView {
    private var session: WorkspaceSession?
    private var sessionManager: SessionManager?
    
    private let headerView = ModernTodoHeaderView()
    private let scrollView = NSScrollView()
    private let textView = NSTextView()
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupViews()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupViews() {
        addGlassmorphismEffect()
        
        headerView.onAddPressed = { [weak self] in
            self?.saveTodoFile()
        }
        
        // Setup scroll view first
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = false
        scrollView.borderType = .noBorder
        
        // Setup text view for markdown editing
        textView.font = NSFont.monospacedSystemFont(ofSize: 14, weight: .regular)
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.backgroundColor = NSColor.textBackgroundColor
        textView.textColor = NSColor.textColor
        textView.isEditable = true
        textView.isSelectable = true
        textView.insertionPointColor = NSColor.textColor
        textView.textContainerInset = NSSize(width: 10, height: 10)
        
        // Configure text container for proper scrolling
        if let textContainer = textView.textContainer {
            textContainer.widthTracksTextView = true
            textContainer.heightTracksTextView = false
            textContainer.containerSize = NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude)
        }
        
        // Configure text view sizing
        textView.minSize = NSSize(width: 0, height: 0)
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        
        // Important: Set the document view
        scrollView.documentView = textView
        
        addSubview(headerView)
        addSubview(scrollView)
        
        headerView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            headerView.topAnchor.constraint(equalTo: topAnchor, constant: DesignSystem.Spacing.xxl),
            headerView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: DesignSystem.Spacing.xxl),
            headerView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -DesignSystem.Spacing.xxl),
            headerView.heightAnchor.constraint(equalToConstant: 44),
            
            scrollView.topAnchor.constraint(equalTo: headerView.bottomAnchor, constant: DesignSystem.Spacing.lg),
            scrollView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: DesignSystem.Spacing.xxl),
            scrollView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -DesignSystem.Spacing.xxl),
            scrollView.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -DesignSystem.Spacing.xxl)
        ])
    }
    
    func configure(with session: WorkspaceSession, sessionManager: SessionManager) {
        self.session = session
        self.sessionManager = sessionManager
        print("DEBUG: Configure called, loading todo file...")
        loadTodoFile()
    }
    
    private func loadTodoFile() {
        guard let session = session else { 
            print("DEBUG: No session available")
            return 
        }
        
        print("DEBUG: Loading todo file for session path: \(session.path)")
        
        let basePath = URL(fileURLWithPath: session.path)
        let possiblePaths = [
            basePath.appendingPathComponent("TODO.md"),
            basePath.appendingPathComponent("todo.md"),
            basePath.appendingPathComponent("Todo.md")
        ]
        
        var content = ""
        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path.path) {
                if let fileContent = try? String(contentsOf: path) {
                    content = fileContent
                    break
                }
            }
        }
        
        if content.isEmpty {
            content = "# TODO\n\n"
        }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            print("DEBUG: Loading content with \(content.count) characters: \(content.prefix(50))...")
            self.textView.string = content
            
            // Force layout update
            self.textView.sizeToFit()
            self.textView.needsLayout = true
            self.textView.layoutSubtreeIfNeeded()
            self.scrollView.needsLayout = true
            self.scrollView.layoutSubtreeIfNeeded()
            
            // Force scroll to top to see the content
            self.textView.scrollToBeginningOfDocument(nil)
        }
    }
    
    private func saveTodoFile() {
        guard let session = session else { return }
        
        let basePath = URL(fileURLWithPath: session.path)
        let possiblePaths = [
            basePath.appendingPathComponent("TODO.md"),
            basePath.appendingPathComponent("todo.md"),
            basePath.appendingPathComponent("Todo.md")
        ]
        
        // Find existing file or default to TODO.md
        var todoPath = basePath.appendingPathComponent("TODO.md")
        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path.path) {
                todoPath = path
                break
            }
        }
        
        let content = textView.string
        try? content.write(to: todoPath, atomically: true, encoding: .utf8)
    }
    
}

// MARK: - Todo Header View
class ModernTodoHeaderView: NSView {
    var onAddPressed: (() -> Void)?
    
    private let titleLabel = NSTextField(labelWithString: "Tasks")
    private let saveButton = ModernButton(title: "Save", style: .accent)
    
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
        
        saveButton.onPressed = { [weak self] in
            self?.onAddPressed?()
        }
        
        addSubview(titleLabel)
        addSubview(saveButton)
        
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        saveButton.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            titleLabel.leadingAnchor.constraint(equalTo: leadingAnchor),
            titleLabel.centerYAnchor.constraint(equalTo: centerYAnchor),
            
            saveButton.trailingAnchor.constraint(equalTo: trailingAnchor),
            saveButton.centerYAnchor.constraint(equalTo: centerYAnchor),
            saveButton.heightAnchor.constraint(equalToConstant: 32)
        ])
    }
}