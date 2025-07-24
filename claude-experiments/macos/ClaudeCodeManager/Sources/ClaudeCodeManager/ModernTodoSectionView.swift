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
        // Keep it simple - just a clean background
        wantsLayer = true
        layer?.backgroundColor = DesignSystem.Colors.surfacePrimary.cgColor
        layer?.cornerRadius = DesignSystem.CornerRadius.md
        
        headerView.onAddPressed = { [weak self] in
            self?.saveTodoFile()
        }
        
        // Setup scroll view first
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = false
        scrollView.borderType = .noBorder
        scrollView.drawsBackground = false
        
        // Setup text view for markdown editing
        textView.font = DesignSystem.Typography.codeBody
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.backgroundColor = NSColor.textBackgroundColor
        textView.textColor = NSColor.textColor
        textView.isEditable = true
        textView.isSelectable = true
        textView.insertionPointColor = DesignSystem.Colors.accent
        textView.textContainerInset = NSSize(width: DesignSystem.Spacing.lg, height: DesignSystem.Spacing.lg)
        
        
        // Configure text container to expand properly
        if let textContainer = textView.textContainer {
            textContainer.widthTracksTextView = true
            textContainer.heightTracksTextView = false
            textContainer.containerSize = NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude)
        }
        
        // Set proper sizing constraints
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
        textView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            headerView.topAnchor.constraint(equalTo: topAnchor, constant: DesignSystem.Spacing.xxl),
            headerView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: DesignSystem.Spacing.xxl),
            headerView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -DesignSystem.Spacing.xxl),
            headerView.heightAnchor.constraint(equalToConstant: 44),
            
            scrollView.topAnchor.constraint(equalTo: headerView.bottomAnchor, constant: DesignSystem.Spacing.lg),
            scrollView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: DesignSystem.Spacing.xxl),
            scrollView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -DesignSystem.Spacing.xxl),
            scrollView.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -DesignSystem.Spacing.xxl),
            
            // Essential width constraint but allow height to be flexible
            textView.widthAnchor.constraint(equalTo: scrollView.widthAnchor, constant: -20)
        ])
    }
    
    func configure(with session: WorkspaceSession, sessionManager: SessionManager) {
        self.session = session
        self.sessionManager = sessionManager
        loadTodoFile()
    }
    
    private func loadTodoFile() {
        guard let session = session else { 
            return 
        }
        
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
            self.textView.string = content
            
            // Force the text view to recalculate its size
            if let textContainer = self.textView.textContainer,
               let layoutManager = self.textView.layoutManager {
                layoutManager.ensureLayout(for: textContainer)
                let usedRect = layoutManager.usedRect(for: textContainer)
                self.textView.frame = NSRect(x: 0, y: 0, width: self.textView.frame.width, height: max(usedRect.height + 20, self.scrollView.frame.height))
            }
            
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
        titleLabel.font = DesignSystem.Typography.headlineEmphasized
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