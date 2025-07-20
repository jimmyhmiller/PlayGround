import AppKit
import Foundation

class MainWindowController: NSWindowController {
    private var sessionManager: SessionManager!
    private var sidebarViewController: SimpleSidebarViewController!
    private var mainContentViewController: ModernContentViewController!
    
    override init(window: NSWindow?) {
        super.init(window: window)
        setupWindow()
        setupSessionManager()
        setupViewControllers()
    }
    
    convenience init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1200, height: 800),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Claude Code Manager"
        window.center()
        window.minSize = NSSize(width: 900, height: 600)
        
        self.init(window: window)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setupWindow() {
        window?.titleVisibility = .visible
        window?.titlebarAppearsTransparent = false
        window?.backgroundColor = NSColor.windowBackgroundColor
    }
    
    private func setupSessionManager() {
        sessionManager = SessionManager()
    }
    
    private func setupViewControllers() {
        let splitViewController = NSSplitViewController()
        splitViewController.splitView.isVertical = true
        splitViewController.splitView.dividerStyle = .thin
        
        // Sidebar
        sidebarViewController = SimpleSidebarViewController(sessionManager: sessionManager)
        let sidebarItem = NSSplitViewItem(sidebarWithViewController: sidebarViewController)
        sidebarItem.minimumThickness = 280
        sidebarItem.maximumThickness = 350
        splitViewController.addSplitViewItem(sidebarItem)
        
        // Main content
        mainContentViewController = ModernContentViewController(sessionManager: sessionManager)
        let contentItem = NSSplitViewItem(viewController: mainContentViewController)
        splitViewController.addSplitViewItem(contentItem)
        
        window?.contentViewController = splitViewController
        
        // Setup toolbar
        setupToolbar()
        
        // Setup delegates
        sidebarViewController.delegate = mainContentViewController
    }
    
    private func setupToolbar() {
        let toolbar = NSToolbar(identifier: "MainToolbar")
        toolbar.delegate = self
        toolbar.allowsUserCustomization = false
        toolbar.autosavesConfiguration = false
        toolbar.displayMode = .iconOnly
        
        window?.toolbar = toolbar
    }
    
    @objc private func toggleSidebar() {
        guard let splitViewController = window?.contentViewController as? NSSplitViewController else { return }
        splitViewController.splitViewItems.first?.animator().isCollapsed.toggle()
    }
    
    @objc private func addWorkspace() {
        sidebarViewController.addWorkspace()
    }
}

// MARK: - Toolbar Delegate
extension MainWindowController: NSToolbarDelegate {
    func toolbar(_ toolbar: NSToolbar, itemForItemIdentifier itemIdentifier: NSToolbarItem.Identifier, willBeInsertedIntoToolbar flag: Bool) -> NSToolbarItem? {
        
        switch itemIdentifier {
        case .toggleSidebar:
            let item = NSToolbarItem(itemIdentifier: itemIdentifier)
            item.label = "Toggle Sidebar"
            item.image = NSImage(systemSymbolName: "sidebar.left", accessibilityDescription: "Toggle Sidebar")
            item.target = self
            item.action = #selector(toggleSidebar)
            return item
            
        case .addWorkspace:
            let item = NSToolbarItem(itemIdentifier: itemIdentifier)
            item.label = "Add Workspace"
            item.image = NSImage(systemSymbolName: "plus", accessibilityDescription: "Add Workspace")
            item.target = self
            item.action = #selector(addWorkspace)
            return item
            
        default:
            return nil
        }
    }
    
    func toolbarDefaultItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        return [.toggleSidebar, .flexibleSpace, .addWorkspace]
    }
    
    func toolbarAllowedItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        return [.toggleSidebar, .addWorkspace, .flexibleSpace, .space]
    }
}

// MARK: - Toolbar Item Identifiers
extension NSToolbarItem.Identifier {
    static let toggleSidebar = NSToolbarItem.Identifier("ToggleSidebar")
    static let addWorkspace = NSToolbarItem.Identifier("AddWorkspace")
}