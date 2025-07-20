import AppKit
import Combine
import PDFKit

// MARK: - Tab System Architecture

public protocol TabItem: AnyObject {
    var id: UUID { get }
    var title: String { get }
    var view: NSView { get }
    var isClosable: Bool { get }
}

public class CanvasTab: TabItem {
    public let id = UUID()
    public let title = "Canvas"
    public let view: NSView
    public let isClosable = false // Canvas tab can't be closed
    
    public init(canvasView: CanvasView) {
        self.view = canvasView
    }
}

public class PDFTab: TabItem {
    public let id = UUID()
    public let title: String
    public let view: NSView
    public let isClosable = true
    
    private let pdfNote: PDFNote
    
    public init(pdfNote: PDFNote) {
        self.pdfNote = pdfNote
        self.title = pdfNote.metadata.title ?? "PDF Document"
        
        // Create standalone PDF view (will be implemented)
        self.view = PDFStandaloneView(pdfNote: pdfNote)
    }
}

// MARK: - Tab Bar

public class TabBarView: NSView {
    private var tabs: [TabItem] = []
    private var activeTabId: UUID?
    private var tabButtons: [UUID: NSButton] = [:]
    
    private let tabHeight: CGFloat = 40
    private let tabMinWidth: CGFloat = 120
    private let tabMaxWidth: CGFloat = 200
    
    public weak var delegate: TabBarDelegate?
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupTabBar()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupTabBar()
    }
    
    private func setupTabBar() {
        wantsLayer = true
        layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
        
        // Add bottom border
        let border = CALayer()
        border.backgroundColor = NSColor.separatorColor.cgColor
        border.frame = CGRect(x: 0, y: 0, width: frame.width, height: 1)
        border.autoresizingMask = [.layerWidthSizable]
        layer?.addSublayer(border)
    }
    
    public func setTabs(_ tabs: [TabItem], activeTabId: UUID?) {
        self.tabs = tabs
        self.activeTabId = activeTabId
        updateTabButtons()
    }
    
    private func updateTabButtons() {
        // Remove existing buttons
        tabButtons.values.forEach { $0.removeFromSuperview() }
        tabButtons.removeAll()
        
        let tabWidth = min(tabMaxWidth, max(tabMinWidth, (frame.width - 20) / CGFloat(tabs.count)))
        
        for (index, tab) in tabs.enumerated() {
            let button = createTabButton(for: tab)
            button.frame = CGRect(
                x: 10 + CGFloat(index) * tabWidth,
                y: 5,
                width: tabWidth - 5,
                height: tabHeight - 10
            )
            
            tabButtons[tab.id] = button
            addSubview(button)
        }
        
        updateActiveTabAppearance()
    }
    
    private func createTabButton(for tab: TabItem) -> NSButton {
        let button = NSButton()
        button.title = tab.title
        button.bezelStyle = .rounded
        button.isBordered = true
        button.target = self
        button.action = #selector(tabButtonClicked(_:))
        button.wantsLayer = true
        
        // Store tab ID in button identifier
        button.identifier = NSUserInterfaceItemIdentifier(tab.id.uuidString)
        
        return button
    }
    
    @objc private func tabButtonClicked(_ sender: NSButton) {
        guard let tabIdString = sender.identifier?.rawValue,
              let tabId = UUID(uuidString: tabIdString) else { return }
        
        delegate?.tabBar(self, didSelectTab: tabId)
    }
    
    private func updateActiveTabAppearance() {
        for (tabId, button) in tabButtons {
            if tabId == activeTabId {
                button.layer?.backgroundColor = NSColor.controlAccentColor.withAlphaComponent(0.2).cgColor
            } else {
                button.layer?.backgroundColor = NSColor.clear.cgColor
            }
        }
    }
    
    override public func layout() {
        super.layout()
        updateTabButtons()
    }
}

public protocol TabBarDelegate: AnyObject {
    func tabBar(_ tabBar: TabBarView, didSelectTab tabId: UUID)
}

// MARK: - Tab Controller

public class TabViewController: NSViewController {
    private var tabs: [TabItem] = []
    private var activeTabId: UUID?
    
    private var tabBarView: TabBarView!
    private var contentContainer: NSView!
    private var currentContentView: NSView?
    
    public weak var delegate: TabViewControllerDelegate?
    
    public override func loadView() {
        view = NSView()
        setupTabInterface()
    }
    
    private func setupTabInterface() {
        // Create tab bar
        tabBarView = TabBarView()
        tabBarView.delegate = self
        tabBarView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(tabBarView)
        
        // Create content container
        contentContainer = NSView()
        contentContainer.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(contentContainer)
        
        // Setup constraints
        NSLayoutConstraint.activate([
            // Tab bar at top
            tabBarView.topAnchor.constraint(equalTo: view.topAnchor),
            tabBarView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            tabBarView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            tabBarView.heightAnchor.constraint(equalToConstant: 40),
            
            // Content container below tab bar
            contentContainer.topAnchor.constraint(equalTo: tabBarView.bottomAnchor),
            contentContainer.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            contentContainer.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            contentContainer.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }
    
    public func addTab(_ tab: TabItem, makeActive: Bool = true) {
        tabs.append(tab)
        
        if makeActive || tabs.count == 1 {
            activeTabId = tab.id
            showTabContent(tab)
        }
        
        updateTabBar()
    }
    
    public func removeTab(withId tabId: UUID) {
        guard let index = tabs.firstIndex(where: { $0.id == tabId }),
              tabs[index].isClosable else { return }
        
        _ = tabs.remove(at: index)
        
        // If we removed the active tab, switch to another tab
        if activeTabId == tabId {
            if !tabs.isEmpty {
                let newActiveTab = index < tabs.count ? tabs[index] : tabs[index - 1]
                activeTabId = newActiveTab.id
                showTabContent(newActiveTab)
            } else {
                activeTabId = nil
                clearContent()
            }
        }
        
        updateTabBar()
    }
    
    public func switchToTab(withId tabId: UUID) {
        guard let tab = tabs.first(where: { $0.id == tabId }) else { return }
        
        activeTabId = tabId
        showTabContent(tab)
        updateTabBar()
    }
    
    private func showTabContent(_ tab: TabItem) {
        clearContent()
        
        currentContentView = tab.view
        guard let contentView = currentContentView else {
            print("Error: Tab view is nil for tab: \(tab.title)")
            return
        }
        
        contentView.translatesAutoresizingMaskIntoConstraints = false
        contentContainer.addSubview(contentView)
        
        NSLayoutConstraint.activate([
            contentView.topAnchor.constraint(equalTo: contentContainer.topAnchor),
            contentView.leadingAnchor.constraint(equalTo: contentContainer.leadingAnchor),
            contentView.trailingAnchor.constraint(equalTo: contentContainer.trailingAnchor),
            contentView.bottomAnchor.constraint(equalTo: contentContainer.bottomAnchor)
        ])
    }
    
    private func clearContent() {
        currentContentView?.removeFromSuperview()
        currentContentView = nil
    }
    
    private func updateTabBar() {
        tabBarView.setTabs(tabs, activeTabId: activeTabId)
    }
}

extension TabViewController: TabBarDelegate {
    public func tabBar(_ tabBar: TabBarView, didSelectTab tabId: UUID) {
        switchToTab(withId: tabId)
        delegate?.tabViewController(self, didSwitchToTab: tabId)
    }
}

public protocol TabViewControllerDelegate: AnyObject {
    func tabViewController(_ controller: TabViewController, didSwitchToTab tabId: UUID)
}

