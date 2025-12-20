import SwiftUI
import AppKit

@main
struct TrackerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        Settings {
            EmptyView()
        }
    }
}

@MainActor
class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    private var statusItem: NSStatusItem!
    private var popover: NSPopover!
    private let viewModel = TrackerViewModel()
    private var clearClickCount = 0
    private var clearDataMenuItem: NSMenuItem?

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupStatusItem()
        setupPopover()
    }

    private func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "chart.bar.fill", accessibilityDescription: "Tracker")
            button.action = #selector(handleClick)
            button.sendAction(on: [.leftMouseUp, .rightMouseUp])
            button.target = self
        }
    }

    private func setupPopover() {
        popover = NSPopover()
        popover.contentSize = NSSize(width: 280, height: 400)
        popover.behavior = .transient
        popover.contentViewController = NSHostingController(
            rootView: MainPopoverView().environmentObject(viewModel)
        )
    }

    @objc private func handleClick(_ sender: NSStatusBarButton) {
        guard let event = NSApp.currentEvent else { return }

        if event.type == .rightMouseUp || event.modifierFlags.contains(.control) {
            showContextMenu(sender)
        } else {
            togglePopover(sender)
        }
    }

    private func togglePopover(_ sender: NSStatusBarButton) {
        if popover.isShown {
            popover.performClose(sender)
        } else {
            popover.show(relativeTo: sender.bounds, of: sender, preferredEdge: .minY)
            popover.contentViewController?.view.window?.makeKey()
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    private func showContextMenu(_ sender: NSStatusBarButton) {
        clearClickCount = 0
        let menu = NSMenu()
        menu.delegate = self

        let clearItem = NSMenuItem()
        let clearView = ClearDataView(
            remainingClicks: 5,
            onClick: { [weak self] in
                self?.handleClearDataClick()
            }
        )
        clearItem.view = clearView
        clearDataMenuItem = clearItem
        menu.addItem(clearItem)
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Quit", action: #selector(quitApp), keyEquivalent: "q"))

        statusItem.menu = menu
        statusItem.button?.performClick(nil)
        statusItem.menu = nil
    }

    private func handleClearDataClick() {
        clearClickCount += 1
        let remainingClicks = 5 - clearClickCount

        if remainingClicks <= 0 {
            viewModel.clearAllData()
            clearClickCount = 0
            statusItem.menu?.cancelTracking()
        } else if let menuItem = clearDataMenuItem,
                  let clearView = menuItem.view as? ClearDataView {
            clearView.updateCount(remainingClicks)
        }
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }

    nonisolated func menuDidClose(_ menu: NSMenu) {
        Task { @MainActor in
            self.clearClickCount = 0
        }
    }
}

class ClearDataView: NSView {
    private let label: NSTextField
    private var onClick: () -> Void

    init(remainingClicks: Int, onClick: @escaping () -> Void) {
        self.onClick = onClick
        self.label = NSTextField(labelWithString: "Clear Data (\(remainingClicks))")
        super.init(frame: NSRect(x: 0, y: 0, width: 150, height: 22))

        label.font = NSFont.menuFont(ofSize: 0)
        label.translatesAutoresizingMaskIntoConstraints = false
        addSubview(label)

        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 14),
            label.centerYAnchor.constraint(equalTo: centerYAnchor)
        ])
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func updateCount(_ remaining: Int) {
        label.stringValue = "Clear Data (\(remaining))"
    }

    override func mouseUp(with event: NSEvent) {
        onClick()
    }
}
