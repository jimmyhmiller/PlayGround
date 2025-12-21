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
    private var clearMenuItem: NSMenuItem?
    private var modifierTimer: Timer?

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
        let menu = NSMenu()
        menu.delegate = self
        menu.autoenablesItems = false

        let modifiersHeld = NSEvent.modifierFlags.contains([.command, .option])

        let clearItem = NSMenuItem(
            title: modifiersHeld ? "Clear Data" : "Clear Data (⌘⌥)",
            action: #selector(clearDataClicked),
            keyEquivalent: ""
        )
        clearItem.isEnabled = modifiersHeld
        clearMenuItem = clearItem
        menu.addItem(clearItem)
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Quit", action: #selector(quitApp), keyEquivalent: "q"))

        statusItem.menu = menu
        statusItem.button?.performClick(nil)
        statusItem.menu = nil
    }

    @objc private func clearDataClicked() {
        viewModel.clearAllData()
    }

    func menuWillOpen(_ menu: NSMenu) {
        // Start polling for modifier changes - must use common modes to work during menu tracking
        modifierTimer = Timer(timeInterval: 0.05, repeats: true) { [weak self] _ in
            DispatchQueue.main.async {
                let modifiersHeld = NSEvent.modifierFlags.contains([.command, .option])
                self?.clearMenuItem?.title = modifiersHeld ? "Clear Data" : "Clear Data (⌘⌥)"
                self?.clearMenuItem?.isEnabled = modifiersHeld
            }
        }
        RunLoop.main.add(modifierTimer!, forMode: .common)
    }

    func menuDidClose(_ menu: NSMenu) {
        modifierTimer?.invalidate()
        modifierTimer = nil
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }
}
