import SwiftUI
import AppKit
import Sparkle

@main
struct EaseApp: App {
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
    private let viewModel = EaseViewModel()
    private let updateManager = UpdateManager.shared
    private var clearMenuItem: NSMenuItem?
    private var modifierTimer: Timer?
    private var appearanceObserver: NSKeyValueObservation?

    private static let showInDockKey = "showInDock"

    private var showInDock: Bool {
        get { UserDefaults.standard.bool(forKey: Self.showInDockKey) }
        set {
            UserDefaults.standard.set(newValue, forKey: Self.showInDockKey)
            updateDockVisibility()
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupStatusItem()
        setupPopover()
        setupAppearanceObserver()
        updateAppIcon()
        updateDockVisibility()
    }

    private func updateDockVisibility() {
        NSApp.setActivationPolicy(showInDock ? .regular : .accessory)
    }

    private func setupAppearanceObserver() {
        appearanceObserver = NSApp.observe(\.effectiveAppearance, options: [.new]) { [weak self] _, _ in
            Task { @MainActor in
                self?.updateAppIcon()
            }
        }
    }

    private func updateAppIcon() {
        let isDarkMode = NSApp.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        let iconName = isDarkMode ? "AppIcon-Dark" : "AppIcon-Light"

        if let iconURL = Bundle.main.url(forResource: iconName, withExtension: "icns"),
           let icon = NSImage(contentsOf: iconURL) {
            NSApp.applicationIconImage = icon
        }
    }

    private func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

        if let button = statusItem.button {
            button.image = createMenuBarIcon()
            button.image?.isTemplate = true
            button.action = #selector(handleClick)
            button.sendAction(on: [.leftMouseUp, .rightMouseUp])
            button.target = self
        }
    }

    private func createMenuBarIcon() -> NSImage? {
        // Short, Long, Medium bar pattern
        return svgToImage("""
            <svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">
              <rect x="2" y="2" width="7" height="4" rx="2" fill="black"/>
              <rect x="2" y="7" width="14" height="4" rx="2" fill="black"/>
              <rect x="2" y="12.5" width="11" height="4" rx="2" fill="black"/>
            </svg>
            """)
    }

    private func svgToImage(_ svgString: String) -> NSImage? {
        guard let data = svgString.data(using: .utf8) else { return nil }
        guard let svgImage = NSImage(data: data) else { return nil }

        // Create a template image at the right size for menu bar
        let size = NSSize(width: 18, height: 18)
        let image = NSImage(size: size)
        image.lockFocus()
        svgImage.draw(in: NSRect(origin: .zero, size: size))
        image.unlockFocus()
        image.isTemplate = true
        return image
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

        // Check for Updates
        let updateItem = NSMenuItem(
            title: "Check for Updates...",
            action: #selector(checkForUpdates),
            keyEquivalent: ""
        )
        updateItem.isEnabled = updateManager.canCheckForUpdates
        menu.addItem(updateItem)

        // Update Channel submenu
        let channelMenu = NSMenu()
        for channel in UpdateChannel.allCases {
            let channelItem = NSMenuItem(
                title: channel.displayName,
                action: #selector(switchChannel(_:)),
                keyEquivalent: ""
            )
            channelItem.representedObject = channel
            channelItem.state = updateManager.currentChannel == channel ? .on : .off
            channelMenu.addItem(channelItem)
        }
        let channelMenuItem = NSMenuItem(title: "Update Channel", action: nil, keyEquivalent: "")
        channelMenuItem.submenu = channelMenu
        menu.addItem(channelMenuItem)

        menu.addItem(NSMenuItem.separator())

        // Show in Dock
        let dockItem = NSMenuItem(
            title: "Show in Dock",
            action: #selector(toggleShowInDock),
            keyEquivalent: ""
        )
        dockItem.state = showInDock ? .on : .off
        menu.addItem(dockItem)

        menu.addItem(NSMenuItem.separator())

        // Clear Data
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

    @objc private func checkForUpdates() {
        updateManager.checkForUpdates()
    }

    @objc private func switchChannel(_ sender: NSMenuItem) {
        guard let channel = sender.representedObject as? UpdateChannel else { return }
        updateManager.currentChannel = channel
    }

    @objc private func clearDataClicked() {
        viewModel.clearAllData()
    }

    @objc private func toggleShowInDock() {
        showInDock.toggle()
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
