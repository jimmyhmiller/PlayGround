import SwiftUI
import AppKit
import Sparkle
import Combine

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
    private var cancellables = Set<AnyCancellable>()
    private var iconUpdateWorkItem: DispatchWorkItem?

    private static let showInDockKey = "showInDock"
    private static let iconStyleKey = "iconStyle"

    private var showInDock: Bool {
        get { UserDefaults.standard.bool(forKey: Self.showInDockKey) }
        set {
            UserDefaults.standard.set(newValue, forKey: Self.showInDockKey)
            updateDockVisibility()
        }
    }

    // Icon style from UserDefaults (for data proportions toggle)
    // 0 = use bundled proportions, 1 = data proportions
    private var useDataProportions: Bool {
        get { UserDefaults.standard.bool(forKey: Self.iconStyleKey) }
        set {
            UserDefaults.standard.set(newValue, forKey: Self.iconStyleKey)
            updateIcons()
        }
    }

    // Read bar proportions from bundled config
    private var bundledBarWidths: [Double] {
        if let configURL = Bundle.main.url(forResource: "MenuBarConfig", withExtension: "plist"),
           let data = try? Data(contentsOf: configURL),
           let plist = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any],
           let widths = plist["barWidths"] as? [Double] {
            return widths
        }
        // Default: Short / Long / Medium
        return [7, 14, 11]
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupStatusItem()
        setupPopover()
        setupAppearanceObserver()
        setupDataObservers()
        updateIcons()
        updateDockVisibility()
    }

    private func updateDockVisibility() {
        NSApp.setActivationPolicy(showInDock ? .regular : .accessory)
    }

    private func setupAppearanceObserver() {
        appearanceObserver = NSApp.observe(\.effectiveAppearance, options: [.new]) { [weak self] _, _ in
            Task { @MainActor in
                self?.updateIcons()
            }
        }
    }

    private func setupDataObservers() {
        // Observe changes to goals and entries with debouncing
        viewModel.$goals
            .combineLatest(viewModel.$entries)
            .sink { [weak self] _, _ in
                self?.scheduleIconUpdate()
            }
            .store(in: &cancellables)
    }

    private func scheduleIconUpdate() {
        // Debounce icon updates to avoid excessive regeneration
        iconUpdateWorkItem?.cancel()
        let workItem = DispatchWorkItem { [weak self] in
            Task { @MainActor in
                self?.updateIcons()
            }
        }
        iconUpdateWorkItem = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1, execute: workItem)
    }

    private func updateIcons() {
        updateMenuBarIcon()
        // Dock icon uses AppIcon.icon automatically - system handles dark/light mode
    }

    private func updateMenuBarIcon() {
        if let button = statusItem.button {
            button.image = createMenuBarIcon()
            button.image?.isTemplate = true
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

    // MARK: - Dynamic Icon Generation

    private func createMenuBarIcon() -> NSImage? {
        let proportions = viewModel.iconProportions(for: .all)
        let barWidths = bundledBarWidths

        var rects = ""
        let yPositions = [2.0, 7.0, 12.5]
        let barHeight = 4.0

        if useDataProportions {
            // Widths are directly proportional to each other
            let maxWidth: Double = 14
            let minWidth: Double = 3

            // Find max proportion to scale bars relative to it
            let maxProportion = proportions.prefix(3).map { $0.proportion }.max() ?? 1.0

            for (index, item) in proportions.prefix(3).enumerated() {
                let width = maxProportion > 0 ? max(minWidth, maxWidth * (item.proportion / maxProportion)) : maxWidth / 2
                let y = yPositions[index]
                rects += """
                    <rect x="2" y="\(y)" width="\(width)" height="\(barHeight)" rx="2" fill="black"/>

                """
            }

            // If fewer than 3 goals, add placeholder bars
            for i in proportions.count..<3 {
                let y = yPositions[i]
                rects += """
                    <rect x="2" y="\(y)" width="\(barWidths[i])" height="\(barHeight)" rx="2" fill="black"/>

                """
            }
        } else {
            // Use bundled bar widths
            for (index, width) in barWidths.enumerated() {
                let y = yPositions[index]
                rects += """
                    <rect x="2" y="\(y)" width="\(width)" height="\(barHeight)" rx="2" fill="black"/>

                """
            }
        }

        let svg = """
            <svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">
            \(rects)</svg>
            """

        return svgToImage(svg)
    }

    private func colorFromHex(_ hex: String) -> NSColor {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (r, g, b) = ((int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (r, g, b) = (int >> 16, int >> 8 & 0xFF, int & 0xFF)
        default:
            (r, g, b) = (128, 128, 128)
        }
        return NSColor(
            red: CGFloat(r) / 255,
            green: CGFloat(g) / 255,
            blue: CGFloat(b) / 255,
            alpha: 1.0
        )
    }

    private func svgToImage(_ svgString: String) -> NSImage? {
        guard let data = svgString.data(using: .utf8) else { return nil }
        guard let svgImage = NSImage(data: data) else { return nil }

        // Create a template image at the right size for menu bar
        let size = NSSize(width: 18, height: 18)
        let image = NSImage(size: size, flipped: false) { rect in
            svgImage.draw(in: rect)
            return true
        }
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

        // Data Proportions toggle
        let dataProportionsItem = NSMenuItem(
            title: "Data Proportions",
            action: #selector(toggleDataProportions),
            keyEquivalent: ""
        )
        dataProportionsItem.state = useDataProportions ? .on : .off
        menu.addItem(dataProportionsItem)

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

    @objc private func toggleDataProportions() {
        useDataProportions.toggle()
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
