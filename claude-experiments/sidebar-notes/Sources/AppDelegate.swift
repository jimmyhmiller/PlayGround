import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var sidebarController: SidebarController!

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        installQuitMenu()

        sidebarController = SidebarController()
        sidebarController.start()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    // Accessory apps don't show a menu bar, but a main menu still wires up keyboard shortcuts
    // so ⌘Q quits when the panel is the key window.
    private func installQuitMenu() {
        let mainMenu = NSMenu()
        let appItem = NSMenuItem()
        mainMenu.addItem(appItem)

        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Quit",
                        action: #selector(NSApplication.terminate(_:)),
                        keyEquivalent: "q")
        appItem.submenu = appMenu

        NSApp.mainMenu = mainMenu
    }
}
