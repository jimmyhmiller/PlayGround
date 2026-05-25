import SwiftUI
import AppKit
import Shared

@main
struct ReleaseTrackerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        // No automatic window. The AppDelegate opens the floating widget;
        // the checklist window opens on demand from the widget.
        Settings {
            EmptyView()
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private let floatingController = FloatingWidgetController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Show in Dock so users can click the icon to bring the widget
        // back if they close it. (.accessory hides Dock + Cmd-Tab.)
        NSApp.setActivationPolicy(.regular)
        floatingController.show()
        setupMenuBar()
    }

    /// Quitting the app via Cmd-Q closes both windows; keep the process
    /// alive when the user just closes the checklist window.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
        floatingController.show()
        return true
    }

    private func setupMenuBar() {
        let mainMenu = NSMenu()

        let appMenuItem = NSMenuItem()
        let appMenu = NSMenu()
        appMenu.addItem(NSMenuItem(
            title: "Show Checklist",
            action: #selector(showChecklist),
            keyEquivalent: "k"
        ))
        appMenu.addItem(NSMenuItem(
            title: "Show Widget",
            action: #selector(showWidget),
            keyEquivalent: "w"
        ))
        appMenu.addItem(.separator())
        appMenu.addItem(NSMenuItem(
            title: "Quit Ease Release",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        ))
        appMenuItem.submenu = appMenu
        mainMenu.addItem(appMenuItem)

        NSApp.mainMenu = mainMenu
    }

    @objc private func showChecklist() {
        ChecklistWindowController.shared.show()
    }

    @objc private func showWidget() {
        floatingController.show()
    }
}
