import Cocoa

print("Starting app...")

let app = NSApplication.shared
app.setActivationPolicy(.regular)

print("Creating window...")

let window = NSWindow(
    contentRect: NSRect(x: 100, y: 100, width: 1200, height: 800),
    styleMask: [.titled, .closable, .miniaturizable, .resizable],
    backing: .buffered,
    defer: false
)

window.title = "LightTable Clone"
window.backgroundColor = NSColor(red: 0.1, green: 0.1, blue: 0.12, alpha: 1.0)

let viewController = MainViewController()
window.contentViewController = viewController

print("Created window, showing...")

window.makeKeyAndOrderFront(nil)
app.activate(ignoringOtherApps: true)

print("About to run app...")

app.run()