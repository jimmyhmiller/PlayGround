import AppKit
import NoteCanvas

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var canvas: Canvas!
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        canvas = Canvas()
        
        let canvasView = CanvasView(canvas: canvas)
        
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1200, height: 800),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        
        window.title = "Note Canvas"
        window.contentView = canvasView
        window.center()
        window.makeKeyAndOrderFront(nil)
        
        addSampleNotes()
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
    
    private func addSampleNotes() {
        let textNote = TextNote(
            position: CGPoint(x: 100, y: 100),
            size: CGSize(width: 250, height: 200),
            content: "Welcome to Note Canvas!\n\nThis is a text note. You can edit this content.",
            font: NSFont.systemFont(ofSize: 14)
        )
        canvas.addNote(textNote)
        
        let stickyNote = StickyNote(
            position: CGPoint(x: 400, y: 150),
            size: CGSize(width: 180, height: 180),
            content: "Don't forget to add animations!",
            stickyColor: StickyColor.yellow
        )
        canvas.addNote(stickyNote)
        
        let stickyNote2 = StickyNote(
            position: CGPoint(x: 600, y: 200),
            size: CGSize(width: 180, height: 180),
            content: "Test multi-selection with drag",
            stickyColor: StickyColor.pink
        )
        canvas.addNote(stickyNote2)
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()