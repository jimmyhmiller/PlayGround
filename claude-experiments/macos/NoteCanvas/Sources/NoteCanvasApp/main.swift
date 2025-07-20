import AppKit
import NoteCanvas
import Foundation

class AppDelegate: NSObject, NSApplicationDelegate, PageNavigationDelegate, PDFOpeningDelegate {
    var window: NSWindow!
    var canvas: Canvas!
    var pageNavigationController: PageNavigationController!
    var canvasView: CanvasView!
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        canvas = Canvas()
        canvasView = CanvasView(canvas: canvas)
        canvasView.pdfOpeningDelegate = self // Set PDF opening delegate
        
        // Create page navigation system
        pageNavigationController = PageNavigationController()
        pageNavigationController.delegate = self
        
        // Force view to load
        _ = pageNavigationController.view
        
        // Set canvas as the initial page
        let canvasPage = CanvasPage(canvasView: canvasView)
        pageNavigationController.setInitialPage(canvasPage)
        
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1200, height: 800),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        
        window.title = "Note Canvas"
        window.contentView = pageNavigationController.view
        window.center()
        window.makeKeyAndOrderFront(nil)
        
        // Force the app to activate and come to front
        NSApp.activate(ignoringOtherApps: true)
        
        // Make the canvas view the first responder to receive keyboard events
        window.makeFirstResponder(canvasView)
        
        addSampleNotes()
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
    
    // MARK: - PageNavigationDelegate
    
    func pageNavigation(_ controller: PageNavigationController, didNavigateToPage pageIndex: Int) {
        print("Navigated to page index: \(pageIndex)")
    }
    
    // MARK: - PDF Page Management
    
    func openPDFPage(_ pdfNote: PDFNote) {
        let pdfPage = PDFPage(pdfNote: pdfNote)
        pageNavigationController.pushPage(pdfPage, animated: true)
    }
    
    // MARK: - PDFOpeningDelegate
    
    func canvasViewDidRequestPDFTab(_ canvasView: CanvasView, for pdfNote: PDFNote) {
        openPDFPage(pdfNote)
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
        
        // Add the default PDF for testing
        let pdfPath = URL(fileURLWithPath: "/Users/jimmyhmiller/Desktop/Alvin Plantinga - Warrant and Proper Function (1993) (dragged) 2.pdf")
        if FileManager.default.fileExists(atPath: pdfPath.path) {
            let pdfNote = PDFNote(
                position: CGPoint(x: 800, y: 100),
                size: CGSize(width: 300, height: 400),
                pdfPath: pdfPath
            )
            canvas.addNote(pdfNote)
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()