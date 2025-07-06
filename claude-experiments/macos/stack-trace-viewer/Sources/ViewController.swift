import Cocoa
import MetalKit
import UniformTypeIdentifiers

class ViewController: NSViewController {
    private var flameGraphView: FlameGraphView!
    
    override func loadView() {
        view = NSView(frame: NSRect(x: 0, y: 0, width: 1200, height: 800))
        setupUI()
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        loadDefaultStackTrace()
    }
    
    private func setupUI() {
        let metalView = FlameGraphView(frame: view.bounds)
        metalView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(metalView)
        
        NSLayoutConstraint.activate([
            metalView.topAnchor.constraint(equalTo: view.topAnchor),
            metalView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            metalView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            metalView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
        
        flameGraphView = metalView
    }
    
    private func loadDefaultStackTrace() {
        let currentDir = FileManager.default.currentDirectoryPath
        let stackTraceURL = URL(fileURLWithPath: currentDir).appendingPathComponent("stack_trace.json")
        
        if FileManager.default.fileExists(atPath: stackTraceURL.path) {
            loadStackTrace(from: stackTraceURL)
        }
    }
    
    private func loadStackTrace(from url: URL) {
        do {
            let stackTrace = try StackTraceParser.parse(from: url)
            print("Loaded stack trace with \(stackTrace.totalCaptures) captures")
            flameGraphView.loadStackTrace(stackTrace)
        } catch {
            print("Error loading stack trace: \(error.localizedDescription)")
            let alert = NSAlert()
            alert.messageText = "Error Loading Stack Trace"
            alert.informativeText = error.localizedDescription
            alert.alertStyle = .warning
            alert.runModal()
        }
    }
    
    @objc func openFile(_ sender: Any) {
        let openPanel = NSOpenPanel()
        openPanel.allowedContentTypes = [UTType.json]
        openPanel.allowsMultipleSelection = false
        openPanel.canChooseDirectories = false
        openPanel.canChooseFiles = true
        
        if openPanel.runModal() == .OK {
            if let url = openPanel.url {
                loadStackTrace(from: url)
            }
        }
    }
}