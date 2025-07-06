import Cocoa
import MetalKit

class ViewController: NSViewController {
    @IBOutlet weak var flameGraphView: FlameGraphView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        loadDefaultStackTrace()
    }
    
    private func setupUI() {
        if flameGraphView == nil {
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
    }
    
    private func loadDefaultStackTrace() {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let stackTraceURL = documentsPath.appendingPathComponent("../../../stack_trace.json")
        
        if FileManager.default.fileExists(atPath: stackTraceURL.path) {
            loadStackTrace(from: stackTraceURL)
        }
    }
    
    private func loadStackTrace(from url: URL) {
        do {
            let stackTrace = try StackTraceParser.parse(from: url)
            flameGraphView.loadStackTrace(stackTrace)
        } catch {
            let alert = NSAlert()
            alert.messageText = "Error Loading Stack Trace"
            alert.informativeText = error.localizedDescription
            alert.alertStyle = .warning
            alert.runModal()
        }
    }
    
    @IBAction func openFile(_ sender: Any) {
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
    
    override var representedObject: Any? {
        didSet {
            
        }
    }
}