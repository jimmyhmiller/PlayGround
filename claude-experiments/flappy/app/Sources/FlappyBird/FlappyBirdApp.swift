import SwiftUI
import WebKit

@main
struct FlappyBirdApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

struct ContentView: View {
    @StateObject private var viewModel = FlappyBirdViewModel()
    
    var body: some View {
        VStack(spacing: 0) {
            // Title bar
            HStack {
                Text("🐦 Flappy Bird Premium")
                    .font(.title2)
                    .fontWeight(.bold)
                
                Spacer()
                
                if viewModel.isServerRunning {
                    Circle()
                        .fill(Color.green)
                        .frame(width: 10, height: 10)
                    Text("Server Running")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 10, height: 10)
                    Text("Server Stopped")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Button(action: {
                    viewModel.toggleServer()
                }) {
                    Image(systemName: viewModel.isServerRunning ? "stop.circle" : "play.circle")
                        .font(.title2)
                }
                .buttonStyle(.plain)
                .help(viewModel.isServerRunning ? "Stop Server" : "Start Server")
                
                Button(action: {
                    viewModel.refreshWebView()
                }) {
                    Image(systemName: "arrow.clockwise")
                        .font(.title2)
                }
                .buttonStyle(.plain)
                .help("Refresh Game")
            }
            .padding()
            .background(Color(NSColor.windowBackgroundColor))
            
            Divider()
            
            // WebView
            WebView(viewModel: viewModel)
                .frame(minWidth: 800, minHeight: 650)
        }
        .frame(width: 800, height: 700)
        .onAppear {
            viewModel.startServer()
        }
        .onDisappear {
            viewModel.stopServer()
        }
    }
}

class FlappyBirdViewModel: ObservableObject {
    @Published var isServerRunning = false
    @Published var shouldRefresh = false
    
    private var serverProcess: Process?
    private let serverURL = "http://localhost:3000"
    
    func startServer() {
        guard !isServerRunning else { return }
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        
        // Get the project root directory (3 levels up from the app directory)
        let appPath = Bundle.main.bundlePath
        let projectRoot = URL(fileURLWithPath: appPath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .path
        
        process.arguments = ["node", "\(projectRoot)/backend/server.js"]
        process.currentDirectoryURL = URL(fileURLWithPath: projectRoot)
        
        let outputPipe = Pipe()
        process.standardOutput = outputPipe
        process.standardError = outputPipe
        
        outputPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if let output = String(data: data, encoding: .utf8), !output.isEmpty {
                print("[Server] \(output)", terminator: "")
            }
        }
        
        do {
            try process.run()
            serverProcess = process
            
            // Wait a bit for server to start
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                self.isServerRunning = true
                self.shouldRefresh = true
            }
        } catch {
            print("Failed to start server: \(error)")
        }
    }
    
    func stopServer() {
        guard let process = serverProcess, process.isRunning else { return }
        process.terminate()
        serverProcess = nil
        isServerRunning = false
    }
    
    func toggleServer() {
        if isServerRunning {
            stopServer()
        } else {
            startServer()
        }
    }
    
    func refreshWebView() {
        shouldRefresh = true
    }
}

struct WebView: NSViewRepresentable {
    @ObservedObject var viewModel: FlappyBirdViewModel
    
    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.preferences.setValue(true, forKey: "developerExtrasEnabled")
        
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        
        return webView
    }
    
    func updateNSView(_ webView: WKWebView, context: Context) {
        if viewModel.shouldRefresh {
            DispatchQueue.main.async {
                viewModel.shouldRefresh = false
                if let url = URL(string: "http://localhost:3000") {
                    let request = URLRequest(url: url)
                    webView.load(request)
                }
            }
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, WKNavigationDelegate {
        var parent: WebView
        
        init(_ parent: WebView) {
            self.parent = parent
        }
        
        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            print("Page loaded successfully")
        }
        
        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            print("Navigation failed: \(error.localizedDescription)")
        }
        
        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            print("Provisional navigation failed: \(error.localizedDescription)")
            
            // Retry after a delay
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                if let url = URL(string: "http://localhost:3000") {
                    let request = URLRequest(url: url)
                    webView.load(request)
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
