import SwiftUI
import AppKit

@main
struct LogViewerApp: App {
    @State private var selectedFileURL: URL?
    
    init() {
        NSApplication.shared.setActivationPolicy(.regular)
    }
    
    var body: some Scene {
        WindowGroup {
            if let url = selectedFileURL {
                VirtualContentView(url: url)
                    .preferredColorScheme(.dark)
                    .onAppear {
                        NSApplication.shared.activate(ignoringOtherApps: true)
                    }
            } else {
                WelcomeView {
                    openFile()
                }
                .preferredColorScheme(.dark)
                .onAppear {
                    NSApplication.shared.activate(ignoringOtherApps: true)
                }
            }
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Log File...") {
                    openFile()
                }
                .keyboardShortcut("o", modifiers: .command)
            }
        }
    }
    
    private func openFile() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.text, .json, .log, .plainText]
        
        if panel.runModal() == .OK, let url = panel.url {
            selectedFileURL = url
        }
    }
}

struct WelcomeView: View {
    let onOpenFile: () -> Void
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Log Viewer")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            Text("Drop a log file here or click to open")
                .foregroundColor(.secondary)
            
            Button("Open Log File") {
                onOpenFile()
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(red: 0.15, green: 0.15, blue: 0.16))
    }
}

struct VirtualContentView: View {
    let url: URL
    
    var body: some View {
        HStack(spacing: 0) {
            VirtualTimelineSidebar(url: url)
                .frame(width: 80)
                .background(Color(red: 0.12, green: 0.12, blue: 0.13))
            
            Divider()
                .background(Color(red: 0.2, green: 0.2, blue: 0.2))
            
            VirtualLogContentView(url: url)
                .background(Color(red: 0.15, green: 0.15, blue: 0.16))
        }
        .frame(minWidth: 800, minHeight: 600)
        .preferredColorScheme(.dark)
    }
}