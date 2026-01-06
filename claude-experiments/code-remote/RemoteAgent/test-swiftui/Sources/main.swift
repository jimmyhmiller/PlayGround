import SwiftUI
import AppKit

@main
struct TestApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}

// Observable class to hold output - can be updated from callbacks
@MainActor
class OutputCollector: ObservableObject {
    @Published var output: String = ""

    nonisolated func append(_ text: String) {
        Task { @MainActor in
            self.output += text
        }
    }
}

struct ContentView: View {
    @State private var isRunning = false
    @StateObject private var collector = OutputCollector()

    var body: some View {
        VStack(spacing: 20) {
            Text("Process Output Test")
                .font(.title)

            Button(isRunning ? "Running..." : "Run npm list") {
                Task {
                    await runProcess()
                }
            }
            .disabled(isRunning)

            ScrollView {
                Text(collector.output.isEmpty ? "No output yet..." : collector.output)
                    .font(.system(.body, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .frame(height: 300)
            .background(Color.black.opacity(0.9))
            .foregroundStyle(collector.output.isEmpty ? .gray : .green)
            .cornerRadius(8)
        }
        .padding()
        .frame(width: 600, height: 450)
    }

    func runProcess() async {
        isRunning = true
        collector.output = ""

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["npm", "list", "-g", "--depth=0"]

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        stdoutPipe.fileHandleForReading.readabilityHandler = { [collector] handle in
            let data = handle.availableData
            if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
                collector.append(output)
            }
        }

        stderrPipe.fileHandleForReading.readabilityHandler = { [collector] handle in
            let data = handle.availableData
            if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
                collector.append(output)
            }
        }

        do {
            try process.run()
        } catch {
            collector.append("Error: \(error)")
            isRunning = false
            return
        }

        await Task.detached {
            process.waitUntilExit()
            stdoutPipe.fileHandleForReading.readabilityHandler = nil
            stderrPipe.fileHandleForReading.readabilityHandler = nil
        }.value

        isRunning = false
    }
}
