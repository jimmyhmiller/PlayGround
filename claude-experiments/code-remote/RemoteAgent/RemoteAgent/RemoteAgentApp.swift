import SwiftUI
import SwiftData
import Citadel
import NIOSSH
import Crypto
import _CryptoExtras
#if os(macOS)
import AppKit
#else
import UIKit
#endif

#if os(macOS)
// Set to true to run SSH debug test on launch
let RUN_SSH_DEBUG_TEST = false
#endif

@main
struct RemoteAgentApp: App {
    #if os(macOS)
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    #endif

    init() {
        // Pre-warm NIO/Citadel in background to avoid UI freeze on first use
        Task.detached(priority: .background) {
            await prewarmNetworkingLibraries()
        }

        #if os(macOS)
        if RUN_SSH_DEBUG_TEST {
            Task {
                await runSSHDebugTest()
            }
        }
        #endif
    }

    var sharedModelContainer: ModelContainer = {
        // Ensure Application Support directory exists before SwiftData tries to create the store
        let fileManager = FileManager.default
        if let appSupportURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            if !fileManager.fileExists(atPath: appSupportURL.path) {
                try? fileManager.createDirectory(at: appSupportURL, withIntermediateDirectories: true)
            }
        }

        let schema = Schema([
            Server.self,
            Project.self,
            CachedSession.self,
            CachedMessage.self,
        ])
        let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)

        do {
            return try ModelContainer(for: schema, configurations: [modelConfiguration])
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }()

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(sharedModelContainer)
    }
}

/// Pre-warm NIO and Citadel libraries to avoid UI freeze on first SSH connection
/// This initializes the event loop and crypto in the background at app startup
private func prewarmNetworkingLibraries() async {
    // Simply creating SSHClientSettings triggers NIO event loop initialization
    // We don't actually connect - just warm up the infrastructure
    _ = SSHClientSettings(
        host: "localhost",
        port: 22,
        authenticationMethod: { .passwordBased(username: "dummy", password: "dummy") },
        hostKeyValidator: .acceptAnything()
    )
    // Touch crypto to initialize it
    _ = Curve25519.Signing.PrivateKey()
}

#if os(macOS)
func runSSHDebugTest() async {
    print("\n========== SSH DEBUG TEST ==========")

    let host = "computer.jimmyhmiller.com"
    let username = "jimmyhmiller"
    let keyPath = FileManager.default.homeDirectoryForCurrentUser.path + "/.ssh/id_ed25519"

    print("[DEBUG] Host: \(username)@\(host)")
    print("[DEBUG] Key: \(keyPath)")

    do {
        // Read key
        let keyData = try Data(contentsOf: URL(fileURLWithPath: keyPath))
        guard let keyString = String(data: keyData, encoding: .utf8) else {
            print("[DEBUG] ERROR: Could not read key as UTF-8")
            return
        }
        print("[DEBUG] Key loaded, length: \(keyString.count)")

        // Parse key
        let ed25519Key = try Curve25519.Signing.PrivateKey(sshEd25519: keyString)
        print("[DEBUG] Key parsed successfully")

        let authMethod = SSHAuthenticationMethod.ed25519(username: username, privateKey: ed25519Key)

        // Connect
        print("[DEBUG] Connecting...")
        let client = try await SSHClient.connect(
            host: host,
            port: 22,
            authenticationMethod: authMethod,
            hostKeyValidator: .acceptAnything(),
            reconnect: .never
        )
        print("[DEBUG] Connected!")

        // Test simple command
        print("[DEBUG] Testing simple command: echo 'hello'")
        let simpleOutput = try await client.executeCommand("echo 'hello'")
        print("[DEBUG] Simple output: \(String(buffer: simpleOutput))")

        // Test claude command with timeout
        print("[DEBUG] Testing claude command...")
        let claudeCommand = "bash -c 'cd /home/jimmyhmiller && /usr/local/bin/claude -p \"test\" --output-format stream-json --verbose'"
        print("[DEBUG] Command: \(claudeCommand)")

        // Try executeCommandStream instead
        print("[DEBUG] Using executeCommandStream...")
        let stream = try await client.executeCommandStream(claudeCommand)

        print("[DEBUG] Stream obtained, reading output...")
        var totalBytes = 0
        var lineBuffer = Data()

        for try await chunk in stream {
            switch chunk {
            case .stdout(let buffer):
                let bytes = buffer.readableBytesView
                totalBytes += bytes.count
                lineBuffer.append(contentsOf: bytes)
                print("[DEBUG] stdout chunk: \(bytes.count) bytes, total: \(totalBytes)")

                // Print any complete lines
                while let newlineIndex = lineBuffer.firstIndex(of: UInt8(ascii: "\n")) {
                    let lineData = lineBuffer.prefix(upTo: newlineIndex)
                    lineBuffer = Data(lineBuffer.suffix(from: lineBuffer.index(after: newlineIndex)))
                    if let line = String(data: Data(lineData), encoding: .utf8) {
                        print("[DEBUG] LINE: \(line.prefix(100))...")
                    }
                }

            case .stderr(let buffer):
                let bytes = buffer.readableBytesView
                print("[DEBUG] stderr chunk: \(bytes.count) bytes")
                if let text = String(bytes: bytes, encoding: .utf8) {
                    print("[DEBUG] stderr: \(text.prefix(200))")
                }
            }
        }

        print("[DEBUG] Stream completed, total bytes: \(totalBytes)")

        // Cleanup
        try await client.close()
        print("[DEBUG] Disconnected")

    } catch {
        print("[DEBUG] ERROR: \(error)")
    }

    print("========== END SSH DEBUG TEST ==========\n")
}
#endif

#if os(macOS)
class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}
#endif
