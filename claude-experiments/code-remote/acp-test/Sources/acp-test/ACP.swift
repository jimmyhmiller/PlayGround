import Foundation

// MARK: - ACP Library
//
// A clean, reusable Swift library for the Agent Client Protocol (ACP).
//
// ## Overview
//
// ACP is a JSON-RPC 2.0 based protocol for communication between AI agent
// clients and agents. This library provides:
//
// - Full JSON-RPC 2.0 implementation
// - All ACP message types
// - Subprocess-based connection (for local agents)
// - High-level ACPClient for easy integration
//
// ## Usage
//
// ```swift
// // Create a client
// let client = ACPClient.forClaudeCode()
//
// // Connect to Claude Code ACP
// try await client.connectToClaudeCodeACP(
//     claudePath: "/usr/local/bin/claude",
//     currentDirectory: "/path/to/project"
// )
//
// // Create a session
// let sessionId = try await client.newSession(cwd: "/path/to/project")
//
// // Listen for events
// for await event in await client.events {
//     switch event {
//     case .textChunk(let text):
//         print(text, terminator: "")
//     case .toolCallStarted(let id, let name, _):
//         print("Tool: \(name)")
//     case .promptComplete(let reason):
//         print("\nDone: \(reason)")
//     default:
//         break
//     }
// }
//
// // Send a prompt (in parallel with event listening)
// Task {
//     let result = try await client.prompt(text: "Hello, Claude!")
// }
// ```
//
// ## Protocol Reference
//
// See https://agentclientprotocol.com for the full ACP specification.
//

// MARK: - Version

/// ACP library version
let acpLibraryVersion = "1.0.0"

// MARK: - Debug Logging

/// Enable/disable ACP debug logging
var acpDebugLoggingEnabled = false

func acpLog(_ message: String) {
    if acpDebugLoggingEnabled {
        print("[ACP] \(message)")
    }
}
