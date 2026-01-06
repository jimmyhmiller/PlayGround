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
// - Message queueing and interruption handling
// - Session history loading
// - Mode management
//
// ## Usage
//
// ```swift
// // Create a client
// let client = ACPClient.forClaudeCode()
//
// // Connect to Claude Code ACP
// try await client.connectToClaudeCodeACP(
//     acpPath: "/usr/local/bin/claude-code-acp",
//     currentDirectory: "/path/to/project"
// )
//
// // Create a session
// let sessionId = try await client.newSession(cwd: "/path/to/project")
//
// // Listen for events
// for await event in await client.events {
//     switch event {
//     case .textChunk(let text, _):
//         print(text, terminator: "")
//     case .toolCallStarted(let id, let name, _, _):
//         print("Tool: \(name)")
//     case .promptComplete(let reason, _):
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
public let acpLibraryVersion = "1.0.0"

/// ACP protocol version
public let acpProtocolVersion = 1

// MARK: - Debug Logging

/// Enable/disable ACP debug logging
public var acpDebugLoggingEnabled = true

/// Log to stderr so it shows in terminal
public func acpLog(_ message: String) {
    guard acpDebugLoggingEnabled else { return }
    let timestamp = ISO8601DateFormatter().string(from: Date())
    fputs("[\(timestamp)] [ACP] \(message)\n", stderr)
}

public func acpLogDebug(_ message: String) {
    guard acpDebugLoggingEnabled else { return }
    let timestamp = ISO8601DateFormatter().string(from: Date())
    fputs("[\(timestamp)] [ACP:DEBUG] \(message)\n", stderr)
}

public func acpLogError(_ message: String) {
    guard acpDebugLoggingEnabled else { return }
    let timestamp = ISO8601DateFormatter().string(from: Date())
    fputs("[\(timestamp)] [ACP:ERROR] \(message)\n", stderr)
}
