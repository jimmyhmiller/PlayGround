import Foundation

// MARK: - ACP Client Delegate

public protocol ACPClientDelegate: Actor {
    /// Called when an event is received
    func acpClient(_ client: ACPClient, didReceive event: ACPEvent) async

    /// Called when the agent requests permission for a tool
    func acpClient(_ client: ACPClient, requestPermissionFor toolName: String, input: String?, prompt: String?) async -> (granted: Bool, context: String?)

    /// Called when the agent wants to read a file
    func acpClient(_ client: ACPClient, readFile path: String, startLine: Int?, endLine: Int?) async throws -> (content: String, lineCount: Int?)

    /// Called when the agent wants to write a file
    func acpClient(_ client: ACPClient, writeFile path: String, content: String, createDirectories: Bool) async throws -> Int
}

// Default implementations for optional delegate methods
public extension ACPClientDelegate {
    func acpClient(_ client: ACPClient, requestPermissionFor toolName: String, input: String?, prompt: String?) async -> (granted: Bool, context: String?) {
        // Default: grant all permissions (for demo purposes)
        return (true, nil)
    }

    func acpClient(_ client: ACPClient, readFile path: String, startLine: Int?, endLine: Int?) async throws -> (content: String, lineCount: Int?) {
        // Default: read from local filesystem
        let url = URL(fileURLWithPath: path)
        let content = try String(contentsOf: url, encoding: .utf8)
        let lines = content.components(separatedBy: "\n")

        if let start = startLine, let end = endLine {
            let sliced = lines[max(0, start - 1)..<min(lines.count, end)]
            return (sliced.joined(separator: "\n"), lines.count)
        }

        return (content, lines.count)
    }

    func acpClient(_ client: ACPClient, writeFile path: String, content: String, createDirectories: Bool) async throws -> Int {
        // Default: write to local filesystem
        let url = URL(fileURLWithPath: path)

        if createDirectories {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
        }

        try content.write(to: url, atomically: true, encoding: .utf8)
        return content.utf8.count
    }
}
