import Foundation

// MARK: - ACP Errors

/// High-level ACP errors
public enum ACPError: Error, LocalizedError, Sendable {
    case notConnected
    case noActiveSession
    case sessionNotFound(String)
    case connectionFailed(String)
    case promptFailed(String)
    case cancelled
    case invalidState(String)
    case notSupported(String)

    public var errorDescription: String? {
        switch self {
        case .notConnected:
            return "Not connected to ACP agent"
        case .noActiveSession:
            return "No active session"
        case .sessionNotFound(let sessionId):
            return "Session not found: \(sessionId)"
        case .connectionFailed(let reason):
            return "Connection failed: \(reason)"
        case .promptFailed(let reason):
            return "Prompt failed: \(reason)"
        case .cancelled:
            return "Operation was cancelled"
        case .invalidState(let state):
            return "Invalid state: \(state)"
        case .notSupported(let feature):
            return "Not supported: \(feature)"
        }
    }
}

// MARK: - ACP Connection Errors

public enum ACPConnectionError: Error, LocalizedError, Sendable {
    case notConnected
    case processTerminated(exitCode: Int32, stderr: String?)
    case encodingError(String)
    case decodingError(String)
    case timeout
    case connectionClosed
    case invalidResponse(String)

    public var errorDescription: String? {
        switch self {
        case .notConnected:
            return "Not connected to ACP agent"
        case .processTerminated(let code, let stderr):
            if let stderr = stderr, !stderr.isEmpty {
                // Extract the key error message
                let lines = stderr.components(separatedBy: "\n")
                let errorLines = lines.filter {
                    $0.contains("Error") || $0.contains("error:") || $0.contains("SyntaxError")
                }
                if let firstError = errorLines.first {
                    return "Agent failed: \(firstError)"
                }
                return "Agent failed (code \(code)): \(stderr.prefix(200))"
            }
            return "Agent process terminated with code \(code)"
        case .encodingError(let msg):
            return "Encoding error: \(msg)"
        case .decodingError(let msg):
            return "Decoding error: \(msg)"
        case .timeout:
            return "Request timed out"
        case .connectionClosed:
            return "Connection closed"
        case .invalidResponse(let msg):
            return "Invalid response: \(msg)"
        }
    }
}

// MARK: - ACP Error Codes

public enum ACPErrorCode: Int, Sendable {
    case parseError = -32700
    case invalidRequest = -32600
    case methodNotFound = -32601
    case invalidParams = -32602
    case internalError = -32603

    // ACP-specific errors
    case sessionNotFound = -32001
    case authenticationRequired = -32002
    case permissionDenied = -32003
    case resourceNotFound = -32004
}
