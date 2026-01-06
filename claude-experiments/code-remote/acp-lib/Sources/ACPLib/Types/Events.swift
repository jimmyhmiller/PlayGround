import Foundation

// MARK: - ACP Client Events

/// Events emitted by ACPClient during operation
public enum ACPEvent: Sendable {
    // Connection events
    case connected(agentInfo: ACPAgentInfo)
    case disconnected

    // Session events
    case sessionCreated(sessionId: String, modes: ACPModeInfo?)
    case sessionLoaded(sessionId: String, modes: ACPModeInfo?)
    case sessionResumed(sessionId: String, modes: ACPModeInfo?)

    // Streaming events (with promptId for filtering stale events)
    case textChunk(String, promptId: String)
    case thinkingChunk(String, promptId: String)

    // Tool events
    case toolCallStarted(id: String, name: String, input: String?, promptId: String)
    case toolCallUpdate(id: String, status: String, title: String?, input: String?, output: String?, error: String?, promptId: String)

    // Plan events
    case planStep(id: String, title: String, status: String, promptId: String)

    // Mode events
    case modeChanged(modeId: String)

    // Completion events
    case promptComplete(stopReason: String, promptId: String)
    case promptInterrupted(partialText: String, promptId: String)

    // Error events
    case error(ACPError)
}

// MARK: - Tool Call Status

public enum ACPToolCallStatus: String, Sendable, Codable {
    case pending
    case running
    case complete
    case error
    case cancelled
}

// MARK: - Prompt Result

public struct ACPPromptResultInfo: Sendable {
    public let stopReason: String
    public let promptId: String
    public let wasInterrupted: Bool

    public init(stopReason: String, promptId: String, wasInterrupted: Bool = false) {
        self.stopReason = stopReason
        self.promptId = promptId
        self.wasInterrupted = wasInterrupted
    }
}
