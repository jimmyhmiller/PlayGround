import Foundation

// MARK: - Tool Call Info

public struct ACPToolCallInfo: Sendable {
    public let id: String
    public let name: String
    public let status: ACPToolCallStatus
    public let input: String?
    public let output: String?

    public init(id: String, name: String, status: ACPToolCallStatus, input: String? = nil, output: String? = nil) {
        self.id = id
        self.name = name
        self.status = status
        self.input = input
        self.output = output
    }
}

// MARK: - Interrupted Prompt State

public struct InterruptedPromptState: Sendable {
    public let text: String
    public let toolCalls: [String: ACPToolCallInfo]
    public let timestamp: Date

    public init(text: String, toolCalls: [String: ACPToolCallInfo], timestamp: Date = Date()) {
        self.text = text
        self.toolCalls = toolCalls
        self.timestamp = timestamp
    }

    /// Format the interrupted text with marker
    public var formattedText: String {
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "[Interrupted]"
        }
        // Always ensure there's a blank line before [Interrupted]
        if text.hasSuffix("\n") {
            return text + "\n[Interrupted]"
        }
        return text + "\n\n[Interrupted]"
    }
}

// MARK: - Prompt Tracker

/// Tracks streaming state for interruption handling
public actor PromptTracker {
    // MARK: - Streaming State

    private var streamingText: String = ""
    private var streamingToolCalls: [String: ACPToolCallInfo] = [:]
    private var currentPromptId: String?

    // MARK: - Prompt ID

    /// Set the current prompt being tracked
    public func setPromptId(_ id: String) {
        currentPromptId = id
    }

    /// Get the current prompt ID
    public var promptId: String? { currentPromptId }

    // MARK: - Text Accumulation

    /// Append text to the current streaming content
    public func appendText(_ text: String) {
        streamingText += text
    }

    /// Get the current accumulated text
    public var currentText: String { streamingText }

    // MARK: - Tool Call Tracking

    /// Update a tool call's information
    public func updateToolCall(id: String, info: ACPToolCallInfo) {
        streamingToolCalls[id] = info
    }

    /// Start tracking a new tool call
    public func startToolCall(id: String, name: String, input: String?) {
        streamingToolCalls[id] = ACPToolCallInfo(
            id: id,
            name: name,
            status: .running,
            input: input,
            output: nil
        )
    }

    /// Complete a tool call
    public func completeToolCall(id: String, status: ACPToolCallStatus, output: String?, error: String?) {
        if let info = streamingToolCalls[id] {
            streamingToolCalls[id] = ACPToolCallInfo(
                id: info.id,
                name: info.name,
                status: status,
                input: info.input,
                output: output ?? error
            )
        }
    }

    /// Get all current tool calls
    public var toolCalls: [String: ACPToolCallInfo] { streamingToolCalls }

    // MARK: - State Checks

    /// Check if there's any accumulated content
    public var hasContent: Bool {
        !streamingText.isEmpty || !streamingToolCalls.isEmpty
    }

    // MARK: - Interruption

    /// Capture the current state for an interrupted prompt
    public func captureInterruptedState() -> InterruptedPromptState {
        let state = InterruptedPromptState(
            text: streamingText,
            toolCalls: streamingToolCalls,
            timestamp: Date()
        )
        clearState()
        return state
    }

    /// Clear all streaming state
    public func clearState() {
        streamingText = ""
        streamingToolCalls = [:]
        currentPromptId = nil
    }
}
