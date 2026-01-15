import Foundation

// MARK: - Claude Event (for UI updates)

enum ClaudeEvent: Sendable {
    case connected
    case disconnected
    case sessionStarted(sessionId: String, model: String, tools: [String])
    case textDelta(String)
    case textComplete(String)
    case toolUseStarted(id: String, name: String, input: String)
    case toolUseCompleted(id: String, result: String?, isError: Bool)
    case turnComplete(result: String?, cost: Double, tokens: Int)
    case error(String)
}

// MARK: - Claude Client

@MainActor
class ClaudeClient: ObservableObject {
    private let sshService: SSHService
    private let decoder = JSONDecoder()

    @Published var isConnected = false
    @Published var currentSessionId: String?
    @Published var model: String?
    @Published var availableTools: [String] = []

    private var eventContinuation: AsyncStream<ClaudeEvent>.Continuation?

    init(sshService: SSHService) {
        self.sshService = sshService
    }

    var events: AsyncStream<ClaudeEvent> {
        AsyncStream { continuation in
            self.eventContinuation = continuation
        }
    }

    // MARK: - Connection

    func connect(to server: Server, password: String?) async throws {
        try await sshService.connect(to: server, password: password)
        isConnected = true
        eventContinuation?.yield(.connected)
    }

    func disconnect() async {
        await sshService.disconnect()
        isConnected = false
        currentSessionId = nil
        eventContinuation?.yield(.disconnected)
    }

    // MARK: - Execute Prompt

    func executePrompt(
        projectPath: String,
        prompt: String,
        resumeSessionId: String? = nil
    ) async throws {
        print("[ClaudeClient] executePrompt starting, path=\(projectPath)")
        let stream = try await sshService.executeClaudePrompt(
            projectPath: projectPath,
            prompt: prompt,
            resumeSessionId: resumeSessionId
        )
        print("[ClaudeClient] Got stream, starting to read lines...")

        var lineCount = 0
        for try await line in stream {
            lineCount += 1
            print("[ClaudeClient] Line \(lineCount): \(line.prefix(100))...")
            guard let data = line.data(using: .utf8) else { continue }

            do {
                let message = try decoder.decode(ClaudeMessage.self, from: data)
                print("[ClaudeClient] Decoded message type: \(type(of: message))")
                handleMessage(message)
            } catch {
                // Log but continue - some lines might be malformed
                print("[ClaudeClient] Failed to decode line: \(line.prefix(100)), error: \(error)")
            }
        }
        print("[ClaudeClient] Stream ended, processed \(lineCount) lines")
    }

    // MARK: - Message Handling

    private func handleMessage(_ message: ClaudeMessage) {
        switch message {
        case .system(let sysMsg):
            currentSessionId = sysMsg.sessionId
            model = sysMsg.model
            availableTools = sysMsg.tools
            eventContinuation?.yield(.sessionStarted(
                sessionId: sysMsg.sessionId,
                model: sysMsg.model,
                tools: sysMsg.tools
            ))

        case .assistant(let assistantMsg):
            for block in assistantMsg.message.content {
                switch block {
                case .text(let textBlock):
                    eventContinuation?.yield(.textComplete(textBlock.text))

                case .toolUse(let toolBlock):
                    let inputJson = prettyPrint(toolBlock.input)
                    eventContinuation?.yield(.toolUseStarted(
                        id: toolBlock.id,
                        name: toolBlock.name,
                        input: inputJson
                    ))

                case .toolResult(let resultBlock):
                    let content = resultBlock.content?.asString() ?? resultBlock.content?.prettyPrinted() ?? ""
                    eventContinuation?.yield(.toolUseCompleted(
                        id: resultBlock.toolUseId,
                        result: content,
                        isError: resultBlock.isError ?? false
                    ))

                case .thinking(let thinkingBlock):
                    eventContinuation?.yield(.textDelta("[Thinking] \(thinkingBlock.thinking)"))

                case .unknown:
                    break
                }
            }

        case .user:
            break

        case .result(let resultMsg):
            eventContinuation?.yield(.turnComplete(
                result: resultMsg.result,
                cost: resultMsg.totalCostUsd,
                tokens: (resultMsg.usage?.inputTokens ?? 0) + (resultMsg.usage?.outputTokens ?? 0)
            ))

        case .streamEvent(let streamEvent):
            if let delta = streamEvent.event.delta {
                if delta.type == "text_delta", let text = delta.text {
                    eventContinuation?.yield(.textDelta(text))
                }
            }
        }
    }

    private func prettyPrint(_ input: [String: AnyCodable]) -> String {
        let dict = input.mapValues { $0.value }
        guard let data = try? JSONSerialization.data(withJSONObject: dict, options: .prettyPrinted),
              let string = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return string
    }
}

// MARK: - Display Models

/// A content block that can be either text or a tool call, maintaining order
enum MessageContentBlock: Identifiable {
    case text(id: String, content: String)
    case toolCall(DisplayToolCall)

    var id: String {
        switch self {
        case .text(let id, _): return id
        case .toolCall(let tc): return tc.id
        }
    }

    var asToolCall: DisplayToolCall? {
        if case .toolCall(let tc) = self { return tc }
        return nil
    }
}

struct ChatDisplayMessage: Identifiable {
    let id: String
    let role: DisplayMessageRole
    var contentBlocks: [MessageContentBlock]
    let timestamp: Date
    var isStreaming: Bool

    /// Convenience property to get combined text content
    var content: String {
        contentBlocks.compactMap { block in
            if case .text(_, let content) = block { return content }
            return nil
        }.joined()
    }

    /// Convenience property to get all tool calls
    var toolCalls: [DisplayToolCall] {
        contentBlocks.compactMap { $0.asToolCall }
    }

    init(
        id: String = UUID().uuidString,
        role: DisplayMessageRole,
        content: String,
        toolCalls: [DisplayToolCall] = [],
        timestamp: Date = Date(),
        isStreaming: Bool = false
    ) {
        self.id = id
        self.role = role
        self.timestamp = timestamp
        self.isStreaming = isStreaming

        // Build content blocks - text first, then tool calls
        var blocks: [MessageContentBlock] = []
        if !content.isEmpty {
            blocks.append(.text(id: UUID().uuidString, content: content))
        }
        for tc in toolCalls {
            blocks.append(.toolCall(tc))
        }
        self.contentBlocks = blocks
    }

    init(
        id: String = UUID().uuidString,
        role: DisplayMessageRole,
        contentBlocks: [MessageContentBlock],
        timestamp: Date = Date(),
        isStreaming: Bool = false
    ) {
        self.id = id
        self.role = role
        self.contentBlocks = contentBlocks
        self.timestamp = timestamp
        self.isStreaming = isStreaming
    }
}

enum DisplayMessageRole: String {
    case user
    case assistant
    case system
}

struct DisplayToolCall: Identifiable {
    let id: String
    var name: String
    var input: String
    var output: String?
    var isExpanded: Bool
    var status: ToolCallStatus

    enum ToolCallStatus {
        case running
        case completed
        case error
    }

    init(id: String, name: String, input: String, output: String? = nil, isExpanded: Bool = false, status: ToolCallStatus = .running) {
        self.id = id
        self.name = name
        self.input = input
        self.output = output
        self.isExpanded = isExpanded
        self.status = status
    }
}
