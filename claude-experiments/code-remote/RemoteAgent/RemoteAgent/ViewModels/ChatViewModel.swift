import Foundation
import SwiftUI

// MARK: - Chat View Model

@MainActor
class ChatViewModel: ObservableObject {
    // MARK: - Published State

    @Published var messages: [ChatDisplayMessage] = []
    @Published var inputText: String = ""
    @Published var isConnected: Bool = false
    @Published var isLoading: Bool = false
    @Published var isStreaming: Bool = false
    @Published var errorMessage: String?
    @Published var sessionId: String?
    @Published var model: String?
    @Published var totalCost: Double = 0

    // MARK: - Dependencies

    private let sshService: SSHService
    private let claudeClient: ClaudeClient
    private let sessionStore: SessionStore

    private let project: Project
    private var eventTask: Task<Void, Never>?
    private var streamingMessageId: String?

    // MARK: - Init

    init(project: Project) {
        self.project = project
        self.sshService = SSHService()
        self.claudeClient = ClaudeClient(sshService: sshService)
        self.sessionStore = SessionStore()
    }

    deinit {
        eventTask?.cancel()
    }

    // MARK: - Public Methods

    func connect(password: String?) async {
        guard let server = project.server else {
            print("[ChatViewModel] No server configured for project")
            errorMessage = "No server configured for project"
            return
        }

        print("[ChatViewModel] Connecting to \(server.name)...")
        isLoading = true
        errorMessage = nil

        do {
            try await claudeClient.connect(to: server, password: password)
            print("[ChatViewModel] Connected successfully")
            isConnected = true
            startEventListener()
        } catch {
            print("[ChatViewModel] Connect error: \(error)")
            errorMessage = error.localizedDescription
            isConnected = false
        }

        isLoading = false
    }

    func sendMessage() async {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        print("[ChatViewModel] sendMessage: \(text.prefix(50))...")
        inputText = ""
        errorMessage = nil

        // Add user message
        let userMessage = ChatDisplayMessage(role: .user, content: text)
        messages.append(userMessage)

        // Add streaming placeholder for assistant
        let assistantMessage = ChatDisplayMessage(role: .assistant, content: "", isStreaming: true)
        messages.append(assistantMessage)
        streamingMessageId = assistantMessage.id
        isStreaming = true

        do {
            print("[ChatViewModel] Calling executePrompt...")
            try await claudeClient.executePrompt(
                projectPath: project.remotePath,
                prompt: text,
                resumeSessionId: sessionId
            )
            print("[ChatViewModel] executePrompt completed")
        } catch {
            print("[ChatViewModel] executePrompt error: \(error)")
            // Remove streaming placeholder on error
            if let id = streamingMessageId {
                messages.removeAll { $0.id == id }
            }
            errorMessage = error.localizedDescription
            isStreaming = false
        }
    }

    func resumeSession(_ session: Session) async {
        sessionId = session.id

        // Update last active time
        var updatedSession = session
        updatedSession.lastActiveAt = Date()
        await sessionStore.updateSession(updatedSession)
    }

    func disconnect() async {
        eventTask?.cancel()
        await claudeClient.disconnect()
        isConnected = false
        sessionId = nil
    }

    func toggleToolCallExpanded(_ toolCallId: String) {
        for i in messages.indices {
            if let toolIndex = messages[i].toolCalls.firstIndex(where: { $0.id == toolCallId }) {
                messages[i].toolCalls[toolIndex].isExpanded.toggle()
                break
            }
        }
    }

    // MARK: - Private Methods

    private func startEventListener() {
        eventTask = Task { [weak self] in
            guard let self = self else { return }

            for await event in claudeClient.events {
                await self.handleEvent(event)
            }
        }
    }

    private func handleEvent(_ event: ClaudeEvent) async {
        switch event {
        case .connected:
            isConnected = true

        case .disconnected:
            isConnected = false
            sessionId = nil

        case .sessionStarted(let id, let modelName, _):
            sessionId = id
            model = modelName

            // Store session for later resume
            let session = Session(id: id, projectId: project.id)
            await sessionStore.addSession(session)

        case .textDelta(let delta):
            if let messageId = streamingMessageId,
               let index = messages.firstIndex(where: { $0.id == messageId }) {
                messages[index].content += delta
            }

        case .textComplete(let text):
            if let messageId = streamingMessageId,
               let index = messages.firstIndex(where: { $0.id == messageId }) {
                messages[index].content = text
                messages[index].isStreaming = false
            }

        case .toolUseStarted(let id, let name, let input):
            let toolCall = DisplayToolCall(
                id: id,
                name: name,
                input: input,
                output: nil,
                isExpanded: true,
                status: .running
            )

            if let lastIndex = messages.indices.last, messages[lastIndex].role == .assistant {
                messages[lastIndex].toolCalls.append(toolCall)
            }

        case .toolUseCompleted(let id, let result, let isError):
            for i in messages.indices.reversed() {
                if let toolIndex = messages[i].toolCalls.firstIndex(where: { $0.id == id }) {
                    messages[i].toolCalls[toolIndex].output = result
                    messages[i].toolCalls[toolIndex].status = isError ? .error : .completed
                    break
                }
            }

        case .turnComplete(_, let cost, _):
            totalCost += cost
            isStreaming = false
            streamingMessageId = nil

        case .error(let errMsg):
            errorMessage = errMsg
            isStreaming = false

            if let id = streamingMessageId {
                messages.removeAll { $0.id == id }
            }
            streamingMessageId = nil
        }
    }
}
