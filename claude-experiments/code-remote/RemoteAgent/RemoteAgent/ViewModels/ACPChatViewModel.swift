import Foundation
import SwiftUI
import ACPLib

// Use the centralized logging that writes to file
private func log(_ message: String) {
    appLog(message, category: "ChatVM")
}

// MARK: - Display Permission Request

struct DisplayPermissionRequest: Identifiable {
    let id: String
    let toolName: String
    let input: String?
    let title: String?
}

// MARK: - ACP Chat View Model

/// ViewModel for chat UI using ACP protocol
@MainActor
class ACPChatViewModel: ObservableObject {
    // MARK: - Published State

    @Published var messages: [ChatDisplayMessage] = []
    @Published var inputText: String = ""
    @Published var isConnected: Bool = false
    @Published var isLoading: Bool = false
    @Published var isStreaming: Bool = false
    @Published var errorMessage: String?
    @Published var sessionId: String?
    @Published var model: String?
    @Published var agentName: String?
    @Published var totalCost: Double = 0
    @Published var availableModes: [ACPMode] = []
    @Published var currentMode: ACPMode?
    @Published var pendingPermission: DisplayPermissionRequest?

    // MARK: - Dependencies

    private let acpService: ACPService
    private let sessionStore: SessionStore

    private let project: Project
    private var eventTask: Task<Void, Never>?
    private var streamingMessageId: String?

    // MARK: - Init

    init(project: Project) {
        self.project = project
        self.acpService = ACPService()
        self.sessionStore = SessionStore()
    }

    deinit {
        eventTask?.cancel()
    }

    // MARK: - Public Methods

    func connect() async {
        guard let server = project.server else {
            log("connect: no server configured for project")
            errorMessage = "No server configured for project"
            return
        }

        log("connect: starting")
        isLoading = true
        errorMessage = nil

        do {
            // Start listening for events first
            startEventListener()

            // Connect based on server configuration
            if server.host == "localhost" || server.host == "127.0.0.1" || server.host.isEmpty {
                // Local connection - uses claude-code-acp
                log("connect: connecting locally to \(self.project.remotePath)")
                try await acpService.connectLocal(
                    workingDirectory: project.remotePath
                )
                log("connect: local connection established")
            } else {
                // Remote connection via SSH tunnel
                log("connect: connecting remotely to \(server.host)")

                // Fetch password from Keychain if using password auth
                var sshPassword: String? = nil
                if server.authMethod == .password {
                    sshPassword = try? await KeychainService.shared.getPassword(for: server.id)
                    log("connect: fetched password from keychain, hasPassword=\(sshPassword != nil)")
                }

                try await acpService.connectRemote(
                    sshHost: server.host,
                    sshPort: server.port,
                    sshUsername: server.username,
                    sshKeyPath: server.privateKeyPath,
                    sshPassword: sshPassword,
                    workingDirectory: project.remotePath
                )
                log("connect: remote connection established")
            }

            isConnected = true
            log("connect: isConnected=true, now creating session...")

            // Create a new session
            let sessionStart = Date()
            log("connect: creating new session at \(sessionStart)")
            let sid = try await acpService.newSession(workingDirectory: project.remotePath)
            let sessionEnd = Date()
            sessionId = sid
            log("connect: session created \(sid) in \(sessionEnd.timeIntervalSince(sessionStart))s")

            if let agentInfo = acpService.agentInfo {
                agentName = agentInfo.title
                log("connect: agent=\(agentInfo.title)")
            }

        } catch {
            log("connect: error \(error.localizedDescription)")
            errorMessage = error.localizedDescription
            isConnected = false
        }

        isLoading = false
        log("connect: completed, isConnected=\(self.isConnected)")
    }

    func sendMessage() async {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        log("sendMessage: starting, textLength=\(text.count)")
        inputText = ""
        errorMessage = nil

        // Add user message
        let userMessage = ChatDisplayMessage(role: .user, content: text)
        messages.append(userMessage)
        log("sendMessage: added user message \(userMessage.id)")

        // Add streaming placeholder for assistant
        let assistantMessage = ChatDisplayMessage(role: .assistant, content: "", isStreaming: true)
        messages.append(assistantMessage)
        streamingMessageId = assistantMessage.id
        isStreaming = true
        log("sendMessage: added assistant placeholder \(assistantMessage.id)")

        do {
            try await acpService.sendPrompt(text)
            log("sendMessage: prompt sent successfully")
        } catch {
            log("sendMessage: error \(error.localizedDescription)")
            // Remove streaming placeholder on error
            if let id = streamingMessageId {
                messages.removeAll { $0.id == id }
            }
            errorMessage = error.localizedDescription
            isStreaming = false
        }
    }

    func resumeSession(_ session: Session) async {
        isLoading = true

        do {
            let sid = try await acpService.loadSession(
                sessionId: session.id,
                workingDirectory: project.remotePath
            )
            sessionId = sid

            // Load session history
            await loadSessionHistory()

            // Update last active time
            var updatedSession = session
            updatedSession.lastActiveAt = Date()
            await sessionStore.updateSession(updatedSession)
        } catch {
            errorMessage = error.localizedDescription
        }

        isLoading = false
    }

    private func loadSessionHistory() async {
        do {
            let history = try await acpService.loadSessionHistory(cwd: project.remotePath)

            // Convert history messages to display messages
            messages = history.map { historyMsg in
                ChatDisplayMessage(
                    id: historyMsg.id,
                    role: historyMsg.role == .user ? .user : .assistant,
                    content: historyMsg.content,
                    timestamp: historyMsg.timestamp
                )
            }
        } catch {
            // History loading is optional - just log and continue
            print("[ACPChatViewModel] Failed to load session history: \(error)")
        }
    }

    func disconnect() async {
        eventTask?.cancel()
        await acpService.disconnect()
        isConnected = false
        sessionId = nil
    }

    func cancel() async {
        do {
            try await acpService.cancel()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Interrupt current streaming and send a new message
    func interruptAndSend() async {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        inputText = ""
        errorMessage = nil

        // Mark current streaming message as interrupted
        if let messageId = streamingMessageId,
           let index = messages.firstIndex(where: { $0.id == messageId }) {
            messages[index].content += "\n\n[Interrupted]"
            messages[index].isStreaming = false
        }
        streamingMessageId = nil

        // Add user message
        let userMessage = ChatDisplayMessage(role: .user, content: text)
        messages.append(userMessage)

        // Add new streaming placeholder for assistant
        let assistantMessage = ChatDisplayMessage(role: .assistant, content: "", isStreaming: true)
        messages.append(assistantMessage)
        streamingMessageId = assistantMessage.id
        isStreaming = true

        do {
            try await acpService.interruptAndPrompt(text)
        } catch {
            if let id = streamingMessageId {
                messages.removeAll { $0.id == id }
            }
            errorMessage = error.localizedDescription
            isStreaming = false
        }
    }

    // MARK: - Mode Management

    func setMode(_ modeId: String) async {
        log("setMode: attempting to set mode to \(modeId)")
        do {
            try await acpService.setMode(modeId)
            log("setMode: success")
        } catch {
            log("setMode: error - \(error)")
            // Check if this is a "method not found" error (server doesn't support mode switching)
            let errorDesc = String(describing: error)
            if errorDesc.contains("-32601") || errorDesc.contains("Method not found") {
                log("setMode: server doesn't support mode switching")
                // Don't show error - mode switching just isn't supported
            } else {
                errorMessage = error.localizedDescription
            }
        }
    }

    func cycleMode() async {
        log("cycleMode: attempting to cycle mode")
        do {
            _ = try await acpService.cycleMode()
            log("cycleMode: success")
        } catch {
            log("cycleMode: error - \(error)")
            let errorDesc = String(describing: error)
            if errorDesc.contains("-32601") || errorDesc.contains("Method not found") {
                log("cycleMode: server doesn't support mode switching")
            } else {
                errorMessage = error.localizedDescription
            }
        }
    }

    var hasMultipleModes: Bool {
        availableModes.count > 1
    }

    func toggleToolCallExpanded(_ toolCallId: String) {
        for i in messages.indices {
            if let toolIndex = messages[i].toolCalls.firstIndex(where: { $0.id == toolCallId }) {
                messages[i].toolCalls[toolIndex].isExpanded.toggle()
                break
            }
        }
    }

    // MARK: - Permission Handling

    func respondToPermission(granted: Bool) {
        log("respondToPermission: granted=\(granted)")
        pendingPermission = nil
        acpService.respondToPermission(granted: granted)
    }

    // MARK: - Private Methods

    private func startEventListener() {
        log("startEventListener: starting")
        eventTask = Task { [weak self] in
            guard let self = self else { return }

            for await event in acpService.events {
                await self.handleEvent(event)
            }
            log("startEventListener: event loop ended")
        }
    }

    private func handleEvent(_ event: ACPServiceEvent) async {
        switch event {
        case .connected:
            log("handleEvent: connected")
            isConnected = true

        case .disconnected:
            log("handleEvent: disconnected")
            isConnected = false
            sessionId = nil

        case .sessionCreated(let id), .sessionLoaded(let id):
            log("handleEvent: session created/loaded, id=\(id)")
            sessionId = id

            // Update modes from service
            availableModes = acpService.availableModes
            currentMode = acpService.currentMode
            log("handleEvent: modes=\(self.availableModes.map { $0.id }), currentMode=\(self.currentMode?.id ?? "none")")

            // Store session for later resume
            let session = Session(id: id, projectId: project.id)
            await sessionStore.addSession(session)

        case .textDelta(let delta):
            log("handleEvent: textDelta, length=\(delta.count), streamingMessageId=\(self.streamingMessageId ?? "nil")")
            if let messageId = streamingMessageId,
               let index = messages.firstIndex(where: { $0.id == messageId }) {
                messages[index].content += delta
            } else {
                log("handleEvent: textDelta received but no streaming message to append to")
            }

        case .thinking(let text):
            log("handleEvent: thinking, length=\(text.count)")
            if let messageId = streamingMessageId,
               let index = messages.firstIndex(where: { $0.id == messageId }) {
                messages[index].content += "[Thinking] \(text)\n"
            }

        case .toolUseStarted(let id, let name, let input):
            log("handleEvent: toolUseStarted, id=\(id), name=\(name), inputLength=\(input.count)")
            let toolCall = DisplayToolCall(
                id: id,
                name: name,
                input: input,
                output: nil,
                isExpanded: false,
                status: .running
            )

            // Ensure we have an assistant message to append to
            if let lastIndex = messages.indices.last, messages[lastIndex].role == .assistant {
                messages[lastIndex].toolCalls.append(toolCall)
                log("handleEvent: toolUseStarted, appended to message at index \(lastIndex), total toolCalls=\(self.messages[lastIndex].toolCalls.count)")
            } else {
                // Create a new assistant message for the tool call
                log("handleEvent: toolUseStarted, creating new assistant message for tool call")
                var assistantMessage = ChatDisplayMessage(role: .assistant, content: "", isStreaming: true)
                assistantMessage.toolCalls.append(toolCall)
                messages.append(assistantMessage)
                streamingMessageId = assistantMessage.id
                isStreaming = true
            }

        case .toolUseProgress(let id, let status, let title, let input):
            log("handleEvent: toolUseProgress, id=\(id), status=\(status), title=\(title ?? "nil")")
            // Update tool status or create if doesn't exist
            var found = false
            var foundIndex: Int? = nil
            var foundToolIndex: Int? = nil
            for i in messages.indices.reversed() {
                if let toolIndex = messages[i].toolCalls.firstIndex(where: { $0.id == id }) {
                    found = true
                    foundIndex = i
                    foundToolIndex = toolIndex
                    log("handleEvent: toolUseProgress, found tool in message at index \(i)")
                    break
                }
            }

            if found, let msgIdx = foundIndex, let toolIdx = foundToolIndex {
                // Update existing tool with new info if available
                if let title = title, messages[msgIdx].toolCalls[toolIdx].name == "Tool" {
                    messages[msgIdx].toolCalls[toolIdx].name = title
                }
                if let input = input, messages[msgIdx].toolCalls[toolIdx].input.isEmpty {
                    messages[msgIdx].toolCalls[toolIdx].input = input
                }
            } else {
                // Tool doesn't exist yet - this happens with MCP tool calls
                // Create a placeholder tool call entry
                log("handleEvent: toolUseProgress, creating tool call for \(id), title=\(title ?? "Tool")")
                let toolCall = DisplayToolCall(
                    id: id,
                    name: title ?? "Tool",
                    input: input ?? "",
                    output: nil,
                    isExpanded: false,
                    status: status == "in_progress" ? .running : .running
                )

                // Add to current assistant message or create one
                if let lastIndex = messages.indices.last, messages[lastIndex].role == .assistant {
                    messages[lastIndex].toolCalls.append(toolCall)
                } else {
                    var assistantMessage = ChatDisplayMessage(role: .assistant, content: "", isStreaming: true)
                    assistantMessage.toolCalls.append(toolCall)
                    messages.append(assistantMessage)
                    streamingMessageId = assistantMessage.id
                    isStreaming = true
                }
            }

        case .toolUseCompleted(let id, let result, let isError):
            log("handleEvent: toolUseCompleted, id=\(id), isError=\(isError), resultLength=\(result?.count ?? 0)")
            var found = false
            for i in messages.indices.reversed() {
                if let toolIndex = messages[i].toolCalls.firstIndex(where: { $0.id == id }) {
                    messages[i].toolCalls[toolIndex].output = result
                    messages[i].toolCalls[toolIndex].status = isError ? .error : .completed
                    found = true
                    log("handleEvent: toolUseCompleted, updated tool at message[\(i)].toolCalls[\(toolIndex)], newStatus=\(String(describing: self.messages[i].toolCalls[toolIndex].status))")
                    break
                }
            }
            if !found {
                log("handleEvent: toolUseCompleted, tool \(id) not found in any message!")
            }

        case .planStep(let id, let title, let status):
            log("handleEvent: planStep, id=\(id), title=\(title), status=\(status)")

        case .modeChanged(let mode):
            log("handleEvent: modeChanged, mode=\(mode)")
            // Update mode from service
            currentMode = acpService.currentMode

        case .turnComplete(let stopReason):
            log("handleEvent: turnComplete, stopReason=\(stopReason), streamingMessageId=\(self.streamingMessageId ?? "nil")")
            isStreaming = false
            if let id = streamingMessageId,
               let index = messages.firstIndex(where: { $0.id == id }) {
                messages[index].isStreaming = false
                log("handleEvent: turnComplete, set isStreaming=false on message at index \(index)")
            }
            streamingMessageId = nil

        case .error(let errMsg):
            log("handleEvent: error, message=\(errMsg)")
            errorMessage = errMsg
            isStreaming = false

            if let id = streamingMessageId {
                messages.removeAll { $0.id == id }
                log("handleEvent: error, removed streaming message \(id)")
            }
            streamingMessageId = nil

        case .permissionRequest(let id, let toolName, let input, let title):
            log("handleEvent: permissionRequest, id=\(id), tool=\(toolName)")
            pendingPermission = DisplayPermissionRequest(
                id: id,
                toolName: toolName,
                input: input,
                title: title
            )
        }
    }
}
