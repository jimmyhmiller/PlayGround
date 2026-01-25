import Foundation
import SwiftUI
import SwiftData
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
    private let cacheService: SessionCacheService?
    private let usesExternalService: Bool  // True if using a pre-connected service

    private let project: Project
    private var eventTask: Task<Void, Never>?
    private var streamingMessageId: String?

    // MARK: - Init

    init(project: Project, modelContainer: ModelContainer? = nil, acpService: ACPService? = nil) {
        self.project = project
        self.sessionStore = SessionStore()
        self.cacheService = modelContainer.map { SessionCacheService(modelContainer: $0) }

        // Use pre-connected service if provided, otherwise create our own
        if let service = acpService {
            self.acpService = service
            self.usesExternalService = true
        } else {
            self.acpService = ACPService()
            self.usesExternalService = false
        }
    }

    deinit {
        eventTask?.cancel()
    }

    // MARK: - Public Methods

    /// Connect to the server. If skipNewSession is true, don't create a new session (used when resuming).
    func connect(skipNewSession: Bool = false) async {
        guard let server = project.server else {
            log("connect: no server configured for project")
            errorMessage = "No server configured for project"
            return
        }

        log("connect: starting, skipNewSession=\(skipNewSession), alreadyConnected=\(acpService.isConnected)")
        isLoading = true
        errorMessage = nil

        do {
            // Start listening for events first
            startEventListener()

            // If using a pre-connected external service, skip the connection step
            if usesExternalService && acpService.isConnected {
                log("connect: using pre-connected service, skipping connection")
            } else {
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
            }

            isConnected = true

            // Start location tracking to keep app alive when backgrounded
            LocationKeepAliveService.shared.startTracking()

            #if os(iOS)
            // Start network keep-alive to maintain connections in background
            NetworkKeepAliveService.shared.start()
            #endif

            // Start Live Activity to show status on lock screen
            ConnectionActivityManager.shared.startActivity(
                serverName: server.name,
                projectName: project.name,
                sessionId: sessionId ?? "connecting"
            )

            // Only create a new session if we're not resuming an old one
            if !skipNewSession {
                log("connect: isConnected=true, now creating session...")

                // Create a new session
                let sessionStart = Date()
                log("connect: creating new session at \(sessionStart)")
                let sid = try await acpService.newSession(workingDirectory: project.remotePath)
                let sessionEnd = Date()
                sessionId = sid
                log("connect: session created \(sid) in \(sessionEnd.timeIntervalSince(sessionStart))s")

                // Save session to local store so it appears in session list
                let newSession = Session(id: sid, projectId: project.id)
                await sessionStore.addSession(newSession)
                log("connect: saved session to store")

                // Update Live Activity with actual session ID
                ConnectionActivityManager.shared.updateConnected()
            } else {
                log("connect: isConnected=true, skipping new session (will resume)")
                ConnectionActivityManager.shared.updateConnected()
            }

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
        log("resumeSession: starting for session \(session.id)")

        do {
            // First check if the session file exists on the remote server
            let sessionExists = await acpService.sessionExistsOnServer(
                sessionId: session.id,
                cwd: project.remotePath
            )
            log("resumeSession: session exists on server = \(sessionExists)")

            if sessionExists {
                // Session exists - resume it properly
                let result = try await acpService.loadSession(
                    sessionId: session.id,
                    workingDirectory: project.remotePath
                )
                sessionId = result.sessionId
                log("resumeSession: resumed session \(result.sessionId)")

                // Load FULL history from JSONL file with preserved order and tool outputs
                log("resumeSession: loading full history for display...")
                do {
                    messages = try await acpService.loadSessionHistoryForDisplay(
                        cwd: project.remotePath,
                        sessionId: session.id
                    )
                    log("resumeSession: loaded \(messages.count) display messages")
                } catch {
                    log("resumeSession: failed to load history for display: \(error)")

                    // Fallback to old approach if new method fails
                    var history: [ACPHistoryMessage] = []

                    // Try cache first
                    if let cache = cacheService {
                        let cached = await cache.loadCachedHistory(sessionId: session.id)
                        if !cached.isEmpty {
                            log("resumeSession: loaded \(cached.count) messages from cache")
                            history = cached
                        }
                    }

                    // Last resort: use ACP's limited history
                    if history.isEmpty {
                        history = result.history
                        log("resumeSession: using ACP history with \(history.count) messages")
                    }

                    // Convert to display messages (old approach - loses order)
                    messages = history.compactMap { historyMsg -> ChatDisplayMessage? in
                        if historyMsg.role == .user && historyMsg.content.isEmpty {
                            return nil
                        }

                        var contentBlocks: [MessageContentBlock] = []

                        if !historyMsg.content.isEmpty {
                            contentBlocks.append(.text(id: UUID().uuidString, content: historyMsg.content))
                        }

                        if let toolCalls = historyMsg.toolCalls {
                            for tc in toolCalls {
                                let displayToolCall = DisplayToolCall(
                                    id: tc.id,
                                    name: tc.name,
                                    input: tc.input ?? "",
                                    output: tc.output,
                                    isExpanded: false,
                                    status: .completed
                                )
                                contentBlocks.append(.toolCall(displayToolCall))
                            }
                        }

                        guard !contentBlocks.isEmpty else { return nil }

                        return ChatDisplayMessage(
                            id: historyMsg.id,
                            role: historyMsg.role == .user ? .user : .assistant,
                            contentBlocks: contentBlocks,
                            timestamp: historyMsg.timestamp,
                            isStreaming: false
                        )
                    }
                    log("resumeSession: fallback - displaying \(messages.count) messages")
                }
            } else {
                // Session doesn't exist on server - create new session
                log("resumeSession: session not found on server, creating new session")
                let sid = try await acpService.newSession(workingDirectory: project.remotePath)
                sessionId = sid
                log("resumeSession: created new session \(sid)")

                // Delete stale session from local storage
                await sessionStore.deleteSession(id: session.id, projectId: project.id)
                if let cache = cacheService {
                    await cache.deleteCachedSession(sessionId: session.id)
                }

                // Save new session to local store
                let newSession = Session(id: sid, projectId: project.id)
                await sessionStore.addSession(newSession)
                log("resumeSession: saved new session to store")
            }
        } catch {
            log("resumeSession: error - \(error)")
            errorMessage = error.localizedDescription
        }

        isLoading = false
    }


    func disconnect() async {
        eventTask?.cancel()
        await acpService.disconnect()
        isConnected = false
        sessionId = nil

        // Stop location tracking
        LocationKeepAliveService.shared.stopTracking()

        #if os(iOS)
        // Stop network keep-alive
        NetworkKeepAliveService.shared.stop()
        #endif

        // End Live Activity
        await ConnectionActivityManager.shared.endActivity()
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
            // Append [Interrupted] to content blocks
            let interruptedText = "\n\n[Interrupted]"
            if let lastBlockIndex = messages[index].contentBlocks.indices.last,
               case .text(let id, let existingContent) = messages[index].contentBlocks[lastBlockIndex] {
                messages[index].contentBlocks[lastBlockIndex] = .text(id: id, content: existingContent + interruptedText)
            } else {
                messages[index].contentBlocks.append(.text(id: UUID().uuidString, content: interruptedText))
            }
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
            // Find the tool call in contentBlocks
            for j in messages[i].contentBlocks.indices {
                if case .toolCall(var tc) = messages[i].contentBlocks[j], tc.id == toolCallId {
                    tc.isExpanded.toggle()
                    messages[i].contentBlocks[j] = .toolCall(tc)
                    return
                }
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
                // Append to the last text block, or create a new one
                if let lastBlockIndex = messages[index].contentBlocks.indices.last,
                   case .text(let id, let existingContent) = messages[index].contentBlocks[lastBlockIndex] {
                    // Append to existing text block
                    messages[index].contentBlocks[lastBlockIndex] = .text(id: id, content: existingContent + delta)
                } else {
                    // Create a new text block
                    messages[index].contentBlocks.append(.text(id: UUID().uuidString, content: delta))
                }
                // Update Live Activity to show streaming
                ConnectionActivityManager.shared.updateStreaming(operation: "Receiving response...")
            } else {
                log("handleEvent: textDelta received but no streaming message to append to")
            }

        case .thinking(let text):
            log("handleEvent: thinking, length=\(text.count)")
            if let messageId = streamingMessageId,
               let index = messages.firstIndex(where: { $0.id == messageId }) {
                // Append thinking text to content blocks
                let thinkingText = "[Thinking] \(text)\n"
                if let lastBlockIndex = messages[index].contentBlocks.indices.last,
                   case .text(let id, let existingContent) = messages[index].contentBlocks[lastBlockIndex] {
                    messages[index].contentBlocks[lastBlockIndex] = .text(id: id, content: existingContent + thinkingText)
                } else {
                    messages[index].contentBlocks.append(.text(id: UUID().uuidString, content: thinkingText))
                }
            }

        case .toolUseStarted(let id, let name, let input):
            log("handleEvent: toolUseStarted, id=\(id), name=\(name), inputLength=\(input.count)")
            // Update Live Activity to show tool running
            ConnectionActivityManager.shared.updateToolRunning(toolName: name)

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
                messages[lastIndex].contentBlocks.append(.toolCall(toolCall))
                log("handleEvent: toolUseStarted, appended to message at index \(lastIndex), total toolCalls=\(self.messages[lastIndex].toolCalls.count)")
            } else {
                // Create a new assistant message for the tool call
                log("handleEvent: toolUseStarted, creating new assistant message for tool call")
                let assistantMessage = ChatDisplayMessage(
                    role: .assistant,
                    contentBlocks: [.toolCall(toolCall)],
                    isStreaming: true
                )
                messages.append(assistantMessage)
                streamingMessageId = assistantMessage.id
                isStreaming = true
            }

        case .toolUseProgress(let id, let status, let title, let input):
            log("handleEvent: toolUseProgress, id=\(id), status=\(status), title=\(title ?? "nil")")
            // Update tool status or create if doesn't exist
            var found = false
            var foundMsgIndex: Int? = nil
            var foundBlockIndex: Int? = nil

            for i in messages.indices.reversed() {
                for j in messages[i].contentBlocks.indices {
                    if case .toolCall(let tc) = messages[i].contentBlocks[j], tc.id == id {
                        found = true
                        foundMsgIndex = i
                        foundBlockIndex = j
                        log("handleEvent: toolUseProgress, found tool in message at index \(i)")
                        break
                    }
                }
                if found { break }
            }

            if found, let msgIdx = foundMsgIndex, let blockIdx = foundBlockIndex {
                // Update existing tool with new info if available
                if case .toolCall(var tc) = messages[msgIdx].contentBlocks[blockIdx] {
                    if let title = title, tc.name == "Tool" {
                        tc.name = title
                    }
                    if let input = input, tc.input.isEmpty {
                        tc.input = input
                    }
                    messages[msgIdx].contentBlocks[blockIdx] = .toolCall(tc)
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
                    messages[lastIndex].contentBlocks.append(.toolCall(toolCall))
                } else {
                    let assistantMessage = ChatDisplayMessage(
                        role: .assistant,
                        contentBlocks: [.toolCall(toolCall)],
                        isStreaming: true
                    )
                    messages.append(assistantMessage)
                    streamingMessageId = assistantMessage.id
                    isStreaming = true
                }
            }

        case .toolUseCompleted(let id, let result, let isError):
            log("handleEvent: toolUseCompleted, id=\(id), isError=\(isError), resultLength=\(result?.count ?? 0)")
            var found = false
            for i in messages.indices.reversed() {
                for j in messages[i].contentBlocks.indices {
                    if case .toolCall(var tc) = messages[i].contentBlocks[j], tc.id == id {
                        tc.output = result
                        tc.status = isError ? .error : .completed
                        messages[i].contentBlocks[j] = .toolCall(tc)
                        found = true
                        log("handleEvent: toolUseCompleted, updated tool at message[\(i)].contentBlocks[\(j)], newStatus=\(tc.status)")
                        break
                    }
                }
                if found { break }
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

            // Update Live Activity to show idle/ready
            ConnectionActivityManager.shared.updateIdle(messagesExchanged: messages.count)

            // Cache messages locally for offline access
            if let sessionId = sessionId, let cache = cacheService {
                Task {
                    let historyMessages = messages.map { msg in
                        ACPHistoryMessage(
                            id: msg.id,
                            role: msg.role == .user ? .user : .assistant,
                            content: msg.content,
                            timestamp: msg.timestamp
                        )
                    }
                    await cache.cacheHistory(
                        sessionId: sessionId,
                        projectId: project.id,
                        workingDirectory: project.remotePath,
                        messages: historyMessages
                    )
                    log("handleEvent: turnComplete, cached \(historyMessages.count) messages")
                }
            }

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
