import SwiftUI
import ACPLib

// Helper class to collect output from non-isolated callbacks
@MainActor
class InstallOutputCollector: ObservableObject {
    @Published var output: String = ""

    nonisolated func append(_ text: String) {
        Task { @MainActor in
            self.output += text
        }
    }

    func clear() {
        output = ""
    }
}

struct ACPChatView: View {
    @StateObject private var viewModel: ACPChatViewModel
    @StateObject private var outputCollector = InstallOutputCollector()
    @State private var showInstallPrompt = false
    @State private var isInstalling = false
    @State private var isRemoteServer = false
    @State private var isCheckingRemote = false

    let project: Project
    let resumeSession: Session?

    private var isRemote: Bool {
        guard let server = project.server else { return false }
        return !server.host.isEmpty && server.host != "localhost" && server.host != "127.0.0.1"
    }

    init(project: Project, resumeSession: Session?) {
        self.project = project
        self.resumeSession = resumeSession
        _viewModel = StateObject(wrappedValue: ACPChatViewModel(project: project))
    }

    var body: some View {
        VStack(spacing: 0) {
            // Connection status bar
            ACPConnectionStatusBar(
                isConnected: viewModel.isConnected,
                sessionId: viewModel.sessionId,
                agentName: viewModel.agentName ?? "Claude Code",
                model: viewModel.model,
                currentMode: viewModel.currentMode,
                availableModes: viewModel.availableModes,
                onModeSelected: { modeId in
                    Task { await viewModel.setMode(modeId) }
                }
            )

            if isCheckingRemote {
                // Checking remote server
                VStack(spacing: 16) {
                    Spacer()
                    ProgressView()
                        .scaleEffect(1.5)
                    Text("Checking remote server...")
                        .font(.headline)
                    if let server = project.server {
                        Text("\(server.username)@\(server.host)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                }
            } else if showInstallPrompt {
                // Install prompt
                InstallACPPromptView(
                    isInstalling: $isInstalling,
                    isRemote: isRemote,
                    serverInfo: project.server.map { "\($0.username)@\($0.host)" },
                    installOutput: outputCollector.output,
                    onInstall: {
                        Task { await installClaudeCodeACP() }
                    },
                    onRetry: {
                        showInstallPrompt = false
                        outputCollector.clear()
                        Task { await checkAndConnect() }
                    }
                )
            } else {
                // Messages
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 16) {
                            ForEach(viewModel.messages) { message in
                                MessageView(
                                    message: message,
                                    onToggleToolCall: viewModel.toggleToolCallExpanded
                                )
                                .id(message.id)
                            }
                        }
                        .padding()
                    }
                    .onChange(of: viewModel.messages.count) { _, _ in
                        if let lastId = viewModel.messages.last?.id {
                            withAnimation {
                                proxy.scrollTo(lastId, anchor: .bottom)
                            }
                        }
                    }
                }

                // Permission request banner
                if let permission = viewModel.pendingPermission {
                    PermissionRequestView(
                        request: permission,
                        onAllow: { viewModel.respondToPermission(granted: true) },
                        onDeny: { viewModel.respondToPermission(granted: false) }
                    )
                }

                // Error banner
                if let error = viewModel.errorMessage {
                    ErrorBanner(message: error) {
                        viewModel.errorMessage = nil
                    }
                }

                // Input bar
                InputBar(
                    text: $viewModel.inputText,
                    isLoading: viewModel.isLoading,
                    isStreaming: viewModel.isStreaming,
                    isConnected: viewModel.isConnected,
                    onSend: {
                        Task { await viewModel.sendMessage() }
                    },
                    onCancel: {
                        Task { await viewModel.cancel() }
                    },
                    onInterruptAndSend: {
                        Task { await viewModel.interruptAndSend() }
                    }
                )
            }
        }
        .navigationTitle(project.name)
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                if let sessionId = viewModel.sessionId {
                    Text(sessionId.prefix(8) + "...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .task {
            // Use task cancellation for cleanup - SwiftUI cancels this when view actually goes away
            await withTaskCancellationHandler {
                await checkAndConnect()
                // Keep task alive while connected to detect cancellation
                while !Task.isCancelled && viewModel.isConnected {
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                }
            } onCancel: {
                Task { await viewModel.disconnect() }
            }
        }
    }

    private func checkAndConnect() async {
        if isRemote {
            // Remote server - check via SSH
            guard let server = project.server else {
                viewModel.errorMessage = "No server configured"
                return
            }

            isCheckingRemote = true

            let isInstalled = await ACPService.isClaudeCodeACPInstalledRemote(server: server)

            isCheckingRemote = false

            if !isInstalled {
                isRemoteServer = true
                showInstallPrompt = true
                return
            }
        } else {
            // Local - check locally
            if !ACPService.isClaudeCodeACPInstalled() {
                isRemoteServer = false
                showInstallPrompt = true
                return
            }
        }

        await viewModel.connect()

        if viewModel.isConnected, let session = resumeSession {
            await viewModel.resumeSession(session)
        }
    }

    private func installClaudeCodeACP() async {
        isInstalling = true
        outputCollector.clear()

        if isRemote {
            // Remote installation via SSH
            guard let server = project.server else {
                viewModel.errorMessage = "No server configured"
                isInstalling = false
                return
            }

            let result = await ACPService.installClaudeCodeACPRemote(
                server: server,
                onOutput: { [outputCollector] output in
                    outputCollector.append(output)
                }
            )

            switch result {
            case .success:
                showInstallPrompt = false
                outputCollector.clear()
                await viewModel.connect()
            case .failure(let error):
                viewModel.errorMessage = "Remote installation failed: \(error.localizedDescription)"
            }
        } else {
            // Local installation - use ACPService helper
            let result = await ACPService.installClaudeCodeACPLocal { [outputCollector] output in
                outputCollector.append(output)
            }

            switch result {
            case .success:
                showInstallPrompt = false
                outputCollector.clear()
                await viewModel.connect()
            case .failure(let error):
                viewModel.errorMessage = "Installation failed: \(error.localizedDescription)"
            }
        }

        isInstalling = false
    }
}

// MARK: - Install ACP Prompt View

struct InstallACPPromptView: View {
    @Binding var isInstalling: Bool
    let isRemote: Bool
    let serverInfo: String?
    let installOutput: String
    let onInstall: () -> Void
    let onRetry: () -> Void

    @State private var showOutput = true

    private var installCommand: String {
        if isRemote, let server = serverInfo {
            return "ssh \(server) 'npm install -g @zed-industries/claude-code-acp'"
        }
        return "npm install -g @zed-industries/claude-code-acp"
    }

    var body: some View {
        VStack(spacing: 16) {
            Spacer()

            Image(systemName: isRemote ? "server.rack" : "shippingbox")
                .font(.system(size: 60))
                .foregroundStyle(.secondary)

            Text("Claude Code ACP Required")
                .font(.title2)
                .fontWeight(.semibold)

            if isRemote, let server = serverInfo {
                Text("on \(server)")
                    .font(.headline)
                    .foregroundStyle(.blue)
            }

            Text(isRemote
                 ? "The claude-code-acp package needs to be installed on the remote server to enable ACP communication."
                 : "This app uses the Agent Client Protocol (ACP) to communicate with Claude Code. The claude-code-acp package needs to be installed.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            VStack(spacing: 12) {
                Button {
                    onInstall()
                } label: {
                    if isInstalling {
                        HStack {
                            ProgressView()
                                .scaleEffect(0.8)
                            Text(isRemote ? "Installing on server..." : "Installing...")
                        }
                        .frame(maxWidth: .infinity)
                    } else {
                        Text(isRemote ? "Install on Remote Server" : "Install claude-code-acp")
                            .frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isInstalling)

                Text("or run manually:")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                HStack {
                    Text(installCommand)
                        .font(.system(.caption, design: .monospaced))
                        .padding(8)
                        .background(Color.secondary.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .lineLimit(1)
                        .truncationMode(.middle)

                    Button {
                        #if os(macOS)
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(installCommand, forType: .string)
                        #endif
                    } label: {
                        Image(systemName: "doc.on.doc")
                    }
                    .buttonStyle(.plain)
                }

                Button("I've installed it - Retry") {
                    onRetry()
                }
                .buttonStyle(.bordered)
                .padding(.top, 8)
                .disabled(isInstalling)
            }
            .padding(.horizontal, 40)

            Spacer()

            // Collapsible output panel at the bottom
            if isInstalling || !installOutput.isEmpty {
                InstallOutputPanel(
                    output: installOutput,
                    isExpanded: $showOutput,
                    isInstalling: isInstalling
                )
            }
        }
        .padding()
    }
}

// MARK: - Install Output Panel

struct InstallOutputPanel: View {
    let output: String
    @Binding var isExpanded: Bool
    let isInstalling: Bool

    var body: some View {
        VStack(spacing: 0) {
            // Header - always visible
            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 16)

                    Image(systemName: "terminal")
                        .foregroundStyle(.green)

                    Text("Installation Output")
                        .font(.headline)
                        .foregroundStyle(.primary)

                    Spacer()

                    if isInstalling {
                        ProgressView()
                            .scaleEffect(0.7)
                    }

                    Text("\(output.components(separatedBy: "\n").count) lines")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                #if os(iOS)
                .background(Color(.systemBackground))
                #else
                .background(Color(nsColor: .windowBackgroundColor))
                #endif
            }
            .buttonStyle(.plain)

            // Expandable content
            if isExpanded {
                Divider()

                ScrollViewReader { proxy in
                    ScrollView {
                        Text(output.isEmpty ? "Waiting for output..." : output)
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(12)
                            .id("bottom")
                    }
                    .frame(height: 200)
                    .background(Color.black.opacity(0.9))
                    .foregroundStyle(output.isEmpty ? .gray : .green)
                    .onChange(of: output) { _, _ in
                        withAnimation {
                            proxy.scrollTo("bottom", anchor: .bottom)
                        }
                    }
                }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
        )
    }
}

// MARK: - Permission Request View

struct PermissionRequestView: View {
    let request: DisplayPermissionRequest
    let onAllow: () -> Void
    let onDeny: () -> Void

    private var toolLabel: String {
        let name = request.toolName.lowercased()
        if name.contains("bash") || name.contains("terminal") {
            return "Bash"
        } else if name.contains("read") {
            return "Read"
        } else if name.contains("write") {
            return "Write"
        } else if name.contains("edit") {
            return "Edit"
        } else if name.contains("glob") {
            return "Glob"
        } else if name.contains("grep") {
            return "Grep"
        } else if name.contains("task") {
            return "Task"
        }
        return request.toolName
    }

    private var displayText: String {
        guard let input = request.input, !input.isEmpty,
              let data = input.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return ""
        }

        if let command = json["command"] as? String {
            return command
        }
        if let path = json["file_path"] as? String {
            return (path as NSString).lastPathComponent
        }
        if let pattern = json["pattern"] as? String {
            return pattern
        }
        if let desc = json["description"] as? String {
            return desc
        }
        if let prompt = json["prompt"] as? String {
            return String(prompt.prefix(50))
        }
        return ""
    }

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: "exclamationmark.shield.fill")
                    .font(.title2)
                    .foregroundStyle(.orange)

                Text("Permission Required")
                    .font(.headline)

                Spacer()
            }

            // Simple tool display like in chat
            HStack(spacing: 6) {
                Text(toolLabel)
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundStyle(.secondary)

                if !displayText.isEmpty {
                    Text(displayText)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.primary)
                        .lineLimit(2)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 12) {
                Button {
                    onDeny()
                } label: {
                    Text("Deny")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(.red)

                Button {
                    onAllow()
                } label: {
                    Text("Allow")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.orange.opacity(0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(.orange.opacity(0.3), lineWidth: 1)
                )
        )
        .padding(.horizontal)
    }
}

// MARK: - ACP Connection Status Bar

struct ACPConnectionStatusBar: View {
    let isConnected: Bool
    let sessionId: String?
    let agentName: String
    let model: String?
    let currentMode: ACPMode?
    let availableModes: [ACPMode]
    let onModeSelected: (String) -> Void

    var body: some View {
        HStack {
            Circle()
                .fill(isConnected ? Color.green : Color.red)
                .frame(width: 8, height: 8)

            Text(isConnected ? "Connected via ACP" : "Disconnected")
                .font(.caption)

            Spacer()

            // Mode selector (only show if multiple modes available)
            if availableModes.count > 1 {
                Menu {
                    ForEach(availableModes) { mode in
                        Button {
                            onModeSelected(mode.id)
                        } label: {
                            HStack {
                                Text(mode.name)
                                if mode.id == currentMode?.id {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text(currentMode?.name ?? "Mode")
                            .font(.caption2)
                        Image(systemName: "chevron.down")
                            .font(.caption2)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(.orange.opacity(0.2))
                    .clipShape(Capsule())
                }
                .menuStyle(.borderlessButton)
            } else if let mode = currentMode {
                // Single mode - just display it
                Text(mode.name)
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.orange.opacity(0.2))
                    .clipShape(Capsule())
            }

            Text(agentName)
                .font(.caption2)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.blue.opacity(0.2))
                .clipShape(Capsule())

            if let model = model {
                Text(model)
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.secondary.opacity(0.2))
                    .clipShape(Capsule())
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }
}

#Preview {
    NavigationStack {
        ACPChatView(
            project: Project.preview,
            resumeSession: nil
        )
    }
}
