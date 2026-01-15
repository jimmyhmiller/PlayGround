import SwiftUI
import SwiftData

struct SessionListView: View {
    @Environment(\.modelContext) private var modelContext
    let project: Project

    @State private var sessions: [Session] = []
    @State private var isLoading = false  // Initial load (no cached sessions)
    @State private var isRefreshing = false  // Background refresh (has cached sessions)
    @State private var errorMessage: String?

    // Pre-connect to ACP when view appears
    @StateObject private var acpService = ACPService()
    @State private var isConnecting = false
    @State private var connectionError: String?

    var body: some View {
        List {
            Section {
                NavigationLink {
                    ACPChatView(project: project, resumeSession: nil, modelContainer: modelContext.container, acpService: acpService)
                } label: {
                    HStack(spacing: 12) {
                        Image(systemName: "plus.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.green)

                        VStack(alignment: .leading, spacing: 2) {
                            Text("New Session")
                                .font(.headline)
                            HStack(spacing: 4) {
                                if isConnecting {
                                    ProgressView()
                                        .scaleEffect(0.6)
                                    Text("Connecting...")
                                } else if acpService.isConnected {
                                    Circle()
                                        .fill(.green)
                                        .frame(width: 6, height: 6)
                                    Text("Ready")
                                } else {
                                    Text("Start a fresh conversation")
                                }
                            }
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }
            }

            if isLoading && sessions.isEmpty {
                // Show blocking loading indicator only on initial load
                Section {
                    HStack {
                        ProgressView()
                        Text("Loading sessions from server...")
                            .foregroundStyle(.secondary)
                    }
                }
            } else if let error = errorMessage, sessions.isEmpty {
                Section {
                    Text(error)
                        .foregroundStyle(.red)
                }
            } else if !sessions.isEmpty {
                Section {
                    ForEach(sessions) { session in
                        NavigationLink {
                            ACPChatView(project: project, resumeSession: session, modelContainer: modelContext.container, acpService: acpService)
                        } label: {
                            SessionRow(session: session)
                        }
                    }
                } header: {
                    HStack {
                        Text("Recent Sessions (\(sessions.count))")
                        if isRefreshing {
                            ProgressView()
                                .scaleEffect(0.7)
                        }
                    }
                }
            }
        }
        .navigationTitle(project.name)
        .task {
            // Load sessions and pre-connect to ACP in parallel
            await withTaskGroup(of: Void.self) { group in
                group.addTask { await self.loadSessions() }
                group.addTask { await self.preConnectToACP() }
            }
        }
        .refreshable {
            await loadSessions()
        }
        .onDisappear {
            // Don't disconnect - let ACPChatView handle the connection
        }
    }

    private func preConnectToACP() async {
        guard !acpService.isConnected else { return }
        guard let server = project.server else { return }

        isConnecting = true
        connectionError = nil

        do {
            if server.host == "localhost" || server.host == "127.0.0.1" || server.host.isEmpty {
                // Check if ACP is installed locally first
                guard ACPService.isClaudeCodeACPInstalled() else {
                    isConnecting = false
                    return
                }
                try await acpService.connectLocal(workingDirectory: project.remotePath)
            } else {
                // Check if ACP is installed on remote first
                let isInstalled = await ACPService.isClaudeCodeACPInstalledRemote(server: server)
                guard isInstalled else {
                    isConnecting = false
                    return
                }

                var sshPassword: String? = nil
                if server.authMethod == .password {
                    sshPassword = try? await KeychainService.shared.getPassword(for: server.id)
                }

                try await acpService.connectRemote(
                    sshHost: server.host,
                    sshPort: server.port,
                    sshUsername: server.username,
                    sshKeyPath: server.privateKeyPath,
                    sshPassword: sshPassword,
                    workingDirectory: project.remotePath
                )
            }
        } catch {
            connectionError = error.localizedDescription
        }

        isConnecting = false
    }

    private func loadSessions() async {
        guard let server = project.server else {
            errorMessage = "No server configured"
            return
        }

        // Use different loading state depending on whether we have cached sessions
        if sessions.isEmpty {
            isLoading = true
        } else {
            isRefreshing = true
        }
        errorMessage = nil

        // Use SSHService directly to list sessions - don't start claude-code-acp
        let sshService = SSHService()

        do {
            var sshPassword: String? = nil
            if server.authMethod == .password {
                sshPassword = try? await KeychainService.shared.getPassword(for: server.id)
            }

            try await sshService.connect(to: server, password: sshPassword)

            // List session files directly via SSH
            let encodedCwd = project.remotePath.replacingOccurrences(of: "/", with: "-")
            let remotePath = "~/.claude/projects/\(encodedCwd)"
            let command = "bash -c 'for f in \(remotePath)/*.jsonl; do [ -f \"$f\" ] && stat --format=\"%Y %s %n\" \"$f\" 2>/dev/null; done | sort -rn'"

            let output = try await sshService.executeCommand(command)

            var loadedSessions: [Session] = []
            let lines = output.components(separatedBy: "\n").filter { !$0.isEmpty }

            for line in lines {
                let parts = line.components(separatedBy: " ")
                guard parts.count >= 3 else { continue }

                guard let epoch = Double(parts[0]) else { continue }
                guard let size = Int(parts[1]), size > 0 else { continue } // Skip empty files

                let fullPath = parts.dropFirst(2).joined(separator: " ")
                guard fullPath.hasSuffix(".jsonl") else { continue }

                let filename = (fullPath as NSString).lastPathComponent
                let sessionId = filename.replacingOccurrences(of: ".jsonl", with: "")

                // Skip internal ACP agent sessions (created by claude-code-acp process itself)
                guard !sessionId.hasPrefix("agent-") else { continue }

                let createdAt = Date(timeIntervalSince1970: epoch)
                loadedSessions.append(Session(id: sessionId, projectId: project.id, createdAt: createdAt))
            }

            // Fetch titles for all sessions in parallel using a single SSH command
            if !loadedSessions.isEmpty {
                let sessionIds = loadedSessions.map { $0.id }
                let titles = await fetchSessionTitles(sshService: sshService, remotePath: remotePath, sessionIds: sessionIds)

                // Update sessions with titles
                for i in loadedSessions.indices {
                    if let title = titles[loadedSessions[i].id] {
                        loadedSessions[i].title = title
                    }
                }
            }

            await sshService.disconnect()

            sessions = loadedSessions
        } catch {
            // Only show error if we don't have cached sessions
            if sessions.isEmpty {
                errorMessage = "Failed to load sessions: \(error.localizedDescription)"
            }
        }

        isLoading = false
        isRefreshing = false
    }

    /// Fetch the first user message from each session file as the title
    private func fetchSessionTitles(sshService: SSHService, remotePath: String, sessionIds: [String]) async -> [String: String] {
        var titles: [String: String] = [:]

        // Batch all title fetches into a single command for efficiency
        // Read first 20 lines of each file and extract the first user message
        let sessionFilePaths = sessionIds.map { "\(remotePath)/\($0).jsonl" }.joined(separator: " ")
        let command = """
        bash -c 'for f in \(sessionFilePaths); do
            if [ -f "$f" ]; then
                filename=$(basename "$f" .jsonl)
                # Look for first user message in first 20 lines
                title=$(head -n 20 "$f" | grep -m1 \'"role":\\s*"user"\' | sed -n "s/.*\\"content\\":\\s*\\"\\([^\\"]*\\)\\".*/\\1/p" | head -c 80)
                if [ -n "$title" ]; then
                    echo "$filename|||$title"
                fi
            fi
        done'
        """

        do {
            let output = try await sshService.executeCommand(command)
            let lines = output.components(separatedBy: "\n").filter { !$0.isEmpty }

            for line in lines {
                let parts = line.components(separatedBy: "|||")
                if parts.count >= 2 {
                    let sessionId = parts[0]
                    let title = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
                    if !title.isEmpty {
                        titles[sessionId] = title
                    }
                }
            }
        } catch {
            // Silent failure - we'll just show IDs for sessions without titles
        }

        return titles
    }
}

// MARK: - Session Row

struct SessionRow: View {
    let session: Session

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.title2)
                .foregroundStyle(.blue)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 2) {
                Text(session.title ?? session.id.prefix(8) + "...")
                    .font(.headline)
                    .lineLimit(1)

                Text(session.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}

#Preview {
    NavigationStack {
        SessionListView(project: Project.preview)
    }
}
