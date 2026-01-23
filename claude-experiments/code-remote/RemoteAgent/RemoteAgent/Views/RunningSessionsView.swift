import SwiftUI
import SwiftData

struct RunningSessionsView: View {
    @Environment(\.dismiss) private var dismiss
    @Query(sort: \Server.name) private var servers: [Server]

    @State private var sessions: [ServerSessions] = []
    @State private var isLoading = false
    @State private var errorMessage: String?

    struct ServerSessions: Identifiable {
        let id = UUID()
        let server: Server
        var sessionIds: [String]
        var error: String?
    }

    var body: some View {
        NavigationStack {
            Group {
                if isLoading {
                    ProgressView("Checking servers...")
                } else if sessions.isEmpty {
                    ContentUnavailableView(
                        "No Running Sessions",
                        systemImage: "terminal",
                        description: Text("No ACP sessions are running on any server")
                    )
                } else {
                    List {
                        ForEach(sessions) { serverSession in
                            Section(serverSession.server.name) {
                                if let error = serverSession.error {
                                    Label(error, systemImage: "exclamationmark.triangle")
                                        .foregroundStyle(.secondary)
                                } else if serverSession.sessionIds.isEmpty {
                                    Text("No running sessions")
                                        .foregroundStyle(.secondary)
                                } else {
                                    ForEach(serverSession.sessionIds, id: \.self) { sessionId in
                                        RunningSessionRow(
                                            sessionId: sessionId,
                                            server: serverSession.server,
                                            onKill: { await killSession(sessionId, on: serverSession.server) }
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Running Sessions")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        Task { await refresh() }
                    } label: {
                        Label("Refresh", systemImage: "arrow.clockwise")
                    }
                    .disabled(isLoading)
                }
            }
            .task {
                await refresh()
            }
        }
    }

    private func refresh() async {
        isLoading = true
        sessions = []

        for server in servers {
            let serverSession = await checkServer(server)
            sessions.append(serverSession)
        }

        isLoading = false
    }

    private func checkServer(_ server: Server) async -> ServerSessions {
        do {
            let password = try? KeychainService.shared.getPassword(for: server.id)
            let sessionIds = try await ACPSSHConnection.listRunningSessions(
                host: server.host,
                port: server.port,
                username: server.username,
                privateKeyPath: server.authMethod == .privateKey ? server.privateKeyPath : nil,
                password: password
            )
            return ServerSessions(server: server, sessionIds: sessionIds)
        } catch {
            return ServerSessions(server: server, sessionIds: [], error: error.localizedDescription)
        }
    }

    private func killSession(_ sessionId: String, on server: Server) async {
        do {
            let password = try? KeychainService.shared.getPassword(for: server.id)

            // Create a temporary connection just to kill the session
            let connection = ACPSSHConnection(existingSessionId: sessionId)
            try await connection.connect(
                host: server.host,
                port: server.port,
                username: server.username,
                privateKeyPath: server.authMethod == .privateKey ? server.privateKeyPath : nil,
                password: password,
                workingDirectory: "/tmp",
                acpPath: "claude-code-acp"
            )
            await connection.close() // This terminates the session

            // Refresh the list
            await refresh()
        } catch {
            // Ignore errors during kill
        }
    }
}

struct RunningSessionRow: View {
    let sessionId: String
    let server: Server
    let onKill: () async -> Void

    @State private var isKilling = false

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(sessionId)
                    .font(.system(.body, design: .monospaced))

                Text("Session ID")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button(role: .destructive) {
                isKilling = true
                Task {
                    await onKill()
                    isKilling = false
                }
            } label: {
                if isKilling {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Label("Kill", systemImage: "xmark.circle")
                        .labelStyle(.iconOnly)
                }
            }
            .buttonStyle(.borderless)
            .disabled(isKilling)
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    RunningSessionsView()
}
