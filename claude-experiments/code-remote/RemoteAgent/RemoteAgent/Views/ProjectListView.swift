import SwiftUI
import SwiftData

struct ProjectListView: View {
    @Environment(\.modelContext) private var modelContext
    @Bindable var server: Server

    @State private var isAddingProject = false
    @State private var runningSessions: [ACPSSHConnection.RunningSession] = []
    @State private var isLoadingSessions = false
    @State private var sessionError: String?

    var body: some View {
        Group {
            if isAddingProject {
                // Inline file browser for adding project
                VStack(spacing: 0) {
                    HStack {
                        Button {
                            isAddingProject = false
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "chevron.left")
                                Text("Back")
                            }
                        }
                        .buttonStyle(.plain)

                        Spacer()

                        Text("Select Project Folder")
                            .font(.headline)

                        Spacer()

                        // Invisible spacer for centering
                        Text("Back").opacity(0)
                    }
                    .padding()

                    Divider()

                    RemoteFileBrowserView(server: server) { path in
                        addProject(path: path)
                        isAddingProject = false
                    }
                }
            } else {
                // Project list
                projectListContent
            }
        }
        .navigationTitle(isAddingProject ? "" : server.name)
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .toolbar {
            if !isAddingProject {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        isAddingProject = true
                    } label: {
                        Label("Add Project", systemImage: "plus")
                    }
                }
            }
        }
        .navigationDestination(for: Project.self) { project in
            SessionListView(project: project)
        }
    }

    private var projectListContent: some View {
        List {
            // Running Sessions Section
            if !runningSessions.isEmpty || isLoadingSessions {
                Section("Running Sessions") {
                    if isLoadingSessions {
                        HStack {
                            ProgressView()
                                .controlSize(.small)
                            Text("Loading...")
                                .foregroundStyle(.secondary)
                        }
                    } else {
                        ForEach(runningSessions) { session in
                            RunningSessionRow(
                                session: session,
                                project: projectForSession(session),
                                modelContainer: modelContext.container
                            )
                            .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                                Button(role: .destructive) {
                                    Task { await killSession(session.id) }
                                } label: {
                                    Label("Kill", systemImage: "xmark.circle")
                                }
                            }
                        }
                    }
                }
            }

            // Projects Section
            Section("Projects") {
                if server.projects.isEmpty {
                    Text("No projects - tap + to add")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(server.projects) { project in
                        NavigationLink(value: project) {
                            ProjectRow(project: project)
                        }
                        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                            Button(role: .destructive) {
                                deleteProject(project)
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                    }
                }
            }
        }
        .task {
            await loadRunningSessions()
        }
        .refreshable {
            await loadRunningSessions()
        }
    }

    private func loadRunningSessions() async {
        isLoadingSessions = true
        sessionError = nil

        do {
            let password = try? KeychainService.shared.getPassword(for: server.id)
            runningSessions = try await ACPSSHConnection.listRunningSessions(
                host: server.host,
                port: server.port,
                username: server.username,
                privateKeyPath: server.authMethod == .privateKey ? server.privateKeyPath : nil,
                password: password
            )
        } catch {
            sessionError = error.localizedDescription
        }

        isLoadingSessions = false
    }

    private func killSession(_ sessionId: String) async {
        do {
            let password = try? KeychainService.shared.getPassword(for: server.id)

            // SSH in and kill the process
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
            await loadRunningSessions()
        } catch {
            // Ignore errors, just refresh
            await loadRunningSessions()
        }
    }

    private func addProject(path: String) {
        let name = (path as NSString).lastPathComponent
        let project = Project(name: name, remotePath: path, server: server)
        modelContext.insert(project)
    }

    private func deleteProject(_ project: Project) {
        modelContext.delete(project)
    }

    /// Find the project that matches a running session's working directory
    private func projectForSession(_ session: ACPSSHConnection.RunningSession) -> Project? {
        guard let cwd = session.workingDirectory else { return nil }
        return server.projects.first { $0.remotePath == cwd }
    }
}

// MARK: - Running Session Row

struct RunningSessionRow: View {
    let session: ACPSSHConnection.RunningSession
    let project: Project?
    let modelContainer: ModelContainer

    @StateObject private var acpService = ACPService()

    var body: some View {
        if let project = project {
            // Session has a matching project - go straight to chat
            NavigationLink {
                ACPChatView(project: project, resumeSession: nil, modelContainer: modelContainer, acpService: acpService)
            } label: {
                sessionLabel
            }
        } else {
            // No matching project - just show info
            sessionLabel
        }
    }

    private var sessionLabel: some View {
        HStack(spacing: 12) {
            Image(systemName: "terminal.fill")
                .foregroundStyle(.green)

            VStack(alignment: .leading, spacing: 2) {
                if let project = project {
                    Text(project.name)
                        .font(.subheadline)
                }
                Text(session.id)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(project == nil ? .primary : .secondary)
            }

            Spacer()

            if project != nil {
                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Project Row

struct ProjectRow: View {
    let project: Project

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "folder.fill")
                .font(.title2)
                .foregroundStyle(.blue)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 2) {
                Text(project.name)
                    .font(.headline)

                Text(project.remotePath)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .padding(.vertical, 2)
    }
}

#Preview {
    NavigationStack {
        ProjectListView(server: Server.preview)
    }
    .modelContainer(for: [Server.self, Project.self], inMemory: true)
}
