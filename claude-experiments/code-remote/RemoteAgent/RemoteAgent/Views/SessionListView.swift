import SwiftUI

struct SessionListView: View {
    let project: Project

    @State private var sessions: [Session] = []
    @State private var navigateToChat = false
    @State private var selectedSession: Session?

    private let sessionStore = SessionStore()

    var body: some View {
        List {
            Section {
                NavigationLink {
                    ACPChatView(project: project, resumeSession: nil)
                } label: {
                    HStack(spacing: 12) {
                        Image(systemName: "plus.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.green)

                        VStack(alignment: .leading, spacing: 2) {
                            Text("New Session")
                                .font(.headline)
                            Text("Start a fresh conversation")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }
            }

            if !sessions.isEmpty {
                Section("Recent Sessions") {
                    ForEach(sessions) { session in
                        NavigationLink {
                            ACPChatView(project: project, resumeSession: session)
                        } label: {
                            SessionRow(session: session)
                        }
                        .swipeActions(edge: .trailing) {
                            Button(role: .destructive) {
                                deleteSession(session)
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle(project.name)
        .task {
            await loadSessions()
        }
        .refreshable {
            await loadSessions()
        }
    }

    private func loadSessions() async {
        sessions = await sessionStore.sessions(for: project.id)
    }

    private func deleteSession(_ session: Session) {
        Task {
            await sessionStore.deleteSession(id: session.id, projectId: project.id)
            await loadSessions()
        }
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
                Text(session.id.prefix(8) + "...")
                    .font(.headline)

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
