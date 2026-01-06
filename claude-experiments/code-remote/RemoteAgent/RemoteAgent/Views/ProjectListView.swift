import SwiftUI
import SwiftData

struct ProjectListView: View {
    @Environment(\.modelContext) private var modelContext
    @Bindable var server: Server

    @State private var isAddingProject = false

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
            if server.projects.isEmpty {
                ContentUnavailableView(
                    "No Projects",
                    systemImage: "folder",
                    description: Text("Tap + to browse and select a folder")
                )
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

    private func addProject(path: String) {
        let name = (path as NSString).lastPathComponent
        let project = Project(name: name, remotePath: path, server: server)
        modelContext.insert(project)
    }

    private func deleteProject(_ project: Project) {
        modelContext.delete(project)
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
