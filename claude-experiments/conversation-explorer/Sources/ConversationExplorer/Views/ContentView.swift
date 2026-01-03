import SwiftUI

struct ContentView: View {
    @StateObject private var dataService = DataService()
    @State private var selectedProject: String?
    @State private var selectedEntry: HistoryEntry?
    @State private var searchQuery = ""
    @State private var isSearching = false

    var filteredProjects: [String] {
        if searchQuery.isEmpty {
            return dataService.uniqueProjects
        }
        let query = searchQuery.lowercased()
        // Show projects that match OR have matching conversations
        return dataService.uniqueProjects.filter { project in
            let projectName = URL(fileURLWithPath: project).lastPathComponent.lowercased()
            if projectName.contains(query) { return true }
            // Check if any conversation in this project matches
            return dataService.entriesForProject(project).contains {
                $0.display.lowercased().contains(query)
            }
        }
    }

    func filteredEntries(for project: String) -> [HistoryEntry] {
        let entries = dataService.entriesForProject(project)
        if searchQuery.isEmpty { return entries }
        let query = searchQuery.lowercased()
        return entries.filter { $0.display.lowercased().contains(query) }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Global search bar
            HStack(spacing: 12) {
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundColor(.secondary)
                    TextField("Search all conversations...", text: $searchQuery)
                        .textFieldStyle(.plain)
                    if !searchQuery.isEmpty {
                        Button(action: { searchQuery = "" }) {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(8)
                .background(Color(nsColor: .textBackgroundColor))
                .cornerRadius(8)
            }
            .padding(12)
            .background(Color(nsColor: .windowBackgroundColor))

            Divider()

            // Main content
            HStack(spacing: 0) {
                // Projects column
                ProjectListView(
                    projects: filteredProjects,
                    dataService: dataService,
                    selectedProject: $selectedProject,
                    searchQuery: searchQuery
                )
                .frame(width: 240)

                Divider()

                // Sessions column
                SessionListView(
                    entries: selectedProject.map { filteredEntries(for: $0) } ?? [],
                    selectedProject: selectedProject,
                    selectedEntry: $selectedEntry
                )
                .frame(width: 320)

                Divider()

                // Detail column
                if let entry = selectedEntry {
                    ConversationDetailView(entry: entry, dataService: dataService)
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "bubble.left.and.bubble.right")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary.opacity(0.4))
                        Text("Select a conversation")
                            .font(.title3)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color(nsColor: .textBackgroundColor))
                }
            }
        }
        .task {
            await dataService.loadHistory()
        }
        .onChange(of: selectedProject) { _, _ in
            selectedEntry = nil
        }
    }
}
