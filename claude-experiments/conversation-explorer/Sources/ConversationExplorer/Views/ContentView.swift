import SwiftUI

struct ContentView: View {
    @StateObject private var dataService = DataService()
    @State private var selectedProject: String?
    @State private var selectedEntry: HistoryEntry?
    @State private var searchQuery = ""
    @State private var isSearching = false
    @State private var fullTextResults: [HistoryEntry]? = nil
    @State private var searchTask: Task<Void, Never>? = nil

    var filteredProjects: [String] {
        if searchQuery.isEmpty {
            return dataService.uniqueProjects
        }

        // If we have full-text results, show projects containing those results
        if let results = fullTextResults {
            let matchingProjects = Set(results.map { $0.project })
            return dataService.uniqueProjects.filter { matchingProjects.contains($0) }
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
        // If we have full-text results, filter to only those results for this project
        if let results = fullTextResults {
            return results.filter { $0.project == project }
                .sorted { $0.timestamp > $1.timestamp }
        }

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
                    if isSearching {
                        ProgressView()
                            .scaleEffect(0.7)
                            .frame(width: 16, height: 16)
                    } else {
                        Image(systemName: "magnifyingglass")
                            .foregroundColor(.secondary)
                    }
                    TextField("Search all conversations...", text: $searchQuery)
                        .textFieldStyle(.plain)
                    if !searchQuery.isEmpty {
                        if let results = fullTextResults {
                            Text("\(results.count) results")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
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
                    searchQuery: searchQuery,
                    fullTextResults: fullTextResults
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
        .onChange(of: searchQuery) { _, newValue in
            // Cancel previous search
            searchTask?.cancel()

            if newValue.isEmpty {
                fullTextResults = nil
                isSearching = false
                return
            }

            // Start new streaming search
            fullTextResults = []
            isSearching = true

            searchTask = Task {
                // Small debounce
                try? await Task.sleep(for: .milliseconds(150))
                guard !Task.isCancelled else { return }

                await dataService.searchFullText(query: newValue) { entry in
                    guard !Task.isCancelled else { return }
                    Task { @MainActor in
                        guard !Task.isCancelled else { return }
                        if fullTextResults != nil {
                            fullTextResults?.append(entry)
                        }
                    }
                }

                guard !Task.isCancelled else { return }
                await MainActor.run {
                    isSearching = false
                }
            }
        }
    }
}
