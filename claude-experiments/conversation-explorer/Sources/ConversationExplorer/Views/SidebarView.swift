import SwiftUI

struct ProjectListView: View {
    let projects: [String]
    @ObservedObject var dataService: DataService
    @Binding var selectedProject: String?
    let searchQuery: String

    var body: some View {
        VStack(spacing: 0) {
            if dataService.isLoading {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Loading...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if projects.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "folder")
                        .font(.system(size: 32))
                        .foregroundColor(.secondary.opacity(0.4))
                    Text("No projects found")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 1) {
                        ForEach(projects, id: \.self) { project in
                            ProjectRow(
                                project: project,
                                count: countForProject(project),
                                isSelected: selectedProject == project
                            )
                            .onTapGesture {
                                selectedProject = project
                            }
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }

    func countForProject(_ project: String) -> Int {
        if searchQuery.isEmpty {
            return dataService.entriesForProject(project).count
        }
        let query = searchQuery.lowercased()
        return dataService.entriesForProject(project).filter {
            $0.display.lowercased().contains(query)
        }.count
    }
}

struct ProjectRow: View {
    let project: String
    let count: Int
    let isSelected: Bool

    var projectName: String {
        URL(fileURLWithPath: project).lastPathComponent
    }

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "folder.fill")
                .font(.system(size: 14))
                .foregroundColor(isSelected ? .white : .blue)

            Text(projectName)
                .font(.system(size: 13))
                .foregroundColor(isSelected ? .white : .primary)
                .lineLimit(1)

            Spacer()

            Text("\(count)")
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(isSelected ? .white.opacity(0.8) : .secondary)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(isSelected ? Color.white.opacity(0.2) : Color.secondary.opacity(0.15))
                .cornerRadius(4)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .contentShape(Rectangle())
        .background(isSelected ? Color.accentColor : Color.clear)
        .cornerRadius(6)
        .padding(.horizontal, 4)
    }
}

struct SessionListView: View {
    let entries: [HistoryEntry]
    let selectedProject: String?
    @Binding var selectedEntry: HistoryEntry?

    var projectName: String {
        guard let project = selectedProject else { return "" }
        return URL(fileURLWithPath: project).lastPathComponent
    }

    var body: some View {
        VStack(spacing: 0) {
            if selectedProject == nil {
                VStack(spacing: 12) {
                    Image(systemName: "arrow.left.circle")
                        .font(.system(size: 32))
                        .foregroundColor(.secondary.opacity(0.4))
                    Text("Select a project")
                        .font(.callout)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if entries.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "doc.text")
                        .font(.system(size: 32))
                        .foregroundColor(.secondary.opacity(0.4))
                    Text("No conversations")
                        .font(.callout)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 1) {
                        ForEach(entries) { entry in
                            SessionRow(
                                entry: entry,
                                isSelected: selectedEntry?.id == entry.id
                            )
                            .onTapGesture {
                                selectedEntry = entry
                            }
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }
}

struct SessionRow: View {
    let entry: HistoryEntry
    let isSelected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(entry.display)
                .font(.system(size: 13))
                .foregroundColor(isSelected ? .white : .primary)
                .lineLimit(2)

            Text(formatDate(entry.date))
                .font(.system(size: 11))
                .foregroundColor(isSelected ? .white.opacity(0.7) : .secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .contentShape(Rectangle())
        .background(isSelected ? Color.accentColor : Color.clear)
        .cornerRadius(6)
        .padding(.horizontal, 4)
    }

    func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        let calendar = Calendar.current

        if calendar.isDateInToday(date) {
            formatter.dateFormat = "'Today at' h:mm a"
        } else if calendar.isDateInYesterday(date) {
            formatter.dateFormat = "'Yesterday at' h:mm a"
        } else if calendar.isDate(date, equalTo: Date(), toGranularity: .weekOfYear) {
            formatter.dateFormat = "EEEE 'at' h:mm a"
        } else {
            formatter.dateFormat = "MMM d, yyyy"
        }
        return formatter.string(from: date)
    }
}
