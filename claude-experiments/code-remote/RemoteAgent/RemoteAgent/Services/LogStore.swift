import Foundation
import SwiftUI

/// A simple in-memory log store for debugging
@MainActor
class LogStore: ObservableObject {
    static let shared = LogStore()

    struct LogEntry: Identifiable {
        let id = UUID()
        let timestamp: Date
        let category: String
        let message: String
    }

    @Published private(set) var entries: [LogEntry] = []
    private let maxEntries = 500
    private let logFileURL: URL?
    private let formatter = ISO8601DateFormatter()

    private init() {
        // Create log file in Documents directory (accessible via devicectl)
        if let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            logFileURL = docs.appendingPathComponent("remote_agent.log")
            // Clear previous log file on app start
            try? "".write(to: logFileURL!, atomically: true, encoding: .utf8)
        } else {
            logFileURL = nil
        }
    }

    func log(_ message: String, category: String = "App") {
        let entry = LogEntry(timestamp: Date(), category: category, message: message)
        entries.append(entry)

        // Trim old entries
        if entries.count > maxEntries {
            entries.removeFirst(entries.count - maxEntries)
        }

        // Format the log line
        let timestamp = formatter.string(from: entry.timestamp)
        let logLine = "[\(timestamp)] [\(category)] \(message)\n"

        // Print to stderr for Xcode console
        fputs(logLine, stderr)

        // Append to log file for remote access
        if let url = logFileURL {
            if let data = logLine.data(using: .utf8) {
                if let handle = try? FileHandle(forWritingTo: url) {
                    handle.seekToEndOfFile()
                    handle.write(data)
                    try? handle.close()
                }
            }
        }
    }

    func clear() {
        entries.removeAll()
        // Also clear the log file
        if let url = logFileURL {
            try? "".write(to: url, atomically: true, encoding: .utf8)
        }
    }

    func exportText() -> String {
        return entries.map { entry in
            "[\(formatter.string(from: entry.timestamp))] [\(entry.category)] \(entry.message)"
        }.joined(separator: "\n")
    }

    /// Get the path to the log file (for display to user)
    var logFilePath: String? {
        logFileURL?.path
    }
}

/// Global logging function that uses LogStore
func appLog(_ message: String, category: String = "App") {
    Task { @MainActor in
        LogStore.shared.log(message, category: category)
    }
}

// MARK: - Log Viewer View

struct LogViewerView: View {
    @ObservedObject private var logStore = LogStore.shared
    @State private var autoScroll = true
    @State private var showingShareSheet = false
    @State private var filterText = ""

    private var filteredEntries: [LogStore.LogEntry] {
        if filterText.isEmpty {
            return logStore.entries
        }
        return logStore.entries.filter {
            $0.message.localizedCaseInsensitiveContains(filterText) ||
            $0.category.localizedCaseInsensitiveContains(filterText)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Filter bar
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Filter logs...", text: $filterText)
                    .textFieldStyle(.plain)
                if !filterText.isEmpty {
                    Button {
                        filterText = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(8)
            .background(.bar)

            Divider()

            // Log entries
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(filteredEntries) { entry in
                            LogEntryRow(entry: entry)
                                .id(entry.id)
                        }
                    }
                    .padding(8)
                }
                .onChange(of: logStore.entries.count) { _, _ in
                    if autoScroll, let last = filteredEntries.last {
                        withAnimation {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }
        }
        .navigationTitle("Logs")
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Toggle(isOn: $autoScroll) {
                    Image(systemName: autoScroll ? "arrow.down.circle.fill" : "arrow.down.circle")
                }
                .help("Auto-scroll")

                Button {
                    showingShareSheet = true
                } label: {
                    Image(systemName: "square.and.arrow.up")
                }
                .help("Export logs")

                Button(role: .destructive) {
                    logStore.clear()
                } label: {
                    Image(systemName: "trash")
                }
                .help("Clear logs")
            }
        }
        .sheet(isPresented: $showingShareSheet) {
            ShareSheet(text: logStore.exportText())
        }
    }
}

struct LogEntryRow: View {
    let entry: LogStore.LogEntry

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Text(Self.timeFormatter.string(from: entry.timestamp))
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)

            Text(entry.category)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(categoryColor)
                .frame(width: 80, alignment: .leading)

            Text(entry.message)
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
        }
    }

    private var categoryColor: Color {
        switch entry.category {
        case "ACPSSHConnection": return .blue
        case "ACPSSHClient": return .cyan
        case "ACPService": return .green
        case "ChatVM": return .orange
        case "Error": return .red
        default: return .secondary
        }
    }
}

struct ShareSheet: View {
    let text: String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                Text(text)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .padding()
            }
            .navigationTitle("Log Export")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        UIPasteboard.general.string = text
                    } label: {
                        Image(systemName: "doc.on.doc")
                    }
                }
            }
        }
    }
}
