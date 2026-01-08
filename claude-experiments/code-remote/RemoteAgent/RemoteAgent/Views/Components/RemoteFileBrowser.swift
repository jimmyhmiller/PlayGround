import SwiftUI

struct RemoteFileBrowserView: View {
    let server: Server
    let onSelect: (String) -> Void

    @State private var currentPath: String = ""
    @State private var items: [RemoteFileItem] = []
    @State private var isLoading = false
    @State private var error: String?
    @State private var sshService: SSHService?
    @State private var isConnected = false
    @State private var hasStartedConnection = false
    
    // Directory cache for better performance
    @State private var directoryCache: [String: [RemoteFileItem]] = [:]
    // Debounce task to prevent rapid directory loading
    @State private var loadDirectoryTask: Task<Void, Never>?

    var body: some View {
        Group {
            if !isConnected {
                connectingView
            } else if let error = error {
                errorView(error)
            } else {
                fileListView
            }
        }
        .task {
            guard !hasStartedConnection else { return }
            hasStartedConnection = true
            await connect()
        }
        .onDisappear {
            // Clean up resources when view disappears
            loadDirectoryTask?.cancel()
            Task { await sshService?.disconnect() }
        }
    }

    private var connectingView: some View {
        VStack(spacing: 12) {
            ProgressView()
            Text("Connecting to \(server.name)...")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private func errorView(_ message: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.title)
                .foregroundStyle(.red)
            Text(message)
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button("Retry") {
                Task { await connect() }
            }
            .buttonStyle(.bordered)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var fileListView: some View {
        VStack(spacing: 0) {
            // Breadcrumb path bar
            HStack(spacing: 4) {
                Button {
                    Task { await loadDirectory("/") }
                } label: {
                    Image(systemName: "house.fill")
                        .font(.caption)
                }
                .buttonStyle(.plain)

                Text(currentPath)
                    .font(.system(.caption, design: .monospaced))
                    .lineLimit(1)
                    .truncationMode(.head)

                Spacer()

                if isLoading {
                    ProgressView()
                        .scaleEffect(0.6)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(Color.secondary.opacity(0.1))

            Divider()

            // File list using ScrollView + LazyVStack for reliable rendering
            ScrollView {
                LazyVStack(spacing: 0) {
                    // Parent directory
                    if currentPath != "/" {
                        Button {
                            let parent = (currentPath as NSString).deletingLastPathComponent
                            Task { await loadDirectory(parent) }
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: "arrow.up.circle")
                                    .foregroundStyle(.secondary)
                                Text("..")
                                    .foregroundStyle(.primary)
                                Spacer()
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 6)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)

                        Divider().padding(.leading, 32)
                    }

                    ForEach(items) { item in
                        Button {
                            if item.isDirectory {
                                Task { await loadDirectory(item.path) }
                            }
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: item.isDirectory ? "folder.fill" : "doc")
                                    .foregroundStyle(item.isDirectory ? .blue : .secondary)
                                    .frame(width: 20)

                                Text(item.name)
                                    .foregroundStyle(item.isDirectory ? .primary : .secondary)
                                    .lineLimit(1)

                                Spacer()

                                if item.isDirectory {
                                    Image(systemName: "chevron.right")
                                        .font(.caption2)
                                        .foregroundStyle(.tertiary)
                                }
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 6)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)

                        if item.id != items.last?.id {
                            Divider().padding(.leading, 36)
                        }
                    }
                }
            }

            Divider()

            // Select current folder button
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Selected:")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text((currentPath as NSString).lastPathComponent)
                        .font(.caption)
                        .fontWeight(.medium)
                }

                Spacer()

                Button("Select This Folder") {
                    onSelect(currentPath)
                }
                .buttonStyle(.borderedProminent)
                .disabled(currentPath.isEmpty)
            }
            .padding(8)
            .background(Color.secondary.opacity(0.1))
        }
    }

    private func connect() async {
        let service = SSHService()
        sshService = service
        error = nil

        do {
            try await service.connect(to: server, password: nil)
            isConnected = true
            let home = try await service.getHomeDirectory()
            await loadDirectory(home)
        } catch {
            await MainActor.run {
                self.error = error.localizedDescription
            }
        }
    }

    private func loadDirectory(_ path: String) async {
        // Cancel any pending directory loading to prevent race conditions
        loadDirectoryTask?.cancel()
        
        loadDirectoryTask = Task { @MainActor in
            guard let service = sshService else { return }

            isLoading = true
            error = nil

            // Check cache first
            if let cachedItems = directoryCache[path] {
                self.items = cachedItems
                self.currentPath = path
                self.isLoading = false
                return
            }

            do {
                let loadedItems = try await service.listDirectory(path)
                // Cache the result for future use
                directoryCache[path] = loadedItems
                self.items = loadedItems
                self.currentPath = path
            } catch {
                self.error = error.localizedDescription
                // Remove from cache on error
                directoryCache.removeValue(forKey: path)
            }

            self.isLoading = false
        }
    }
}

// Keep the old view signature for compatibility but redirect to new flow
struct RemoteFileBrowserViewCompat: View {
    let server: Server
    @Binding var selectedPath: String
    @Binding var isPresented: Bool

    var body: some View {
        NavigationStack {
            RemoteFileBrowserView(server: server) { path in
                selectedPath = path
                isPresented = false
            }
            .navigationTitle("Select Directory")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        isPresented = false
                    }
                }
            }
        }
    }
}