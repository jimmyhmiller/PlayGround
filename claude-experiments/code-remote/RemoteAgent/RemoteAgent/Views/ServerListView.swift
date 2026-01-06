import SwiftUI
import SwiftData

struct ServerListView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Server.name) private var servers: [Server]

    @State private var showingAddSheet = false
    @State private var selectedServer: Server?

    var body: some View {
        List {
            if servers.isEmpty {
                ContentUnavailableView(
                    "No Servers",
                    systemImage: "server.rack",
                    description: Text("Add a server to get started")
                )
            } else {
                ForEach(servers) { server in
                    NavigationLink(value: server) {
                        ServerRow(server: server)
                    }
                    .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                        Button(role: .destructive) {
                            deleteServer(server)
                        } label: {
                            Label("Delete", systemImage: "trash")
                        }

                        Button {
                            selectedServer = server
                        } label: {
                            Label("Edit", systemImage: "pencil")
                        }
                        .tint(.orange)
                    }
                }
            }
        }
        .navigationTitle("Servers")
        .navigationDestination(for: Server.self) { server in
            ProjectListView(server: server)
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showingAddSheet = true
                } label: {
                    Label("Add Server", systemImage: "plus")
                }
            }
        }
        .sheet(isPresented: $showingAddSheet) {
            ServerFormView(mode: .add) { server in
                modelContext.insert(server)
            }
        }
        .sheet(item: $selectedServer) { server in
            ServerFormView(mode: .edit(server)) { _ in
                // Changes are made directly to the managed object
            }
        }
    }

    private func deleteServer(_ server: Server) {
        modelContext.delete(server)
    }
}

// MARK: - Server Row

struct ServerRow: View {
    let server: Server

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "server.rack")
                .font(.title2)
                .foregroundStyle(.secondary)
                .frame(width: 32)

            VStack(alignment: .leading, spacing: 2) {
                Text(server.name)
                    .font(.headline)

                Text("\(server.username)@\(server.host):\(server.port)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Text("\(server.projects.count)")
                .font(.caption)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.secondary.opacity(0.2))
                .clipShape(Capsule())
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Server Form

enum ServerFormMode {
    case add
    case edit(Server)
}

struct ServerFormView: View {
    @Environment(\.dismiss) private var dismiss

    let mode: ServerFormMode
    let onSave: (Server) -> Void

    @State private var name: String = ""
    @State private var host: String = ""
    @State private var port: String = "22"
    @State private var username: String = ""
    @State private var authMethod: AuthMethod = .password
    @State private var privateKeyPath: String = ""

    @State private var isTesting = false
    @State private var testResult: TestResult?

    enum TestResult {
        case success
        case failure(String)
    }

    private var isEditing: Bool {
        if case .edit = mode { return true }
        return false
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Server Details") {
                    TextField("Name", text: $name)
                        .textContentType(.name)

                    TextField("Host", text: $host)
                        .textContentType(.URL)
                        #if os(iOS)
                        .autocapitalization(.none)
                        .keyboardType(.URL)
                        #endif

                    TextField("Port", text: $port)
                        #if os(iOS)
                        .keyboardType(.numberPad)
                        #endif

                    TextField("Username", text: $username)
                        .textContentType(.username)
                        #if os(iOS)
                        .autocapitalization(.none)
                        #endif
                }

                Section("Authentication") {
                    Picker("Method", selection: $authMethod) {
                        Text("Password").tag(AuthMethod.password)
                        Text("Private Key").tag(AuthMethod.privateKey)
                    }
                    .pickerStyle(.segmented)

                    if authMethod == .privateKey {
                        TextField("Private Key Path", text: $privateKeyPath)
                            #if os(iOS)
                            .autocapitalization(.none)
                            #endif

                        Text("e.g., ~/.ssh/id_rsa")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section {
                    Button {
                        testConnection()
                    } label: {
                        HStack {
                            if isTesting {
                                ProgressView()
                                    .padding(.trailing, 4)
                            }
                            Text("Test Connection")
                        }
                    }
                    .disabled(host.isEmpty || username.isEmpty || isTesting)

                    if let result = testResult {
                        HStack {
                            switch result {
                            case .success:
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(.green)
                                Text("Connection successful")
                            case .failure(let error):
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.red)
                                Text(error)
                                    .font(.caption)
                            }
                        }
                    }
                }
            }
            .navigationTitle(isEditing ? "Edit Server" : "Add Server")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        saveServer()
                        dismiss()
                    }
                    .disabled(name.isEmpty || host.isEmpty || username.isEmpty)
                }
            }
            .onAppear {
                if case .edit(let server) = mode {
                    name = server.name
                    host = server.host
                    port = String(server.port)
                    username = server.username
                    authMethod = server.authMethod
                    privateKeyPath = server.privateKeyPath ?? ""
                }
            }
        }
    }

    private func saveServer() {
        let portNumber = Int(port) ?? 22

        switch mode {
        case .add:
            let server = Server(
                name: name,
                host: host,
                port: portNumber,
                username: username,
                authMethod: authMethod
            )
            if authMethod == .privateKey {
                server.privateKeyPath = privateKeyPath
            }
            onSave(server)

        case .edit(let server):
            server.name = name
            server.host = host
            server.port = portNumber
            server.username = username
            server.authMethod = authMethod
            server.privateKeyPath = authMethod == .privateKey ? privateKeyPath : nil
        }
    }

    private func testConnection() {
        isTesting = true
        testResult = nil

        Task {
            // Simulate connection test
            try? await Task.sleep(nanoseconds: 1_000_000_000)

            await MainActor.run {
                isTesting = false
                // In real implementation, actually test SSH connection
                testResult = .success
            }
        }
    }
}

#Preview {
    NavigationStack {
        ServerListView()
    }
    .modelContainer(for: [Server.self, Project.self], inMemory: true)
}
