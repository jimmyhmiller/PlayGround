import SwiftUI

struct ReconnectSessionView: View {
    let server: Server
    let sessionId: String

    @StateObject private var acpService = ACPService()
    @State private var isConnecting = true
    @State private var error: String?
    @Environment(\.modelContext) private var modelContext

    var body: some View {
        Group {
            if isConnecting {
                VStack(spacing: 16) {
                    ProgressView()
                    Text("Reconnecting to session...")
                        .foregroundStyle(.secondary)
                }
            } else if let error = error {
                ContentUnavailableView(
                    "Connection Failed",
                    systemImage: "exclamationmark.triangle",
                    description: Text(error)
                )
            } else {
                // Show a basic chat interface for the reconnected session
                ReconnectedChatView(acpService: acpService, sessionId: sessionId)
            }
        }
        .navigationTitle("Session")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            await reconnect()
        }
    }

    private func reconnect() async {
        isConnecting = true
        error = nil

        do {
            let password = try? KeychainService.shared.getPassword(for: server.id)

            try await acpService.connectToExistingSession(
                sessionId: sessionId,
                host: server.host,
                port: server.port,
                username: server.username,
                privateKeyPath: server.authMethod == .privateKey ? server.privateKeyPath : nil,
                password: password
            )
            isConnecting = false
        } catch {
            self.error = error.localizedDescription
            isConnecting = false
        }
    }
}

struct ReconnectedChatView: View {
    @ObservedObject var acpService: ACPService
    let sessionId: String

    @State private var inputText = ""
    @State private var messages: [ChatMessage] = []

    struct ChatMessage: Identifiable {
        let id = UUID()
        let role: String
        let content: String
    }

    var body: some View {
        VStack(spacing: 0) {
            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: messages.count) { _, _ in
                    if let last = messages.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }

            Divider()

            // Input
            HStack(spacing: 12) {
                TextField("Message...", text: $inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...5)

                Button {
                    sendMessage()
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                }
                .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            .padding()
        }
        .onAppear {
            messages.append(ChatMessage(role: "system", content: "Reconnected to session \(sessionId)"))
        }
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        messages.append(ChatMessage(role: "user", content: text))
        inputText = ""

        Task {
            do {
                // Send message through ACP
                for try await event in acpService.sendMessage(text) {
                    await MainActor.run {
                        switch event {
                        case .text(let content):
                            if let last = messages.last, last.role == "assistant" {
                                // Append to existing assistant message
                                messages[messages.count - 1] = ChatMessage(role: "assistant", content: last.content + content)
                            } else {
                                messages.append(ChatMessage(role: "assistant", content: content))
                            }
                        default:
                            break
                        }
                    }
                }
            } catch {
                await MainActor.run {
                    messages.append(ChatMessage(role: "error", content: error.localizedDescription))
                }
            }
        }
    }
}

struct MessageBubble: View {
    let message: ReconnectedChatView.ChatMessage

    var body: some View {
        HStack {
            if message.role == "user" {
                Spacer()
            }

            Text(message.content)
                .padding(12)
                .background(backgroundColor)
                .foregroundStyle(foregroundColor)
                .clipShape(RoundedRectangle(cornerRadius: 12))

            if message.role != "user" {
                Spacer()
            }
        }
    }

    private var backgroundColor: Color {
        switch message.role {
        case "user": return .blue
        case "assistant": return Color(.systemGray5)
        case "error": return .red.opacity(0.2)
        default: return Color(.systemGray6)
        }
    }

    private var foregroundColor: Color {
        switch message.role {
        case "user": return .white
        default: return .primary
        }
    }
}
