import SwiftUI
import AppKit

struct ConversationDetailView: View {
    let entry: HistoryEntry
    @ObservedObject var dataService: DataService
    @State private var conversation: Conversation?
    @State private var isLoading = true
    @State private var copiedType: String?

    var body: some View {
        VStack(spacing: 0) {
            // Header
            VStack(spacing: 0) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(conversation?.summary ?? entry.display)
                            .font(.system(size: 14, weight: .semibold))
                            .lineLimit(2)
                            .foregroundColor(.primary)

                        HStack(spacing: 16) {
                            if let first = conversation?.firstTimestamp {
                                Label(formatDate(first), systemImage: "calendar")
                                    .font(.system(size: 11))
                                    .foregroundColor(.secondary)
                            }

                            if let count = conversation?.messages.count {
                                Label("\(count) messages", systemImage: "bubble.left.and.bubble.right")
                                    .font(.system(size: 11))
                                    .foregroundColor(.secondary)
                            }
                        }
                    }

                    Spacer()

                    if let sessionId = entry.sessionId {
                        HStack(spacing: 8) {
                            CopyButton(
                                label: "Copy ID",
                                icon: "doc.on.doc",
                                isCopied: copiedType == "id"
                            ) {
                                copyToClipboard(sessionId, type: "id")
                            }

                            CopyButton(
                                label: "Resume",
                                icon: "terminal",
                                isCopied: copiedType == "resume"
                            ) {
                                copyToClipboard("claude --resume \(sessionId) --dangerously-skip-permissions", type: "resume")
                            }
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
            }
            .background(Color(nsColor: .windowBackgroundColor))

            Divider()

            // Messages
            if isLoading {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Loading...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(nsColor: .textBackgroundColor))
            } else if let conversation = conversation {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(conversation.messages) { message in
                            MessageView(message: message)
                        }
                    }
                    .padding(.vertical, 8)
                }
                .background(Color(nsColor: .textBackgroundColor))
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 32))
                        .foregroundColor(.secondary.opacity(0.4))
                    Text("Could not load conversation")
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(nsColor: .textBackgroundColor))
            }
        }
        .task(id: entry.id) {
            isLoading = true
            conversation = await dataService.loadConversation(for: entry)
            isLoading = false
        }
    }

    func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d, yyyy 'at' h:mm a"
        return formatter.string(from: date)
    }

    func copyToClipboard(_ text: String, type: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        copiedType = type
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            if copiedType == type {
                copiedType = nil
            }
        }
    }
}

struct CopyButton: View {
    let label: String
    let icon: String
    let isCopied: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: isCopied ? "checkmark" : icon)
                    .font(.system(size: 11))
                Text(isCopied ? "Copied" : label)
                    .font(.system(size: 11, weight: .medium))
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(isCopied ? Color.green.opacity(0.15) : Color.secondary.opacity(0.1))
            .foregroundColor(isCopied ? .green : .primary)
            .cornerRadius(6)
        }
        .buttonStyle(.plain)
    }
}
