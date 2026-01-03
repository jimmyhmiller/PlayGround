import SwiftUI

struct MessageView: View {
    let message: Message
    @State private var showThinking = false

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            // Avatar
            Circle()
                .fill(message.isUser ? Color.blue : Color.purple)
                .frame(width: 28, height: 28)
                .overlay(
                    Image(systemName: message.isUser ? "person.fill" : "cpu")
                        .font(.system(size: 12))
                        .foregroundColor(.white)
                )

            VStack(alignment: .leading, spacing: 6) {
                // Header
                HStack(spacing: 8) {
                    Text(message.isUser ? "You" : "Claude")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.primary)

                    if let date = message.dateFromTimestamp {
                        Text(formatTime(date))
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                    }
                }

                // Content
                if let content = message.message?.content {
                    switch content {
                    case .string(let text):
                        Text(text)
                            .font(.system(size: 13))
                            .foregroundColor(.primary)
                            .textSelection(.enabled)
                            .lineSpacing(3)

                    case .blocks(let blocks):
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                                ContentBlockView(block: block, showThinking: $showThinking)
                            }
                        }
                    }
                }
            }

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    func formatTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm a"
        return formatter.string(from: date)
    }
}

struct ContentBlockView: View {
    let block: ContentBlock
    @Binding var showThinking: Bool

    var body: some View {
        switch block {
        case .text(let text):
            Text(text)
                .font(.system(size: 13))
                .foregroundColor(.primary)
                .textSelection(.enabled)
                .lineSpacing(3)

        case .thinking(let thinking, _):
            VStack(alignment: .leading, spacing: 0) {
                Button(action: { showThinking.toggle() }) {
                    HStack(spacing: 6) {
                        Image(systemName: showThinking ? "chevron.down" : "chevron.right")
                            .font(.system(size: 10, weight: .semibold))
                        Image(systemName: "brain")
                            .font(.system(size: 11))
                        Text("Thinking")
                            .font(.system(size: 11, weight: .medium))
                        Spacer()
                    }
                    .foregroundColor(.purple)
                    .padding(.vertical, 6)
                }
                .buttonStyle(.plain)

                if showThinking {
                    Text(thinking)
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                        .textSelection(.enabled)
                        .lineSpacing(2)
                        .padding(.leading, 20)
                        .padding(.top, 4)
                }
            }
            .padding(8)
            .background(Color.purple.opacity(0.05))
            .cornerRadius(6)

        case .toolUse(let name, _):
            HStack(spacing: 6) {
                Image(systemName: "hammer.fill")
                    .font(.system(size: 10))
                Text(name)
                    .font(.system(size: 11, weight: .medium))
            }
            .foregroundColor(.orange)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.orange.opacity(0.1))
            .cornerRadius(4)

        case .toolResult(let result):
            if !result.isEmpty {
                Text(result)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.secondary)
                    .lineLimit(5)
                    .padding(8)
                    .background(Color.secondary.opacity(0.05))
                    .cornerRadius(4)
            }

        case .unknown:
            EmptyView()
        }
    }
}
