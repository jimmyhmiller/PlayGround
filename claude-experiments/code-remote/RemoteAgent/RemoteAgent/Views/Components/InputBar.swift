import SwiftUI

struct InputBar: View {
    @Binding var text: String
    let isLoading: Bool
    let isStreaming: Bool
    let isConnected: Bool
    let onSend: () -> Void
    let onCancel: () -> Void
    var onInterruptAndSend: (() -> Void)? = nil

    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            Divider()

            HStack(alignment: .bottom, spacing: 12) {
                // Text input
                TextField("Message...", text: $text, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...6)
                    .focused($isFocused)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 20)
                            .fill(Color.secondary.opacity(0.1))
                    )
                    .onSubmit {
                        if hasText && isConnected && !isLoading {
                            if isStreaming, let interruptAndSend = onInterruptAndSend {
                                interruptAndSend()
                            } else if !isStreaming {
                                onSend()
                            }
                        }
                    }

                // Action buttons
                if isStreaming {
                    HStack(spacing: 8) {
                        // Cancel button
                        Button {
                            onCancel()
                        } label: {
                            Image(systemName: "stop.circle.fill")
                                .font(.title2)
                                .foregroundStyle(.red)
                        }
                        .help("Cancel")

                        // Interrupt and send button (when there's text)
                        if hasText, onInterruptAndSend != nil {
                            Button {
                                onInterruptAndSend?()
                            } label: {
                                Image(systemName: "arrow.up.circle.fill")
                                    .font(.title2)
                                    .foregroundStyle(.orange)
                            }
                            .help("Interrupt and send")
                        }
                    }
                } else if isLoading {
                    ProgressView()
                        .frame(width: 28, height: 28)
                } else {
                    Button {
                        onSend()
                    } label: {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.title2)
                            .foregroundStyle(canSend ? .blue : .gray)
                    }
                    .disabled(!canSend)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(.bar)
        }
    }

    private var hasText: Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var canSend: Bool {
        isConnected && hasText && !isLoading
    }
}

#Preview {
    VStack {
        Spacer()
        InputBar(
            text: .constant("Hello"),
            isLoading: false,
            isStreaming: false,
            isConnected: true,
            onSend: {},
            onCancel: {}
        )
    }
}
