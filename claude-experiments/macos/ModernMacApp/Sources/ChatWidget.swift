import SwiftUI

struct ChatWidget: View {
    @State private var messages: [ChatMessage] = [
        ChatMessage(id: UUID(), content: "Hello! I'm Claude, your AI assistant. How can I help you today?", isFromAgent: true, agentName: "Claude", timestamp: Date().addingTimeInterval(-60))
    ]
    
    @State private var newMessage = ""
    @State private var isTyping = false
    @State private var streamingMessage = ""
    @State private var currentStreamingMessageId: UUID?
    @State private var selectedAgent = "Claude"
    
    @StateObject private var claudeService = SimpleClaudeService()
    @StateObject private var mockService = MockClaudeService()
    @State private var useMockService = false
    
    let availableAgents = ["Claude", "Claude Assistant", "Claude Helper"]
    
    var body: some View {
        VStack(spacing: 0) {
            // Chat header
            ChatHeader(
                selectedAgent: $selectedAgent, 
                availableAgents: availableAgents,
                useMockService: $useMockService
            )
            
            Divider()
                .background(.white.opacity(0.1))
            
            // Messages area
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(messages) { message in
                            ChatMessageView(message: message)
                                .id(message.id)
                        }
                        
                        // Show typing indicator when request starts and no content yet
                        if isTyping && streamingMessage.isEmpty {
                            TypingIndicator(agentName: selectedAgent)
                        }
                        
                        // Show streaming message if one is in progress
                        if let streamingId = currentStreamingMessageId, !streamingMessage.isEmpty {
                            StreamingMessageView(
                                content: streamingMessage,
                                agentName: selectedAgent
                            )
                            .id(streamingId)
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                }
                .onChange(of: messages.count) {
                    if let lastMessage = messages.last {
                        withAnimation(.easeInOut(duration: 0.3)) {
                            proxy.scrollTo(lastMessage.id, anchor: .bottom)
                        }
                    }
                }
                .onChange(of: streamingMessage) {
                    if let streamingId = currentStreamingMessageId {
                        withAnimation(.easeInOut(duration: 0.1)) {
                            proxy.scrollTo(streamingId, anchor: .bottom)
                        }
                    }
                }
            }
            
            Divider()
                .background(.white.opacity(0.1))
            
            // Input area
            ChatInputArea(
                newMessage: $newMessage,
                isTyping: $isTyping,
                selectedAgent: selectedAgent,
                onSendMessage: sendMessage
            )
        }
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay {
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(.white.opacity(0.2), lineWidth: 1)
        }
        .shadow(color: .black.opacity(0.1), radius: 20, x: 0, y: 10)
    }
    
    private func sendMessage() {
        guard !newMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        let userMessage = ChatMessage(
            id: UUID(),
            content: newMessage,
            isFromAgent: false,
            agentName: nil,
            timestamp: Date()
        )
        
        messages.append(userMessage)
        let messageToSend = newMessage
        newMessage = ""
        
        // Start typing indicator immediately
        isTyping = true
        
        // Create streaming message
        let streamingId = UUID()
        currentStreamingMessageId = streamingId
        streamingMessage = ""
        
        // Use mock or real service based on toggle
        if useMockService {
            mockService.sendMessage(
                messageToSend,
                conversationHistory: messages,
                onDelta: { delta in
                    // Hide typing indicator when first content arrives
                    if isTyping {
                        isTyping = false
                    }
                    // Update streaming message with each delta
                    streamingMessage += delta
                },
                onComplete: { fullResponse in
                    // Hide typing indicator
                    isTyping = false
                    
                    // Convert streaming message to final message
                    let agentResponse = ChatMessage(
                        id: UUID(),
                        content: fullResponse,
                        isFromAgent: true,
                        agentName: selectedAgent,
                        timestamp: Date()
                    )
                    
                    withAnimation(.easeInOut(duration: 0.3)) {
                        messages.append(agentResponse)
                        
                        // Clear streaming state
                        currentStreamingMessageId = nil
                        streamingMessage = ""
                    }
                },
                onError: { error in
                    // Hide typing indicator
                    isTyping = false
                    
                    // Show error message
                    let errorResponse = ChatMessage(
                        id: UUID(),
                        content: "Sorry, I encountered an error: \(error.localizedDescription)",
                        isFromAgent: true,
                        agentName: selectedAgent,
                        timestamp: Date()
                    )
                    
                    withAnimation(.easeInOut(duration: 0.3)) {
                        messages.append(errorResponse)
                        
                        // Clear streaming state
                        currentStreamingMessageId = nil
                        streamingMessage = ""
                    }
                }
            )
        } else {
            claudeService.sendMessage(
                messageToSend,
                conversationHistory: messages,
                onDelta: { delta in
                    // Hide typing indicator when first content arrives
                    if isTyping {
                        isTyping = false
                    }
                    // Update streaming message with each delta
                    streamingMessage += delta
                },
                onComplete: { fullResponse in
                    // Hide typing indicator
                    isTyping = false
                    
                    // Convert streaming message to final message
                    let agentResponse = ChatMessage(
                        id: UUID(),
                        content: fullResponse,
                        isFromAgent: true,
                        agentName: selectedAgent,
                        timestamp: Date()
                    )
                    
                    withAnimation(.easeInOut(duration: 0.3)) {
                        messages.append(agentResponse)
                        
                        // Clear streaming state
                        currentStreamingMessageId = nil
                        streamingMessage = ""
                    }
                },
                onError: { error in
                    // Hide typing indicator
                    isTyping = false
                    
                    // Show error message
                    let errorResponse = ChatMessage(
                        id: UUID(),
                        content: "Sorry, I encountered an error: \(error.localizedDescription)",
                        isFromAgent: true,
                        agentName: selectedAgent,
                        timestamp: Date()
                    )
                    
                    withAnimation(.easeInOut(duration: 0.3)) {
                        messages.append(errorResponse)
                        
                        // Clear streaming state
                        currentStreamingMessageId = nil
                        streamingMessage = ""
                    }
                }
            )
        }
    }
}

struct ChatHeader: View {
    @Binding var selectedAgent: String
    let availableAgents: [String]
    @Binding var useMockService: Bool
    
    var body: some View {
        HStack {
            // Agent selector
            Menu {
                ForEach(availableAgents, id: \.self) { agent in
                    Button(agent) {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            selectedAgent = agent
                        }
                    }
                }
            } label: {
                HStack(spacing: 8) {
                    AgentAvatar(agentName: selectedAgent, size: 32)
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text(selectedAgent)
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(.primary)
                        
                        HStack(spacing: 4) {
                            Circle()
                                .fill(.green)
                                .frame(width: 6, height: 6)
                            
                            Text("Online")
                                .font(.system(size: 12))
                                .foregroundStyle(.secondary)
                        }
                    }
                    
                    Spacer()
                    
                    Image(systemName: "chevron.down")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 4)
            }
            .buttonStyle(.plain)
            
            Spacer()
            
            // Chat actions
            HStack(spacing: 8) {
                // Mock service toggle
                Toggle(isOn: $useMockService) {
                    Text("Mock")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                .toggleStyle(.switch)
                .scaleEffect(0.8)
                
                GlassButton(
                    icon: "paperclip",
                    compact: true,
                    action: { /* Attach file */ }
                )
                
                GlassButton(
                    icon: "ellipsis",
                    compact: true,
                    action: { /* More options */ }
                )
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }
}

struct ChatMessageView: View {
    let message: ChatMessage
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            if message.isFromAgent {
                AgentAvatar(agentName: message.agentName ?? "Agent", size: 28)
            } else {
                Spacer()
                    .frame(width: 28)
            }
            
            VStack(alignment: message.isFromAgent ? .leading : .trailing, spacing: 4) {
                if message.isFromAgent, let agentName = message.agentName {
                    Text(agentName)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                
                Text(message.content)
                    .font(.system(size: 14))
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background {
                        if message.isFromAgent {
                            RoundedRectangle(cornerRadius: 16)
                                .fill(.ultraThinMaterial)
                                .overlay {
                                    RoundedRectangle(cornerRadius: 16)
                                        .strokeBorder(.white.opacity(0.2), lineWidth: 1)
                                }
                        } else {
                            RoundedRectangle(cornerRadius: 16)
                                .fill(.blue.gradient)
                        }
                    }
                    .foregroundStyle(message.isFromAgent ? .primary : Color.white)
                
                Text(formatTimestamp(message.timestamp))
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
            
            if !message.isFromAgent {
                Spacer()
                    .frame(width: 28)
            }
        }
        .frame(maxWidth: .infinity, alignment: message.isFromAgent ? .leading : .trailing)
    }
    
    private func formatTimestamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

struct StreamingMessageView: View {
    let content: String
    let agentName: String
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            AgentAvatar(agentName: agentName, size: 28)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(agentName)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                
                Text(content + "▊") // Add cursor indicator
                    .font(.system(size: 14))
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background {
                        RoundedRectangle(cornerRadius: 16)
                            .fill(.ultraThinMaterial)
                            .overlay {
                                RoundedRectangle(cornerRadius: 16)
                                    .strokeBorder(.white.opacity(0.2), lineWidth: 1)
                            }
                    }
                    .textSelection(.enabled) // Allow text selection even while streaming
            }
            
            Spacer()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct TypingIndicator: View {
    let agentName: String
    @State private var animationOffset: CGFloat = 0
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            AgentAvatar(agentName: agentName, size: 28)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(agentName)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                
                HStack(spacing: 4) {
                    ForEach(0..<3, id: \.self) { index in
                        Circle()
                            .fill(.secondary)
                            .frame(width: 6, height: 6)
                            .offset(y: animationOffset)
                            .animation(
                                .easeInOut(duration: 0.6)
                                .repeatForever()
                                .delay(Double(index) * 0.2),
                                value: animationOffset
                            )
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background {
                    RoundedRectangle(cornerRadius: 16)
                        .fill(.ultraThinMaterial)
                        .overlay {
                            RoundedRectangle(cornerRadius: 16)
                                .strokeBorder(.white.opacity(0.2), lineWidth: 1)
                        }
                }
            }
            
            Spacer()
        }
        .onAppear {
            animationOffset = -3
        }
    }
}

struct ChatInputArea: View {
    @Binding var newMessage: String
    @Binding var isTyping: Bool
    let selectedAgent: String
    let onSendMessage: () -> Void
    
    @FocusState private var isInputFocused: Bool
    
    var body: some View {
        HStack(spacing: 12) {
            // Message input
            HStack(spacing: 8) {
                TextField("Message \(selectedAgent)...", text: $newMessage, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 14))
                    .focused($isInputFocused)
                    .onSubmit {
                        if !newMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            onSendMessage()
                        }
                    }
                
                if !newMessage.isEmpty {
                    Button {
                        newMessage = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                            .font(.system(size: 16))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background {
                RoundedRectangle(cornerRadius: 20)
                    .fill(.ultraThinMaterial)
                    .overlay {
                        RoundedRectangle(cornerRadius: 20)
                            .strokeBorder(
                                isInputFocused ? .blue.opacity(0.5) : .white.opacity(0.2),
                                lineWidth: isInputFocused ? 2 : 1
                            )
                    }
            }
            
            // Send button
            Button(action: onSendMessage) {
                Image(systemName: "arrow.up")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 32, height: 32)
                    .background {
                        if newMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            Circle()
                                .fill(Color.secondary)
                        } else {
                            Circle()
                                .fill(Color.blue.gradient)
                        }
                    }
            }
            .buttonStyle(.plain)
            .disabled(newMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isTyping)
            .scaleEffect(newMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0.9 : 1.0)
            .animation(.spring(response: 0.3, dampingFraction: 0.7), value: newMessage.isEmpty)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }
}

struct AgentAvatar: View {
    let agentName: String
    let size: CGFloat
    
    private var initials: String {
        agentName.components(separatedBy: " ")
            .compactMap { $0.first }
            .map(String.init)
            .joined()
            .prefix(2)
            .uppercased()
    }
    
    private var avatarColor: Color {
        let colors: [Color] = [.blue, .green, .purple, .orange, .pink]
        let index = abs(agentName.hashValue) % colors.count
        return colors[index]
    }
    
    var body: some View {
        Circle()
            .fill(avatarColor.gradient)
            .frame(width: size, height: size)
            .overlay {
                Text(initials)
                    .font(.system(size: size * 0.4, weight: .semibold))
                    .foregroundStyle(.white)
            }
            .shadow(color: avatarColor.opacity(0.3), radius: 4, x: 0, y: 2)
    }
}

// MARK: - Data Models
struct ChatMessage: Identifiable {
    let id: UUID
    let content: String
    let isFromAgent: Bool
    let agentName: String?
    let timestamp: Date
}