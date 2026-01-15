import SwiftUI

struct ChatView: View {
    @StateObject private var viewModel: ChatViewModel

    let project: Project
    let resumeSession: Session?

    init(project: Project, resumeSession: Session?) {
        self.project = project
        self.resumeSession = resumeSession
        _viewModel = StateObject(wrappedValue: ChatViewModel(project: project))
    }

    var body: some View {
        VStack(spacing: 0) {
            // Connection status bar
            ConnectionStatusBar(
                isConnected: viewModel.isConnected,
                sessionId: viewModel.sessionId,
                serverName: project.server?.name ?? "Unknown",
                model: viewModel.model,
                totalCost: viewModel.totalCost
            )

            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 16) {
                        ForEach(viewModel.messages) { message in
                            MessageView(
                                message: message,
                                onToggleToolCall: viewModel.toggleToolCallExpanded
                            )
                            .id(message.id)
                        }
                    }
                    .padding()
                }
                .onChange(of: viewModel.messages.count) { _, _ in
                    if let lastId = viewModel.messages.last?.id {
                        withAnimation {
                            proxy.scrollTo(lastId, anchor: .bottom)
                        }
                    }
                }
            }

            // Error banner
            if let error = viewModel.errorMessage {
                ErrorBanner(message: error) {
                    viewModel.errorMessage = nil
                }
            }

            // Input bar
            InputBar(
                text: $viewModel.inputText,
                isLoading: viewModel.isLoading,
                isStreaming: viewModel.isStreaming,
                isConnected: viewModel.isConnected,
                onSend: {
                    Task { await viewModel.sendMessage() }
                },
                onCancel: {
                    // Cancel not implemented for non-interactive mode
                }
            )
        }
        .navigationTitle(project.name)
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                if let sessionId = viewModel.sessionId {
                    Text(sessionId.prefix(8) + "...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .task {
            await viewModel.connect(password: nil)

            if viewModel.isConnected, let session = resumeSession {
                await viewModel.resumeSession(session)
            }
        }
        .onDisappear {
            Task { await viewModel.disconnect() }
        }
    }
}

// MARK: - Connection Status Bar

struct ConnectionStatusBar: View {
    let isConnected: Bool
    let sessionId: String?
    let serverName: String
    let model: String?
    let totalCost: Double

    var body: some View {
        HStack {
            Circle()
                .fill(isConnected ? Color.green : Color.red)
                .frame(width: 8, height: 8)

            Text(isConnected ? "Connected to \(serverName)" : "Disconnected")
                .font(.caption)

            Spacer()

            if let model = model {
                Text(model.split(separator: "-").prefix(2).joined(separator: "-"))
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.secondary.opacity(0.2))
                    .clipShape(Capsule())
            }

            if totalCost > 0 {
                Text("$\(totalCost, specifier: "%.4f")")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }
}

// MARK: - Message View

struct MessageView: View {
    let message: ChatDisplayMessage
    let onToggleToolCall: (String) -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            // Avatar
            Circle()
                .fill(message.role == .user ? Color.blue : Color.purple)
                .frame(width: 32, height: 32)
                .overlay {
                    Image(systemName: message.role == .user ? "person.fill" : "cpu")
                        .font(.caption)
                        .foregroundStyle(.white)
                }

            VStack(alignment: .leading, spacing: 8) {
                // Role header
                HStack {
                    Text(message.role == .user ? "You" : "Assistant")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(.secondary)

                    Spacer()

                    Text(message.timestamp.formatted(date: .omitted, time: .shortened))
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }

                // Content blocks - rendered in order (text and tool calls interleaved)
                ForEach(message.contentBlocks) { block in
                    switch block {
                    case .text(_, let content):
                        if !content.isEmpty {
                            MessageContentView(content: content, isStreaming: message.isStreaming)
                        }
                    case .toolCall(let toolCall):
                        ToolCallView(toolCall: toolCall) {
                            onToggleToolCall(toolCall.id)
                        }
                    }
                }

                // Show streaming indicator if no content blocks yet
                if message.contentBlocks.isEmpty && message.isStreaming {
                    HStack(spacing: 4) {
                        ForEach(0..<3, id: \.self) { _ in
                            Circle()
                                .fill(.secondary)
                                .frame(width: 6, height: 6)
                                .opacity(0.5)
                        }
                    }
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(message.role == .user ? Color.blue.opacity(0.1) : Color.secondary.opacity(0.1))
        )
    }
}

// MARK: - Message Content View

struct MessageContentView: View {
    let content: String
    let isStreaming: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Parse and display content with code blocks
            ForEach(parseContent(), id: \.self) { block in
                if block.isCode {
                    CodeBlockView(code: block.content, language: block.language)
                } else {
                    // Render markdown for non-code content
                    MarkdownText(content: block.content)
                }
            }

            if isStreaming {
                HStack(spacing: 4) {
                    ForEach(0..<3, id: \.self) { _ in
                        Circle()
                            .fill(.secondary)
                            .frame(width: 6, height: 6)
                            .opacity(0.5)
                    }
                }
            }
        }
    }

    private func parseContent() -> [ContentBlockUI] {
        var blocks: [ContentBlockUI] = []
        var remaining = content
        let codeBlockStart = "```"
        let codeBlockEnd = "```"

        while let startRange = remaining.range(of: codeBlockStart) {
            // Text before code block
            let textBefore = String(remaining[..<startRange.lowerBound])
            if !textBefore.isEmpty {
                blocks.append(ContentBlockUI(content: textBefore, isCode: false, language: nil))
            }

            // Find the end of the code block
            let afterStart = remaining[startRange.upperBound...]

            // Check for language identifier (until newline)
            var language: String? = nil
            var codeStart = afterStart.startIndex

            if let newlineIndex = afterStart.firstIndex(of: "\n") {
                let langPart = String(afterStart[..<newlineIndex]).trimmingCharacters(in: .whitespaces)
                if !langPart.isEmpty && langPart.allSatisfy({ $0.isLetter || $0.isNumber }) {
                    language = langPart
                }
                codeStart = afterStart.index(after: newlineIndex)
            }

            // Find closing ```
            let searchRange = codeStart..<afterStart.endIndex
            if let endRange = afterStart.range(of: codeBlockEnd, range: searchRange) {
                let code = String(afterStart[codeStart..<endRange.lowerBound])
                blocks.append(ContentBlockUI(content: code, isCode: true, language: language))
                remaining = String(afterStart[endRange.upperBound...])
            } else {
                // No closing ```, treat rest as code
                let code = String(afterStart[codeStart...])
                blocks.append(ContentBlockUI(content: code, isCode: true, language: language))
                remaining = ""
            }
        }

        // Remaining text
        if !remaining.isEmpty {
            blocks.append(ContentBlockUI(content: remaining, isCode: false, language: nil))
        }

        return blocks
    }

    struct ContentBlockUI: Hashable {
        let content: String
        let isCode: Bool
        let language: String?
    }
}

// MARK: - Markdown Text

struct MarkdownText: View {
    let content: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(parseMarkdownBlocks().enumerated()), id: \.offset) { _, block in
                switch block {
                case .paragraph(let text):
                    Text(parseInlineMarkdown(text))
                        .font(.body)
                        .textSelection(.enabled)

                case .header(let level, let text):
                    Text(parseInlineMarkdown(text))
                        .font(headerFont(level: level))
                        .fontWeight(.bold)
                        .padding(.top, level == 1 ? 8 : 4)
                        .textSelection(.enabled)

                case .listItem(let text, let ordered, let index):
                    HStack(alignment: .top, spacing: 8) {
                        Text(ordered ? "\(index)." : "•")
                            .foregroundStyle(.secondary)
                            .frame(width: 20, alignment: .trailing)
                        Text(parseInlineMarkdown(text))
                            .font(.body)
                            .textSelection(.enabled)
                    }

                case .blockquote(let text):
                    HStack(spacing: 8) {
                        Rectangle()
                            .fill(Color.secondary.opacity(0.5))
                            .frame(width: 3)
                        Text(parseInlineMarkdown(text))
                            .font(.body)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                    .padding(.vertical, 4)

                case .horizontalRule:
                    Divider()
                        .padding(.vertical, 8)

                case .table(let rows):
                    MarkdownTableView(rows: rows)
                        .padding(.vertical, 4)
                }
            }
        }
    }

    private func headerFont(level: Int) -> Font {
        switch level {
        case 1: return .title
        case 2: return .title2
        case 3: return .title3
        default: return .headline
        }
    }

    private func parseInlineMarkdown(_ text: String) -> AttributedString {
        // Try to parse as markdown for inline elements
        if let attributed = try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)) {
            return attributed
        }
        return AttributedString(text)
    }

    private enum MarkdownBlock {
        case paragraph(String)
        case header(Int, String)
        case listItem(String, ordered: Bool, index: Int)
        case blockquote(String)
        case horizontalRule
        case table([[String]])  // rows of cells
    }

    private func parseMarkdownBlocks() -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        let lines = content.components(separatedBy: "\n")
        var currentParagraph = ""
        var orderedListIndex = 0
        var tableRows: [[String]] = []

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // Table row (starts and ends with |, or just contains |)
            if trimmed.contains("|") && (trimmed.hasPrefix("|") || trimmed.contains(" | ")) {
                // Flush paragraph first
                if !currentParagraph.isEmpty {
                    blocks.append(.paragraph(currentParagraph.trimmingCharacters(in: .whitespacesAndNewlines)))
                    currentParagraph = ""
                }

                // Check if it's a separator row (|---|---|)
                let isSeparator = trimmed.replacingOccurrences(of: "|", with: "")
                    .replacingOccurrences(of: "-", with: "")
                    .replacingOccurrences(of: ":", with: "")
                    .trimmingCharacters(in: .whitespaces)
                    .isEmpty

                if !isSeparator {
                    // Parse cells
                    let cells = trimmed
                        .trimmingCharacters(in: CharacterSet(charactersIn: "|"))
                        .components(separatedBy: "|")
                        .map { $0.trimmingCharacters(in: .whitespaces) }
                    tableRows.append(cells)
                }
                continue
            }

            // If we were building a table and hit a non-table line, flush the table
            if !tableRows.isEmpty {
                blocks.append(.table(tableRows))
                tableRows = []
            }

            // Horizontal rule
            if trimmed.allSatisfy({ $0 == "-" || $0 == "*" || $0 == "_" }) && trimmed.count >= 3 {
                if !currentParagraph.isEmpty {
                    blocks.append(.paragraph(currentParagraph.trimmingCharacters(in: .whitespacesAndNewlines)))
                    currentParagraph = ""
                }
                blocks.append(.horizontalRule)
                orderedListIndex = 0
                continue
            }

            // Headers (# Header)
            if let headerInfo = parseHeader(trimmed) {
                if !currentParagraph.isEmpty {
                    blocks.append(.paragraph(currentParagraph.trimmingCharacters(in: .whitespacesAndNewlines)))
                    currentParagraph = ""
                }
                blocks.append(.header(headerInfo.level, headerInfo.text))
                orderedListIndex = 0
                continue
            }

            // Unordered list items (- item, * item, + item)
            if let listText = parseUnorderedListItem(trimmed) {
                if !currentParagraph.isEmpty {
                    blocks.append(.paragraph(currentParagraph.trimmingCharacters(in: .whitespacesAndNewlines)))
                    currentParagraph = ""
                }
                blocks.append(.listItem(listText, ordered: false, index: 0))
                orderedListIndex = 0
                continue
            }

            // Ordered list items (1. item)
            if let listText = parseOrderedListItem(trimmed) {
                if !currentParagraph.isEmpty {
                    blocks.append(.paragraph(currentParagraph.trimmingCharacters(in: .whitespacesAndNewlines)))
                    currentParagraph = ""
                }
                orderedListIndex += 1
                blocks.append(.listItem(listText, ordered: true, index: orderedListIndex))
                continue
            }

            // Blockquote
            if trimmed.hasPrefix(">") {
                if !currentParagraph.isEmpty {
                    blocks.append(.paragraph(currentParagraph.trimmingCharacters(in: .whitespacesAndNewlines)))
                    currentParagraph = ""
                }
                var text = String(trimmed.dropFirst())
                if text.hasPrefix(" ") {
                    text = String(text.dropFirst())
                }
                blocks.append(.blockquote(text))
                orderedListIndex = 0
                continue
            }

            // Empty line - end paragraph
            if trimmed.isEmpty {
                if !currentParagraph.isEmpty {
                    blocks.append(.paragraph(currentParagraph.trimmingCharacters(in: .whitespacesAndNewlines)))
                    currentParagraph = ""
                }
                orderedListIndex = 0
                continue
            }

            // Regular text - add to paragraph
            if !currentParagraph.isEmpty {
                currentParagraph += " "
            }
            currentParagraph += line
            orderedListIndex = 0
        }

        // Flush remaining table
        if !tableRows.isEmpty {
            blocks.append(.table(tableRows))
        }

        // Remaining paragraph
        if !currentParagraph.isEmpty {
            blocks.append(.paragraph(currentParagraph.trimmingCharacters(in: .whitespacesAndNewlines)))
        }

        return blocks
    }

    private func parseHeader(_ line: String) -> (level: Int, text: String)? {
        var level = 0
        var idx = line.startIndex
        while idx < line.endIndex && line[idx] == "#" && level < 6 {
            level += 1
            idx = line.index(after: idx)
        }
        guard level > 0, idx < line.endIndex, line[idx] == " " else { return nil }
        let text = String(line[line.index(after: idx)...]).trimmingCharacters(in: .whitespaces)
        return text.isEmpty ? nil : (level, text)
    }

    private func parseUnorderedListItem(_ line: String) -> String? {
        guard line.count >= 2 else { return nil }
        let first = line[line.startIndex]
        let second = line[line.index(after: line.startIndex)]
        if (first == "-" || first == "*" || first == "+") && second == " " {
            return String(line.dropFirst(2))
        }
        return nil
    }

    private func parseOrderedListItem(_ line: String) -> String? {
        // Find digits at start
        var idx = line.startIndex
        while idx < line.endIndex && line[idx].isNumber {
            idx = line.index(after: idx)
        }
        guard idx > line.startIndex else { return nil }
        // Check for ". "
        guard idx < line.endIndex && line[idx] == "." else { return nil }
        let afterDot = line.index(after: idx)
        guard afterDot < line.endIndex && line[afterDot] == " " else { return nil }
        return String(line[line.index(after: afterDot)...])
    }
}

// MARK: - Markdown Table View

struct MarkdownTableView: View {
    let rows: [[String]]

    var body: some View {
        if rows.isEmpty { return AnyView(EmptyView()) }

        let columnCount = rows.map { $0.count }.max() ?? 0
        guard columnCount > 0 else { return AnyView(EmptyView()) }

        return AnyView(
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(rows.enumerated()), id: \.offset) { rowIndex, row in
                    HStack(spacing: 0) {
                        ForEach(0..<columnCount, id: \.self) { colIndex in
                            let cellText = colIndex < row.count ? row[colIndex] : ""
                            Text(cellText)
                                .font(rowIndex == 0 ? .body.bold() : .body)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .textSelection(.enabled)
                                .background(rowIndex == 0 ? Color.secondary.opacity(0.1) : Color.clear)

                            if colIndex < columnCount - 1 {
                                Divider()
                            }
                        }
                    }
                    .background(
                        Rectangle()
                            .stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
                    )
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .overlay(
                RoundedRectangle(cornerRadius: 4)
                    .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
            )
        )
    }
}

// MARK: - Error Banner

struct ErrorBanner: View {
    let message: String
    let onDismiss: () -> Void

    var body: some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.yellow)

            Text(message)
                .font(.caption)
                .lineLimit(2)

            Spacer()

            Button {
                onDismiss()
            } label: {
                Image(systemName: "xmark")
                    .font(.caption)
            }
        }
        .padding()
        .background(.red.opacity(0.1))
    }
}

#Preview {
    NavigationStack {
        ChatView(
            project: Project.preview,
            resumeSession: nil
        )
    }
}
