import SwiftUI

struct ToolCallView: View {
    let toolCall: DisplayToolCall
    let onToggle: () -> Void

    private var displayText: String {
        // Extract command or key info from input JSON
        if !toolCall.input.isEmpty,
           let data = toolCall.input.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            // For bash/terminal - show command
            if let command = json["command"] as? String {
                return command
            }
            // For file operations - show path
            if let path = json["file_path"] as? String {
                return (path as NSString).lastPathComponent
            }
            // For search - show pattern
            if let pattern = json["pattern"] as? String {
                return pattern
            }
            // For description
            if let desc = json["description"] as? String {
                return desc
            }
            // For prompt (Task tool)
            if let prompt = json["prompt"] as? String {
                return String(prompt.prefix(50))
            }
        }
        // Don't repeat the tool name - return empty if no details
        return ""
    }

    private var toolLabel: String {
        let name = toolCall.name.lowercased()
        if name.contains("bash") || name.contains("terminal") || toolCall.name.hasPrefix("`") {
            return "Bash"
        } else if name.contains("read") {
            return "Read"
        } else if name.contains("write") {
            return "Write"
        } else if name.contains("edit") {
            return "Edit"
        } else if name.contains("glob") {
            return "Glob"
        } else if name.contains("grep") {
            return "Grep"
        } else if name.contains("task") {
            return "Task"
        }
        return toolCall.name
    }

    var body: some View {
        HStack(spacing: 6) {
            Text(toolLabel)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundStyle(.secondary)

            if !displayText.isEmpty {
                Text(displayText)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
    }
}

#Preview {
    VStack(spacing: 8) {
        ToolCallView(
            toolCall: DisplayToolCall(
                id: "1",
                name: "Read",
                input: "{\"file_path\": \"/Users/dev/project/src/main.swift\"}",
                isExpanded: false,
                status: .completed
            ),
            onToggle: {}
        )

        ToolCallView(
            toolCall: DisplayToolCall(
                id: "2",
                name: "Bash",
                input: "{\"command\": \"npm install\"}",
                isExpanded: false,
                status: .running
            ),
            onToggle: {}
        )

        ToolCallView(
            toolCall: DisplayToolCall(
                id: "3",
                name: "Grep",
                input: "{\"pattern\": \"func.*async\"}",
                isExpanded: false,
                status: .error
            ),
            onToggle: {}
        )
    }
    .padding()
}
