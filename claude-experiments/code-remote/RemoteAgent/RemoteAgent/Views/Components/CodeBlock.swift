import SwiftUI
#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

struct CodeBlockView: View {
    let code: String
    let language: String?

    @State private var isCopied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                if let language = language {
                    Text(language)
                        .font(.caption2)
                        .fontWeight(.medium)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button {
                    copyToClipboard()
                } label: {
                    Label(
                        isCopied ? "Copied" : "Copy",
                        systemImage: isCopied ? "checkmark" : "doc.on.doc"
                    )
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color(white: 0.15))

            // Code content with syntax highlighting
            ScrollView(.horizontal, showsIndicators: true) {
                Text(highlightedCode)
                    .font(.system(size: 12, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(12)
            }
        }
        .background(Color(white: 0.1))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var highlightedCode: AttributedString {
        var result = AttributedString(code)
        result.foregroundColor = .white

        let lang = language?.lowercased() ?? ""

        // Keywords for various languages
        let keywords: [String]
        switch lang {
        case "swift":
            keywords = ["func", "var", "let", "if", "else", "for", "while", "return", "import", "struct", "class", "enum", "protocol", "extension", "guard", "switch", "case", "default", "try", "catch", "throw", "async", "await", "private", "public", "internal", "static", "self", "Self", "true", "false", "nil"]
        case "python", "py":
            keywords = ["def", "class", "if", "elif", "else", "for", "while", "return", "import", "from", "as", "try", "except", "finally", "with", "lambda", "yield", "async", "await", "True", "False", "None", "and", "or", "not", "in", "is"]
        case "javascript", "js", "typescript", "ts":
            keywords = ["function", "const", "let", "var", "if", "else", "for", "while", "return", "import", "export", "from", "class", "extends", "async", "await", "try", "catch", "throw", "new", "this", "true", "false", "null", "undefined"]
        case "rust":
            keywords = ["fn", "let", "mut", "if", "else", "for", "while", "loop", "return", "use", "mod", "pub", "struct", "enum", "impl", "trait", "async", "await", "match", "self", "Self", "true", "false"]
        case "go":
            keywords = ["func", "var", "const", "if", "else", "for", "range", "return", "import", "package", "type", "struct", "interface", "map", "chan", "go", "defer", "true", "false", "nil"]
        case "bash", "sh", "shell", "zsh":
            keywords = ["if", "then", "else", "elif", "fi", "for", "while", "do", "done", "case", "esac", "function", "return", "export", "local", "echo", "cd", "ls", "grep", "awk", "sed"]
        default:
            keywords = ["if", "else", "for", "while", "return", "function", "class", "import", "export", "true", "false", "null", "nil"]
        }

        // Apply keyword highlighting
        for keyword in keywords {
            var searchRange = result.startIndex..<result.endIndex
            while let range = result[searchRange].range(of: "\\b\(keyword)\\b", options: .regularExpression) {
                result[range].foregroundColor = Color(red: 0.8, green: 0.4, blue: 0.8) // Purple for keywords
                searchRange = range.upperBound..<result.endIndex
            }
        }

        // Highlight strings (basic - double quotes)
        var searchRange = result.startIndex..<result.endIndex
        while let range = result[searchRange].range(of: "\"[^\"\\n]*\"", options: .regularExpression) {
            result[range].foregroundColor = Color(red: 0.6, green: 0.8, blue: 0.5) // Green for strings
            searchRange = range.upperBound..<result.endIndex
        }

        // Highlight single-quoted strings
        searchRange = result.startIndex..<result.endIndex
        while let range = result[searchRange].range(of: "'[^'\\n]*'", options: .regularExpression) {
            result[range].foregroundColor = Color(red: 0.6, green: 0.8, blue: 0.5) // Green for strings
            searchRange = range.upperBound..<result.endIndex
        }

        // Highlight comments (// style)
        searchRange = result.startIndex..<result.endIndex
        while let range = result[searchRange].range(of: "//[^\\n]*", options: .regularExpression) {
            result[range].foregroundColor = Color(white: 0.5) // Gray for comments
            searchRange = range.upperBound..<result.endIndex
        }

        // Highlight comments (# style for Python/Shell)
        if ["python", "py", "bash", "sh", "shell", "zsh"].contains(lang) {
            searchRange = result.startIndex..<result.endIndex
            while let range = result[searchRange].range(of: "#[^\\n]*", options: .regularExpression) {
                result[range].foregroundColor = Color(white: 0.5)
                searchRange = range.upperBound..<result.endIndex
            }
        }

        // Highlight numbers
        searchRange = result.startIndex..<result.endIndex
        while let range = result[searchRange].range(of: "\\b\\d+\\.?\\d*\\b", options: .regularExpression) {
            result[range].foregroundColor = Color(red: 0.9, green: 0.7, blue: 0.4) // Orange for numbers
            searchRange = range.upperBound..<result.endIndex
        }

        return result
    }

    private func copyToClipboard() {
        #if os(iOS)
        UIPasteboard.general.string = code
        #elseif os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(code, forType: .string)
        #endif
        isCopied = true

        Task {
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            isCopied = false
        }
    }
}

#Preview {
    VStack {
        CodeBlockView(
            code: """
            func greet(name: String) -> String {
                let message = "Hello, \\(name)!"
                return message
            }
            """,
            language: "swift"
        )

        CodeBlockView(
            code: "npm install express",
            language: "bash"
        )
    }
    .padding()
}
