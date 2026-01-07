import Foundation

// MARK: - Session History Types

public struct ACPHistoryMessage: Sendable, Identifiable {
    public let id: String
    public let role: ACPMessageRole
    public let content: String
    public let timestamp: Date
    public let toolCalls: [ACPHistoryToolCall]?

    public init(id: String, role: ACPMessageRole, content: String, timestamp: Date, toolCalls: [ACPHistoryToolCall]? = nil) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.toolCalls = toolCalls
    }
}

public enum ACPMessageRole: String, Sendable, Codable {
    case user
    case assistant
}

public struct ACPHistoryToolCall: Sendable, Identifiable {
    public let id: String
    public let name: String
    public let input: String?
    public let output: String?

    public init(id: String, name: String, input: String? = nil, output: String? = nil) {
        self.id = id
        self.name = name
        self.input = input
        self.output = output
    }
}

// MARK: - Path Encoding

/// Encodes paths for use in session file paths
public struct PathEncoder {
    /// Encode a path by replacing "/" with "-"
    /// This matches the dashboard behavior: `cwd.replace(/\//g, '-')`
    public static func encode(_ path: String) -> String {
        path.replacingOccurrences(of: "/", with: "-")
    }
}

// MARK: - Session File Entry (JSONL format)

struct SessionFileEntry: Decodable {
    let type: String
    let message: SessionMessage?
    let uuid: String?
    let timestamp: String?

    var parsedTimestamp: Date? {
        guard let timestamp = timestamp else { return nil }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: timestamp) ?? ISO8601DateFormatter().date(from: timestamp)
    }
}

struct SessionMessage: Decodable {
    let content: [SessionContentBlock]?
}

enum SessionContentBlock: Decodable {
    case text(String)
    case toolUse(id: String, name: String)
    case other

    enum CodingKeys: String, CodingKey {
        case type, text, id, name
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "text":
            let text = try container.decode(String.self, forKey: .text)
            self = .text(text)
        case "tool_use":
            let id = try container.decode(String.self, forKey: .id)
            let name = try container.decode(String.self, forKey: .name)
            self = .toolUse(id: id, name: name)
        default:
            self = .other
        }
    }
}

// MARK: - Session History Loader

/// Loads session history from Claude Code's local JSONL files
public struct SessionHistoryLoader {

    #if os(macOS)
    /// Get the path to a session file
    /// Path pattern: ~/.claude/projects/{encoded-cwd}/{sessionId}.jsonl
    public static func sessionFilePath(sessionId: String, cwd: String) -> URL {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let encodedCwd = PathEncoder.encode(cwd)
        return homeDir
            .appendingPathComponent(".claude")
            .appendingPathComponent("projects")
            .appendingPathComponent(encodedCwd)
            .appendingPathComponent("\(sessionId).jsonl")
    }

    /// Load and parse session history
    public static func loadHistory(sessionId: String, cwd: String) async throws -> [ACPHistoryMessage] {
        let filePath = sessionFilePath(sessionId: sessionId, cwd: cwd)

        guard FileManager.default.fileExists(atPath: filePath.path) else {
            throw ACPError.sessionNotFound(sessionId)
        }

        let data = try Data(contentsOf: filePath)
        let lines = String(data: data, encoding: .utf8)?
            .components(separatedBy: "\n")
            .filter { !$0.isEmpty } ?? []

        var messages: [ACPHistoryMessage] = []
        let decoder = JSONDecoder()

        for line in lines {
            guard let lineData = line.data(using: .utf8) else { continue }

            do {
                let entry = try decoder.decode(SessionFileEntry.self, from: lineData)

                if let message = parseEntry(entry) {
                    messages.append(message)
                }
            } catch {
                // Skip malformed lines
                acpLog("Failed to parse session history line: \(error)")
                continue
            }
        }

        return messages
    }

    private static func parseEntry(_ entry: SessionFileEntry) -> ACPHistoryMessage? {
        guard let role = ACPMessageRole(rawValue: entry.type) else {
            return nil
        }

        guard let messageContent = entry.message?.content else { return nil }

        var textParts: [String] = []
        var toolCalls: [ACPHistoryToolCall] = []

        for block in messageContent {
            switch block {
            case .text(let text):
                textParts.append(text)
            case .toolUse(let id, let name):
                toolCalls.append(ACPHistoryToolCall(id: id, name: name))
            case .other:
                break
            }
        }

        let textContent = textParts.joined()
        guard !textContent.isEmpty else { return nil }

        return ACPHistoryMessage(
            id: entry.uuid ?? UUID().uuidString,
            role: role,
            content: textContent,
            timestamp: entry.parsedTimestamp ?? Date(),
            toolCalls: toolCalls.isEmpty ? nil : toolCalls
        )
    }

    /// Check if a session file exists
    public static func sessionExists(sessionId: String, cwd: String) -> Bool {
        let filePath = sessionFilePath(sessionId: sessionId, cwd: cwd)
        return FileManager.default.fileExists(atPath: filePath.path)
    }

    /// List all sessions for a given working directory
    public static func listSessions(cwd: String) throws -> [String] {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let encodedCwd = PathEncoder.encode(cwd)
        let projectDir = homeDir
            .appendingPathComponent(".claude")
            .appendingPathComponent("projects")
            .appendingPathComponent(encodedCwd)

        guard FileManager.default.fileExists(atPath: projectDir.path) else {
            return []
        }

        let contents = try FileManager.default.contentsOfDirectory(at: projectDir, includingPropertiesForKeys: nil)
        return contents
            .filter { $0.pathExtension == "jsonl" }
            .map { $0.deletingPathExtension().lastPathComponent }
    }
    #else
    /// Get the path to a session file - not available on iOS
    public static func sessionFilePath(sessionId: String, cwd: String) -> URL {
        fatalError("Local session history is not available on iOS - sessions are stored on the remote Mac")
    }

    /// Load and parse session history - not available on iOS
    public static func loadHistory(sessionId: String, cwd: String) async throws -> [ACPHistoryMessage] {
        throw ACPError.notSupported("Local session history is not available on iOS")
    }

    /// Check if a session file exists - not available on iOS
    public static func sessionExists(sessionId: String, cwd: String) -> Bool {
        return false
    }

    /// List all sessions for a given working directory - not available on iOS
    public static func listSessions(cwd: String) throws -> [String] {
        throw ACPError.notSupported("Local session history is not available on iOS")
    }
    #endif
}
