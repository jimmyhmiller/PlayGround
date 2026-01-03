import Foundation

struct HistoryEntry: Codable, Identifiable, Hashable {
    let display: String
    let timestamp: Int64
    let project: String
    let sessionId: String?

    var id: String { sessionId ?? "\(timestamp)" }

    var date: Date {
        Date(timeIntervalSince1970: Double(timestamp) / 1000.0)
    }

    var projectName: String {
        URL(fileURLWithPath: project).lastPathComponent
    }

    var encodedProjectPath: String {
        project
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: "_", with: "-")
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }

    static func == (lhs: HistoryEntry, rhs: HistoryEntry) -> Bool {
        lhs.id == rhs.id
    }
}
