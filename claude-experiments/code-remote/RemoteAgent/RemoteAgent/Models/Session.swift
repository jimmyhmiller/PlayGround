import Foundation

struct Session: Identifiable, Codable {
    let id: String
    let projectId: UUID
    let createdAt: Date
    var lastActiveAt: Date
    var title: String?

    init(id: String, projectId: UUID, title: String? = nil) {
        self.id = id
        self.projectId = projectId
        self.createdAt = Date()
        self.lastActiveAt = Date()
        self.title = title
    }

    init(id: String, projectId: UUID, createdAt: Date, title: String? = nil) {
        self.id = id
        self.projectId = projectId
        self.createdAt = createdAt
        self.lastActiveAt = createdAt
        self.title = title
    }
}

extension Session {
    static var preview: Session {
        Session(id: "session-abc123", projectId: UUID(), title: "Implement login feature")
    }
}
