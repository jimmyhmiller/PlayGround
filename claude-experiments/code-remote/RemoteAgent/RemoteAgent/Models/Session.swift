import Foundation

struct Session: Identifiable, Codable {
    let id: String
    let projectId: UUID
    let createdAt: Date
    var lastActiveAt: Date

    init(id: String, projectId: UUID) {
        self.id = id
        self.projectId = projectId
        self.createdAt = Date()
        self.lastActiveAt = Date()
    }
}

extension Session {
    static var preview: Session {
        Session(id: "session-abc123", projectId: UUID())
    }
}
