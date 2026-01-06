import Foundation
import SwiftData

@Model
final class Project {
    var id: UUID
    var name: String
    var remotePath: String
    var server: Server?
    var createdAt: Date

    init(name: String, remotePath: String, server: Server? = nil) {
        self.id = UUID()
        self.name = name
        self.remotePath = remotePath
        self.server = server
        self.createdAt = Date()
    }
}

extension Project {
    static var preview: Project {
        Project(name: "My Project", remotePath: "/home/developer/projects/myapp")
    }
}
