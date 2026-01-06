import Foundation
import SwiftData

enum AuthMethod: String, Codable, CaseIterable {
    case password = "password"
    case privateKey = "privateKey"
}

@Model
final class Server {
    var id: UUID
    var name: String
    var host: String
    var port: Int
    var username: String
    var authMethod: AuthMethod

    // For password auth - stored in Keychain in production
    @Transient var password: String?

    // For key auth
    var privateKeyPath: String?
    var privateKeyPassphrase: String?

    @Relationship(deleteRule: .cascade, inverse: \Project.server)
    var projects: [Project] = []

    var createdAt: Date
    var lastConnectedAt: Date?

    init(
        name: String,
        host: String,
        port: Int = 22,
        username: String,
        authMethod: AuthMethod = .password
    ) {
        self.id = UUID()
        self.name = name
        self.host = host
        self.port = port
        self.username = username
        self.authMethod = authMethod
        self.createdAt = Date()
    }
}

extension Server {
    static var preview: Server {
        Server(name: "Dev Server", host: "192.168.1.100", username: "developer")
    }
}
