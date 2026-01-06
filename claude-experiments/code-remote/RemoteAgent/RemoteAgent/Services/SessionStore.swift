import Foundation

// MARK: - Session Store

actor SessionStore {
    private let userDefaults: UserDefaults
    private let sessionsKey = "stored_sessions"

    private var sessions: [UUID: [Session]] = [:] // projectId -> sessions

    init(userDefaults: UserDefaults = .standard) {
        self.userDefaults = userDefaults
        // Load synchronously during init (before actor is fully isolated)
        if let data = userDefaults.data(forKey: sessionsKey),
           let decoded = try? JSONDecoder().decode([UUID: [Session]].self, from: data) {
            sessions = decoded
        }
    }

    // MARK: - Public API

    func sessions(for projectId: UUID) -> [Session] {
        return sessions[projectId] ?? []
    }

    func addSession(_ session: Session) {
        var projectSessions = sessions[session.projectId] ?? []
        projectSessions.insert(session, at: 0)

        // Keep only last 50 sessions per project
        if projectSessions.count > 50 {
            projectSessions = Array(projectSessions.prefix(50))
        }

        sessions[session.projectId] = projectSessions
        saveToDisk()
    }

    func updateSession(_ session: Session) {
        guard var projectSessions = sessions[session.projectId] else { return }

        if let index = projectSessions.firstIndex(where: { $0.id == session.id }) {
            projectSessions[index] = session
            sessions[session.projectId] = projectSessions
            saveToDisk()
        }
    }

    func deleteSession(id: String, projectId: UUID) {
        guard var projectSessions = sessions[projectId] else { return }

        projectSessions.removeAll { $0.id == id }
        sessions[projectId] = projectSessions
        saveToDisk()
    }

    func deleteAllSessions(for projectId: UUID) {
        sessions[projectId] = nil
        saveToDisk()
    }

    // MARK: - Persistence

    private func saveToDisk() {
        guard let data = try? JSONEncoder().encode(sessions) else { return }
        userDefaults.set(data, forKey: sessionsKey)
    }
}
