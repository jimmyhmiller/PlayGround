import Foundation

struct LocalComment: Codable, Hashable {
    var id: String
    var path: String?
    var line: Int?
    var body: String
    var createdAt: Date
}

/// Local, per-review curation state: deleted/reworded AI comments, own comments,
/// viewed files, chosen verdict.
struct ReviewLocalState: Codable {
    var deletedRemote: Set<String> = []
    var reworded: [String: String] = [:]
    var localComments: [LocalComment] = []
    var viewed: Set<String> = []
    var verdict: String = Verdict.comment.rawValue
}

struct PersistedState: Codable {
    var savedReplies: [String] = [
        "Please add a test that covers this case.",
        "Add error handling here — this can throw.",
        "Remove this; it's dead code.",
        "Extract this into a helper — it repeats elsewhere.",
        "Use a descriptive name here.",
    ]
    var reviews: [String: ReviewLocalState] = [:]
}

enum Persistence {
    static var stateURL: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AgentReview", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("state.json")
    }

    static func load() -> PersistedState {
        guard let data = try? Data(contentsOf: stateURL),
              let state = try? JSONDecoder().decode(PersistedState.self, from: data) else {
            return PersistedState()
        }
        return state
    }

    static func save(_ state: PersistedState) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(state) {
            try? data.write(to: stateURL, options: .atomic)
        }
    }
}
