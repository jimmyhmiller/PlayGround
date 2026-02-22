import Foundation

struct StoredData: Codable {
    var goals: [Goal]
    var entries: [Entry]
    var deletedGoalIds: Set<UUID>
    var deletedEntryIds: Set<UUID>

    init(goals: [Goal] = [], entries: [Entry] = [], deletedGoalIds: Set<UUID> = [], deletedEntryIds: Set<UUID> = []) {
        self.goals = goals
        self.entries = entries
        self.deletedGoalIds = deletedGoalIds
        self.deletedEntryIds = deletedEntryIds
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        goals = try container.decode([Goal].self, forKey: .goals)
        entries = try container.decode([Entry].self, forKey: .entries)
        deletedGoalIds = try container.decodeIfPresent(Set<UUID>.self, forKey: .deletedGoalIds) ?? []
        deletedEntryIds = try container.decodeIfPresent(Set<UUID>.self, forKey: .deletedEntryIds) ?? []
    }
}

class DataStore {
    static let shared = DataStore()

    private let fileManager = FileManager.default
    private var dataFileURL: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let appFolder = appSupport.appendingPathComponent("Ease", isDirectory: true)

        if !fileManager.fileExists(atPath: appFolder.path) {
            try? fileManager.createDirectory(at: appFolder, withIntermediateDirectories: true)
        }

        return appFolder.appendingPathComponent("data.json")
    }

    private init() {}

    func load() -> StoredData {
        guard fileManager.fileExists(atPath: dataFileURL.path),
              let data = try? Data(contentsOf: dataFileURL),
              let stored = try? JSONDecoder().decode(StoredData.self, from: data) else {
            return StoredData()
        }
        return stored
    }

    func save(_ data: StoredData) {
        guard let encoded = try? JSONEncoder().encode(data) else { return }
        try? encoded.write(to: dataFileURL, options: .atomic)
    }

    func saveGoals(_ goals: [Goal], entries: [Entry], deletedGoalIds: Set<UUID> = [], deletedEntryIds: Set<UUID> = []) {
        save(StoredData(goals: goals, entries: entries, deletedGoalIds: deletedGoalIds, deletedEntryIds: deletedEntryIds))
    }
}
