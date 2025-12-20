import Foundation

struct StoredData: Codable {
    var goals: [Goal]
    var entries: [Entry]
}

class DataStore {
    static let shared = DataStore()

    private let fileManager = FileManager.default
    private var dataFileURL: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let appFolder = appSupport.appendingPathComponent("ProgressTracker", isDirectory: true)

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
            return StoredData(goals: [], entries: [])
        }
        return stored
    }

    func save(_ data: StoredData) {
        guard let encoded = try? JSONEncoder().encode(data) else { return }
        try? encoded.write(to: dataFileURL, options: .atomic)
    }

    func saveGoals(_ goals: [Goal], entries: [Entry]) {
        save(StoredData(goals: goals, entries: entries))
    }
}
