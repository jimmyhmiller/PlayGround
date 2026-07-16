import Foundation

/// Disk cache so the app renders instantly from the last known state and
/// refreshes in the background.
struct AppCache: Codable {
    var slugs: [String: String] = [:]
    var prs: [String: [PullRequest]] = [:]
    var wtCounts: [String: Int] = [:]
    var needsAuth: Set<String> = []
    var expanded: Set<String> = []
    var prDiffs: [String: String] = [:]
    var prComments: [String: [RemoteComment]] = [:]
    var prTouched: [String: Date] = [:]
}

enum CacheStore {
    static var cacheURL: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AgentReview", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("cache.json")
    }

    static func load() -> AppCache {
        guard let data = try? Data(contentsOf: cacheURL),
              let cache = try? JSONDecoder().decode(AppCache.self, from: data) else {
            return AppCache()
        }
        return cache
    }

    static func save(_ cache: AppCache) {
        var c = cache
        // Don't let the cache grow without bound: drop giant diffs, keep the 50
        // most recently touched PRs.
        c.prDiffs = c.prDiffs.filter { $0.value.utf8.count < 1_500_000 }
        if c.prDiffs.count > 50 {
            let keep = Set(c.prTouched.sorted { $0.value > $1.value }.prefix(50).map(\.key))
            c.prDiffs = c.prDiffs.filter { keep.contains($0.key) }
            c.prComments = c.prComments.filter { keep.contains($0.key) }
            c.prTouched = c.prTouched.filter { keep.contains($0.key) }
        }
        if let data = try? JSONEncoder().encode(c) {
            try? data.write(to: cacheURL, options: .atomic)
        }
    }
}
