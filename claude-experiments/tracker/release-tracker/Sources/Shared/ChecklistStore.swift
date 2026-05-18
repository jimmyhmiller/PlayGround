import Foundation
#if canImport(WidgetKit)
import WidgetKit
#endif

/// Reads and writes the checklist JSON inside the shared App Group container.
///
/// Both the host app and the widget extension instantiate this; the App
/// Group entitlement (`group.com.jimmyhmiller.ReleaseTracker`) is what lets
/// them point at the same file.
///
/// When the host app mutates the data, it should call `reloadWidgets()` so
/// the desktop widget refreshes immediately.
public final class ChecklistStore {
    public static let appGroup = "group.com.jimmyhmiller.ReleaseTracker"
    public static let widgetKind = "ReleaseTrackerWidget"

    public init() {}

    /// URL of the checklist JSON file inside the App Group container.
    ///
    /// Falls back to the user's standard Application Support directory when
    /// the App Group container is unavailable (e.g. running unsigned via
    /// `swift run`). This means the widget will see stale/empty data in
    /// that case, but the host app still functions.
    public var fileURL: URL {
        let fm = FileManager.default
        if let container = fm.containerURL(forSecurityApplicationGroupIdentifier: Self.appGroup) {
            let dir = container.appendingPathComponent("Library/Application Support", isDirectory: true)
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
            return dir.appendingPathComponent("checklist.json")
        }
        let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dir = support.appendingPathComponent("ReleaseTracker", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("checklist.json")
    }

    public func load() -> Checklist {
        let url = fileURL
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url),
              var existing = try? JSONDecoder().decode(Checklist.self, from: data) else {
            return seedChecklist()
        }

        if existing.seedVersion < SeedData.currentSeedVersion {
            existing = merge(existing: existing, with: SeedData.seedItems())
            try? save(existing)
        }
        return existing
    }

    public func save(_ checklist: Checklist) throws {
        let data = try JSONEncoder().encode(checklist)
        try data.write(to: fileURL, options: .atomic)
        reloadWidgets()
    }

    public func toggle(_ itemID: UUID) throws {
        var current = load()
        guard let idx = current.items.firstIndex(where: { $0.id == itemID }) else { return }
        current.items[idx].isDone.toggle()
        try save(current)
    }

    /// Re-seed everything. Wipes user state — call only on explicit user action.
    public func resetToSeed() throws {
        try save(seedChecklist())
    }

    private func seedChecklist() -> Checklist {
        Checklist(items: SeedData.seedItems(), seedVersion: SeedData.currentSeedVersion)
    }

    /// Merge a new seed list into an existing checklist, preserving the
    /// `isDone` state of items the user has already checked. New items
    /// keep `isDone = false`. Items that no longer exist in the seed are
    /// dropped.
    private func merge(existing: Checklist, with newSeed: [ChecklistItem]) -> Checklist {
        let priorByTitle = Dictionary(
            uniqueKeysWithValues: existing.items.map { ($0.title, $0) }
        )
        let merged = newSeed.map { fresh -> ChecklistItem in
            var item = fresh
            if let prior = priorByTitle[fresh.title] {
                item.id = prior.id
                item.isDone = prior.isDone
            }
            return item
        }
        return Checklist(items: merged, seedVersion: SeedData.currentSeedVersion)
    }

    public func reloadWidgets() {
        #if canImport(WidgetKit)
        WidgetCenter.shared.reloadAllTimelines()
        #endif
    }
}
