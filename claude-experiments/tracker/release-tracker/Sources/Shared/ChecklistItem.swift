import Foundation

public struct ChecklistItem: Codable, Identifiable, Equatable, Hashable {
    public var id: UUID
    public var category: String
    public var title: String
    public var note: String?
    public var isDone: Bool
    public var sortOrder: Int

    public init(
        id: UUID = UUID(),
        category: String,
        title: String,
        note: String? = nil,
        isDone: Bool = false,
        sortOrder: Int
    ) {
        self.id = id
        self.category = category
        self.title = title
        self.note = note
        self.isDone = isDone
        self.sortOrder = sortOrder
    }
}

public struct Checklist: Codable, Equatable {
    public var items: [ChecklistItem]
    public var seedVersion: Int

    public init(items: [ChecklistItem], seedVersion: Int) {
        self.items = items
        self.seedVersion = seedVersion
    }

    public var totalCount: Int { items.count }
    public var doneCount: Int { items.filter(\.isDone).count }
    public var fraction: Double {
        guard totalCount > 0 else { return 0 }
        return Double(doneCount) / Double(totalCount)
    }

    public var categories: [String] {
        var seen = Set<String>()
        var ordered: [String] = []
        for item in items.sorted(by: { $0.sortOrder < $1.sortOrder }) {
            if seen.insert(item.category).inserted {
                ordered.append(item.category)
            }
        }
        return ordered
    }

    public func items(in category: String) -> [ChecklistItem] {
        items
            .filter { $0.category == category }
            .sorted { $0.sortOrder < $1.sortOrder }
    }

    public var nextUp: [ChecklistItem] {
        items
            .filter { !$0.isDone }
            .sorted { $0.sortOrder < $1.sortOrder }
    }
}
