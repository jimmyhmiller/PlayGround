import Foundation

struct Entry: Identifiable, Codable, Equatable {
    let id: UUID
    let goalId: UUID
    let amount: Double
    let timestamp: Date
    var isDeleted: Bool

    init(id: UUID = UUID(), goalId: UUID, amount: Double, timestamp: Date = Date(), isDeleted: Bool = false) {
        self.id = id
        self.goalId = goalId
        self.amount = amount
        self.timestamp = timestamp
        self.isDeleted = isDeleted
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        goalId = try container.decode(UUID.self, forKey: .goalId)
        amount = try container.decode(Double.self, forKey: .amount)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        isDeleted = try container.decodeIfPresent(Bool.self, forKey: .isDeleted) ?? false
    }
}
