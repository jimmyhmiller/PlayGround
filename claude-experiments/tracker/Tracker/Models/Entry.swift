import Foundation

struct Entry: Identifiable, Codable, Equatable {
    let id: UUID
    let goalId: UUID
    let amount: Double
    let timestamp: Date

    init(id: UUID = UUID(), goalId: UUID, amount: Double, timestamp: Date = Date()) {
        self.id = id
        self.goalId = goalId
        self.amount = amount
        self.timestamp = timestamp
    }
}
