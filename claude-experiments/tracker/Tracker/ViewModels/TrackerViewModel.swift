import SwiftUI
import Combine

enum TimePeriod: String, CaseIterable {
    case day = "Day"
    case week = "Week"
    case month = "Month"
    case all = "All"

    var startDate: Date? {
        let calendar = Calendar.current
        let now = Date()
        switch self {
        case .day:
            return calendar.startOfDay(for: now)
        case .week:
            return calendar.date(from: calendar.dateComponents([.yearForWeekOfYear, .weekOfYear], from: now))
        case .month:
            return calendar.date(from: calendar.dateComponents([.year, .month], from: now))
        case .all:
            return nil
        }
    }
}

@MainActor
class TrackerViewModel: ObservableObject {
    @Published var goals: [Goal] = []
    @Published var entries: [Entry] = []
    @Published var selectedPeriod: TimePeriod = .week
    @Published var isAddingGoal: Bool = false

    private let dataStore = DataStore.shared

    init() {
        loadData()
    }

    func loadData() {
        let stored = dataStore.load()
        goals = stored.goals
        entries = stored.entries
    }

    func save() {
        dataStore.saveGoals(goals, entries: entries)
    }

    func addGoal(name: String, colorHex: String) {
        let goal = Goal(name: name, colorHex: colorHex)
        goals.append(goal)
        save()
    }

    func deleteGoal(_ goal: Goal) {
        goals.removeAll { $0.id == goal.id }
        entries.removeAll { $0.goalId == goal.id }
        save()
    }

    func clearAllData() {
        entries = []
        save()
    }

    func updateGoalColor(_ goal: Goal, colorHex: String) {
        if let index = goals.firstIndex(where: { $0.id == goal.id }) {
            goals[index].colorHex = colorHex
            save()
        }
    }

    func addEntry(for goal: Goal, amount: Double) {
        let entry = Entry(goalId: goal.id, amount: amount)
        entries.append(entry)
        save()
    }

    func filteredEntries(for period: TimePeriod) -> [Entry] {
        guard let startDate = period.startDate else {
            return entries
        }
        return entries.filter { $0.timestamp >= startDate }
    }

    func totalForGoal(_ goal: Goal, in period: TimePeriod) -> Double {
        filteredEntries(for: period)
            .filter { $0.goalId == goal.id }
            .reduce(0) { $0 + $1.amount }
    }

    func proportions(for period: TimePeriod) -> [(goal: Goal, proportion: Double)] {
        let totals = goals.map { goal in
            (goal: goal, total: totalForGoal(goal, in: period))
        }

        let grandTotal = totals.reduce(0) { $0 + $1.total }

        if grandTotal == 0 {
            // No data - show empty bars
            return goals.map { ($0, 0.0) }
        }

        return totals.map { ($0.goal, $0.total / grandTotal) }
    }
}
