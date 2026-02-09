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

enum UndoAction {
    case addEntry(Entry)
    case deleteGoal(Goal, [Entry]) // Goal and its associated entries
}

struct HeatmapCell {
    let index: Int
    let date: Date
    let totalAmount: Double
    let goalAmounts: [(goalId: UUID, amount: Double)]
}

@MainActor
class EaseViewModel: ObservableObject {
    @Published var goals: [Goal] = []
    @Published var entries: [Entry] = []
    @Published var selectedPeriod: TimePeriod = .week {
        didSet { UserDefaults.standard.set(selectedPeriod.rawValue, forKey: "selectedPeriod") }
    }
    @Published var isAddingGoal: Bool = false
    @Published var showCalendarView: Bool = false {
        didSet { UserDefaults.standard.set(showCalendarView, forKey: "showCalendarView") }
    }
    @Published var hoveredGoalId: UUID? = nil

    private let dataStore = DataStore.shared
    private var undoStack: [UndoAction] = []
    private var redoStack: [UndoAction] = []

    init() {
        if let savedPeriod = UserDefaults.standard.string(forKey: "selectedPeriod"),
           let period = TimePeriod(rawValue: savedPeriod) {
            selectedPeriod = period
        }
        showCalendarView = UserDefaults.standard.bool(forKey: "showCalendarView")
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
        let goalEntries = entries.filter { $0.goalId == goal.id }
        goals.removeAll { $0.id == goal.id }
        entries.removeAll { $0.goalId == goal.id }
        undoStack.append(.deleteGoal(goal, goalEntries))
        redoStack.removeAll()
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

    func moveGoal(from source: IndexSet, to destination: Int) {
        goals.move(fromOffsets: source, toOffset: destination)
        save()
    }

    func addEntry(for goal: Goal, amount: Double) {
        var adjustedAmount = amount

        // Don't allow going below 0
        if amount < 0 {
            let currentTotal = totalForGoal(goal, in: .all)
            if currentTotal + amount < 0 {
                adjustedAmount = -currentTotal // Only subtract what's available
            }
            if adjustedAmount == 0 { return } // Nothing to subtract
        }

        let entry = Entry(goalId: goal.id, amount: adjustedAmount)
        entries.append(entry)
        undoStack.append(.addEntry(entry))
        redoStack.removeAll()
        save()
    }

    func undo() {
        guard let action = undoStack.popLast() else { return }
        switch action {
        case .addEntry(let entry):
            entries.removeAll { $0.id == entry.id }
            redoStack.append(action)
        case .deleteGoal(let goal, let goalEntries):
            goals.append(goal)
            entries.append(contentsOf: goalEntries)
            redoStack.append(action)
        }
        save()
    }

    func redo() {
        guard let action = redoStack.popLast() else { return }
        switch action {
        case .addEntry(let entry):
            entries.append(entry)
            undoStack.append(action)
        case .deleteGoal(let goal, _):
            let goalEntries = entries.filter { $0.goalId == goal.id }
            goals.removeAll { $0.id == goal.id }
            entries.removeAll { $0.goalId == goal.id }
            undoStack.append(.deleteGoal(goal, goalEntries))
        }
        save()
    }

    var canUndo: Bool { !undoStack.isEmpty }
    var canRedo: Bool { !redoStack.isEmpty }

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

    // MARK: - Icon Goals

    // MARK: - Heatmap Data

    func heatmapCells(for period: TimePeriod) -> [HeatmapCell] {
        let calendar = Calendar.current

        switch period {
        case .day:
            let startOfDay = calendar.startOfDay(for: Date())
            return (0..<24).map { hour in
                let hourStart = calendar.date(byAdding: .hour, value: hour, to: startOfDay)!
                let hourEnd = calendar.date(byAdding: .hour, value: 1, to: hourStart)!
                return makeCell(index: hour, date: hourStart, from: hourStart, to: hourEnd)
            }

        case .week:
            let startOfWeek = period.startDate!
            return (0..<7).map { day in
                let dayStart = calendar.date(byAdding: .day, value: day, to: startOfWeek)!
                let dayEnd = calendar.date(byAdding: .day, value: 1, to: dayStart)!
                return makeCell(index: day, date: dayStart, from: dayStart, to: dayEnd)
            }

        case .month:
            let startOfMonth = period.startDate!
            let range = calendar.range(of: .day, in: .month, for: startOfMonth)!
            return range.enumerated().map { (i, day) in
                let dayDate = calendar.date(byAdding: .day, value: day - 1, to: startOfMonth)!
                let dayStart = calendar.startOfDay(for: dayDate)
                let dayEnd = calendar.date(byAdding: .day, value: 1, to: dayStart)!
                return makeCell(index: i, date: dayStart, from: dayStart, to: dayEnd)
            }

        case .all:
            let today = calendar.startOfDay(for: Date())
            guard let earliest = entries.map({ $0.timestamp }).min() else {
                return []
            }
            let startOfFirstWeek = calendar.date(from: calendar.dateComponents([.yearForWeekOfYear, .weekOfYear], from: earliest))!
            let startOfCurrentWeek = calendar.date(from: calendar.dateComponents([.yearForWeekOfYear, .weekOfYear], from: today))!
            let endDate = calendar.date(byAdding: .day, value: 6, to: startOfCurrentWeek)!
            let totalDays = calendar.dateComponents([.day], from: startOfFirstWeek, to: endDate).day! + 1
            let totalWeeks = (totalDays + 6) / 7
            let totalCells = totalWeeks * 7
            return (0..<totalCells).map { offset in
                let dayStart = calendar.date(byAdding: .day, value: offset, to: startOfFirstWeek)!
                let dayEnd = calendar.date(byAdding: .day, value: 1, to: dayStart)!
                return makeCell(index: offset, date: dayStart, from: dayStart, to: dayEnd)
            }
        }
    }

    private func makeCell(index: Int, date: Date, from: Date, to: Date) -> HeatmapCell {
        let cellEntries = entries.filter { $0.timestamp >= from && $0.timestamp < to }
        let goalAmounts = goals.compactMap { goal -> (goalId: UUID, amount: Double)? in
            let amount = cellEntries.filter { $0.goalId == goal.id }.reduce(0) { $0 + $1.amount }
            return amount > 0 ? (goalId: goal.id, amount: amount) : nil
        }
        let total = cellEntries.reduce(0) { $0 + $1.amount }
        return HeatmapCell(index: index, date: date, totalAmount: total, goalAmounts: goalAmounts)
    }

    /// Proportions for the first 3 goals (used for icon rendering)
    func iconProportions(for period: TimePeriod) -> [(proportion: Double, colorHex: String)] {
        let iconGoalsList = Array(goals.prefix(3))

        if iconGoalsList.isEmpty {
            // No goals - show 3 equal bars with default gray
            return [(1.0/3.0, "#888888"), (1.0/3.0, "#888888"), (1.0/3.0, "#888888")]
        }

        let totals = iconGoalsList.map { goal in
            (colorHex: goal.colorHex, total: totalForGoal(goal, in: period))
        }

        let grandTotal = totals.reduce(0) { $0 + $1.total }

        if grandTotal == 0 {
            // No data logged - show equal-length bars
            let equalProportion = 1.0 / Double(iconGoalsList.count)
            return totals.map { (equalProportion, $0.colorHex) }
        }

        return totals.map { ($0.total / grandTotal, $0.colorHex) }
    }
}
