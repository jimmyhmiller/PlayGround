import SwiftUI

struct ExportHeatmapView: View {
    let goals: [Goal]
    let cells: [HeatmapCell]

    private let calendar = Calendar.current
    private let cellSize: CGFloat = 12
    private let cellSpacing: CGFloat = 2

    var body: some View {
        let today = calendar.startOfDay(for: Date())
        let numWeeks = cells.count / 7
        let maxAmount = cells.map { $0.totalAmount }.max() ?? 1
        let dayLabels = ["S", "M", "T", "W", "T", "F", "S"]

        HStack(alignment: .top, spacing: 16) {
            // Left: header + grid
            VStack(alignment: .leading, spacing: 0) {
                // Header
                Text("Ease")
                    .font(.system(size: 16, weight: .semibold, design: .rounded))
                    .foregroundColor(Color(red: 0.2, green: 0.2, blue: 0.2))

                if let firstDate = cells.first?.date, let lastDate = cells.last?.date {
                    Text(dateRangeString(from: firstDate, to: lastDate))
                        .font(.system(size: 11))
                        .foregroundColor(Color(red: 0.6, green: 0.6, blue: 0.6))
                        .padding(.top, 1)
                }

                Spacer().frame(height: 14)

                // Day-of-week headers
                HStack(spacing: cellSpacing) {
                    ForEach(0..<7, id: \.self) { i in
                        Text(dayLabels[i])
                            .font(.system(size: 8, weight: .medium))
                            .foregroundColor(Color(red: 0.55, green: 0.55, blue: 0.55))
                            .frame(width: cellSize)
                    }
                }
                .padding(.bottom, 4)

                // Heatmap grid
                VStack(alignment: .leading, spacing: cellSpacing) {
                    ForEach(0..<numWeeks, id: \.self) { weekIndex in
                        if shouldShowMonthLabel(weekIndex: weekIndex) {
                            Text(monthLabelText(weekIndex: weekIndex))
                                .font(.system(size: 9, weight: .medium))
                                .foregroundColor(Color(red: 0.45, green: 0.45, blue: 0.45))
                                .padding(.top, weekIndex == 0 ? 0 : 6)
                                .padding(.bottom, 2)
                        }

                        HStack(spacing: cellSpacing) {
                            ForEach(0..<7, id: \.self) { dayIndex in
                                let cellIndex = weekIndex * 7 + dayIndex
                                if cellIndex < cells.count {
                                    let cell = cells[cellIndex]
                                    if cell.date > today {
                                        Color.clear
                                            .frame(width: cellSize, height: cellSize)
                                    } else {
                                        exportCell(cell, maxAmount: maxAmount)
                                            .frame(width: cellSize, height: cellSize)
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Right: legend
            if !goals.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(goals) { goal in
                        HStack(spacing: 5) {
                            RoundedRectangle(cornerRadius: 2)
                                .fill(goal.color)
                                .frame(width: 8, height: 8)
                            Text(goal.name)
                                .font(.system(size: 10))
                                .foregroundColor(Color(red: 0.4, green: 0.4, blue: 0.4))
                        }
                    }
                }
            }
        }
        .padding(24)
        .background(Color.white)
        .colorScheme(.light)
    }

    // MARK: - Cell

    private func exportCell(_ cell: HeatmapCell, maxAmount: Double) -> some View {
        let amounts = orderedGoalAmounts(for: cell)

        return ZStack {
            RoundedRectangle(cornerRadius: 2)
                .fill(Color(red: 0.94, green: 0.94, blue: 0.94))

            if cell.totalAmount > 0 {
                if amounts.count == 1, let goal = goals.first(where: { $0.id == amounts[0].goalId }) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(goal.color)
                } else if amounts.count > 1 {
                    HStack(spacing: 0) {
                        ForEach(amounts.indices, id: \.self) { idx in
                            let ga = amounts[idx]
                            if let goal = goals.first(where: { $0.id == ga.goalId }) {
                                goal.color
                                    .frame(width: cellSize * (ga.amount / cell.totalAmount))
                            }
                        }
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 2))
                }
            }
        }
    }

    // MARK: - Helpers

    private func orderedGoalAmounts(for cell: HeatmapCell) -> [(goalId: UUID, amount: Double)] {
        goals.compactMap { goal in
            cell.goalAmounts.first(where: { $0.goalId == goal.id })
        }
    }

    private func shouldShowMonthLabel(weekIndex: Int) -> Bool {
        let idx = weekIndex * 7
        guard idx < cells.count else { return false }
        if weekIndex == 0 { return true }
        let prevIdx = (weekIndex - 1) * 7
        guard prevIdx >= 0, prevIdx < cells.count else { return true }
        let cur = calendar.component(.month, from: cells[idx].date)
        let prev = calendar.component(.month, from: cells[prevIdx].date)
        return cur != prev
    }

    private func monthLabelText(weekIndex: Int) -> String {
        let idx = weekIndex * 7
        guard idx < cells.count else { return "" }
        let formatter = DateFormatter()
        let date = cells[idx].date
        if weekIndex == 0 || calendar.component(.year, from: date) != calendar.component(.year, from: cells[(weekIndex - 1) * 7].date) {
            formatter.dateFormat = "MMM yyyy"
        } else {
            formatter.dateFormat = "MMM"
        }
        return formatter.string(from: date)
    }

    private func dateRangeString(from start: Date, to end: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM yyyy"
        let startStr = formatter.string(from: start)
        let endStr = formatter.string(from: end)
        if startStr == endStr { return startStr }
        return "\(startStr) – \(endStr)"
    }
}
