import SwiftUI

struct CalendarHeatmapView: View {
    @EnvironmentObject var viewModel: EaseViewModel

    private let calendar = Calendar.current

    // Fixed cell heights based on known popover width (280 - 32 padding - 8 heatmap padding = 240)
    // Day: 12 columns, 2px gaps → (240 - 22) / 12 ≈ 18
    // Week/Month: 7 columns, 2px gaps → (240 - 12) / 7 ≈ 32
    private let dayCellHeight: CGFloat = 18
    private let weekMonthCellHeight: CGFloat = 32
    private let allTimeCellHeight: CGFloat = 8
    private let allTimeCellSpacing: CGFloat = 1

    @State private var displayedPeriod: TimePeriod = .week

    var body: some View {
        let cells = viewModel.heatmapCells(for: displayedPeriod)
        let filteredGoalId = viewModel.hoveredGoalId

        VStack(spacing: 4) {
            switch displayedPeriod {
            case .day:
                dayView(cells: cells, filteredGoalId: filteredGoalId)
            case .week:
                weekView(cells: cells, filteredGoalId: filteredGoalId)
            case .month:
                monthView(cells: cells, filteredGoalId: filteredGoalId)
            case .all:
                allTimeView(cells: cells, filteredGoalId: filteredGoalId)
            }
        }
        .onAppear { displayedPeriod = viewModel.selectedPeriod }
        .onChange(of: viewModel.selectedPeriod) { newPeriod in
            withAnimation(.easeInOut(duration: 0.25)) {
                displayedPeriod = newPeriod
            }
        }
        .contentShape(Rectangle())
        .onTapGesture {
            withAnimation(.easeInOut(duration: 0.2)) {
                viewModel.showCalendarView = false
            }
        }
    }

    // MARK: - Day View (24 hours, 2 rows of 12)

    private func dayView(cells: [HeatmapCell], filteredGoalId: UUID?) -> some View {
        VStack(spacing: 2) {
            HStack(spacing: 2) {
                ForEach(0..<12, id: \.self) { i in
                    segmentedCell(cells[i], filteredGoalId: filteredGoalId)
                        .frame(height: dayCellHeight)
                        .help(hourLabel(i))
                }
            }
            HStack(spacing: 2) {
                ForEach(12..<24, id: \.self) { i in
                    segmentedCell(cells[i], filteredGoalId: filteredGoalId)
                        .frame(height: dayCellHeight)
                        .help(hourLabel(i))
                }
            }
            HStack {
                Text("12am").font(.system(size: 8)).foregroundColor(.secondary)
                Spacer()
                Text("12pm").font(.system(size: 8)).foregroundColor(.secondary)
                Spacer()
                Text("11pm").font(.system(size: 8)).foregroundColor(.secondary)
            }
        }
    }

    // MARK: - Week View (7 days)

    private func weekView(cells: [HeatmapCell], filteredGoalId: UUID?) -> some View {
        let dayLabels = ["S", "M", "T", "W", "T", "F", "S"]

        return VStack(spacing: 2) {
            HStack(spacing: 2) {
                ForEach(0..<min(7, cells.count), id: \.self) { i in
                    VStack(spacing: 2) {
                        Text(dayLabels[i])
                            .font(.system(size: 9, weight: .medium))
                            .foregroundColor(.secondary)
                        segmentedCell(cells[i], filteredGoalId: filteredGoalId)
                            .frame(height: weekMonthCellHeight)
                    }
                }
            }
        }
    }

    // MARK: - Month View (calendar grid)

    private func monthView(cells: [HeatmapCell], filteredGoalId: UUID?) -> some View {
        let startOfMonth = displayedPeriod.startDate!
        let weekday = calendar.component(.weekday, from: startOfMonth)
        let offset = weekday - 1
        let dayLabels = ["S", "M", "T", "W", "T", "F", "S"]
        let totalSlots = offset + cells.count
        let numRows = (totalSlots + 6) / 7

        return VStack(spacing: 2) {
            // Weekday headers
            HStack(spacing: 2) {
                ForEach(0..<7, id: \.self) { i in
                    Text(dayLabels[i])
                        .font(.system(size: 9, weight: .medium))
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity)
                }
            }

            // Calendar rows
            ForEach(0..<numRows, id: \.self) { row in
                HStack(spacing: 2) {
                    ForEach(0..<7, id: \.self) { col in
                        let slotIndex = row * 7 + col
                        let cellIndex = slotIndex - offset
                        if cellIndex >= 0 && cellIndex < cells.count {
                            ZStack {
                                segmentedCell(cells[cellIndex], filteredGoalId: filteredGoalId)
                                Text("\(calendar.component(.day, from: cells[cellIndex].date))")
                                    .font(.system(size: 8))
                                    .foregroundColor(.primary.opacity(0.5))
                            }
                            .frame(height: weekMonthCellHeight)
                        } else {
                            Color.clear
                                .frame(height: weekMonthCellHeight)
                        }
                    }
                }
            }
        }
    }

    // MARK: - All Time View (vertical, compact rows)

    private func allTimeView(cells: [HeatmapCell], filteredGoalId: UUID?) -> some View {
        let numWeeks = cells.count / 7
        let today = calendar.startOfDay(for: Date())
        let dayLabels = ["S", "M", "T", "W", "T", "F", "S"]

        let monthLabelCount = (0..<numWeeks).filter { shouldShowMonthLabel(weekIndex: $0, cells: cells) }.count
        let rowHeight = allTimeCellHeight + allTimeCellSpacing
        let contentHeight = CGFloat(numWeeks) * rowHeight + CGFloat(monthLabelCount) * 14
        let viewHeight = min(contentHeight, 300)

        return VStack(spacing: 2) {
            if cells.isEmpty {
                Text("No data yet")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.vertical, 8)
            } else {
                HStack(spacing: allTimeCellSpacing) {
                    ForEach(0..<7, id: \.self) { i in
                        Text(dayLabels[i])
                            .font(.system(size: 8, weight: .medium))
                            .foregroundColor(.secondary)
                            .frame(maxWidth: .infinity)
                    }
                }

                ScrollView(.vertical, showsIndicators: false) {
                    VStack(spacing: allTimeCellSpacing) {
                        ForEach(0..<numWeeks, id: \.self) { weekIndex in
                            if shouldShowMonthLabel(weekIndex: weekIndex, cells: cells) {
                                HStack {
                                    Text(monthLabelText(weekIndex: weekIndex, cells: cells))
                                        .font(.system(size: 9, weight: .medium))
                                        .foregroundColor(.secondary)
                                    Spacer()
                                }
                                .padding(.top, weekIndex == 0 ? 0 : 2)
                            }

                            HStack(spacing: allTimeCellSpacing) {
                                ForEach(0..<7, id: \.self) { dayIndex in
                                    let cellIndex = weekIndex * 7 + dayIndex
                                    if cellIndex < cells.count {
                                        let cell = cells[cellIndex]
                                        let isFuture = cell.date > today
                                        if isFuture {
                                            Color.clear
                                                .frame(height: allTimeCellHeight)
                                        } else {
                                            segmentedCell(cell, filteredGoalId: filteredGoalId, cornerRadius: 1.5, horizontal: true)
                                                .frame(height: allTimeCellHeight)
                                        }
                                    }
                                }
                            }
                        }
                    }
                    .rotationEffect(.degrees(180))
                }
                .rotationEffect(.degrees(180))
                .frame(height: viewHeight)
            }
        }
    }

    // MARK: - Month Label Helpers

    private func shouldShowMonthLabel(weekIndex: Int, cells: [HeatmapCell]) -> Bool {
        let idx = weekIndex * 7
        guard idx < cells.count else { return false }
        if weekIndex == 0 { return true }
        let prevIdx = (weekIndex - 1) * 7
        guard prevIdx >= 0, prevIdx < cells.count else { return true }
        let cur = calendar.component(.month, from: cells[idx].date)
        let prev = calendar.component(.month, from: cells[prevIdx].date)
        return cur != prev
    }

    private func monthLabelText(weekIndex: Int, cells: [HeatmapCell]) -> String {
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

    // MARK: - Segmented Cell (proportional color stripes)

    private func segmentedCell(
        _ cell: HeatmapCell,
        filteredGoalId: UUID?,
        cornerRadius: CGFloat = 2,
        horizontal: Bool = false
    ) -> some View {
        Group {
            if let goalId = filteredGoalId {
                let goal = viewModel.goals.first(where: { $0.id == goalId })
                let amount = cell.goalAmounts.first(where: { $0.goalId == goalId })?.amount ?? 0
                if amount > 0 {
                    RoundedRectangle(cornerRadius: cornerRadius)
                        .fill(goal?.color ?? .gray)
                } else {
                    RoundedRectangle(cornerRadius: cornerRadius)
                        .fill(Color.primary.opacity(0.06))
                }
            } else if cell.totalAmount > 0 {
                let amounts = orderedGoalAmounts(for: cell)
                if horizontal {
                    GeometryReader { geo in
                        HStack(spacing: 0) {
                            ForEach(amounts.indices, id: \.self) { idx in
                                let ga = amounts[idx]
                                if let goal = viewModel.goals.first(where: { $0.id == ga.goalId }) {
                                    goal.color
                                        .frame(width: geo.size.width * (ga.amount / cell.totalAmount))
                                }
                            }
                        }
                        .clipShape(RoundedRectangle(cornerRadius: cornerRadius))
                    }
                } else {
                    GeometryReader { geo in
                        VStack(spacing: 0) {
                            ForEach(amounts.indices, id: \.self) { idx in
                                let ga = amounts[idx]
                                if let goal = viewModel.goals.first(where: { $0.id == ga.goalId }) {
                                    goal.color
                                        .frame(height: geo.size.height * (ga.amount / cell.totalAmount))
                                }
                            }
                        }
                        .clipShape(RoundedRectangle(cornerRadius: cornerRadius))
                    }
                }
            } else {
                RoundedRectangle(cornerRadius: cornerRadius)
                    .fill(Color.primary.opacity(0.06))
            }
        }
    }

    // MARK: - Helpers

    private func orderedGoalAmounts(for cell: HeatmapCell) -> [(goalId: UUID, amount: Double)] {
        viewModel.goals.compactMap { goal in
            cell.goalAmounts.first(where: { $0.goalId == goal.id })
        }
    }

    private func hourLabel(_ hour: Int) -> String {
        if hour == 0 { return "12am" }
        if hour < 12 { return "\(hour)am" }
        if hour == 12 { return "12pm" }
        return "\(hour - 12)pm"
    }
}
