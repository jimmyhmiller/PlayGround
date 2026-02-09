import SwiftUI

struct MainPopoverView: View {
    @EnvironmentObject var viewModel: EaseViewModel
    @State private var isCmdHeld: Bool = false
    @State private var eventMonitor: Any?
    @State private var draggingGoal: Goal?

    var body: some View {
        VStack(spacing: 16) {
            if viewModel.isAddingGoal {
                // Inline add goal form
                AddGoalView()
            } else {
                // Main content
                mainContent
            }
        }
        .padding()
        .frame(width: 280)
        .background(
            Group {
                Button("") { viewModel.undo() }
                    .keyboardShortcut("z", modifiers: .command)
                    .opacity(0)

                Button("") { viewModel.redo() }
                    .keyboardShortcut("z", modifiers: [.command, .shift])
                    .opacity(0)
            }
        )
        .onAppear {
            isCmdHeld = NSEvent.modifierFlags.contains(.command)
            eventMonitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { event in
                isCmdHeld = event.modifierFlags.contains(.command)
                // Clear dragging state when Cmd is released
                if !isCmdHeld {
                    draggingGoal = nil
                }
                return event
            }
        }
        .onDisappear {
            if let monitor = eventMonitor {
                NSEvent.removeMonitor(monitor)
                eventMonitor = nil
            }
        }
    }

    private var mainContent: some View {
        VStack(spacing: 16) {
            // Time period picker
            TimePeriodPicker(selection: $viewModel.selectedPeriod, animate: !viewModel.showCalendarView)

            // Bars or calendar heatmap visualization
            if !viewModel.goals.isEmpty {
                Group {
                    if viewModel.showCalendarView {
                        CalendarHeatmapView()
                    } else {
                        ProportionalBarsView()
                    }
                }
                .padding(.horizontal, 4)

                Divider()
            }

            // Goal list
            VStack(spacing: 1) {
                if viewModel.goals.isEmpty {
                    Text("No goals yet")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.vertical, 20)
                } else {
                    ForEach(viewModel.goals) { goal in
                        GoalRowView(
                            goal: goal,
                            isReorderMode: isCmdHeld,
                            onAdd: { amount in
                                viewModel.addEntry(for: goal, amount: amount)
                            },
                            onDelete: {
                                viewModel.deleteGoal(goal)
                            },
                            onColorChange: { colorHex in
                                viewModel.updateGoalColor(goal, colorHex: colorHex)
                            }
                        )
                        .onHover { hovering in
                            viewModel.hoveredGoalId = hovering ? goal.id : nil
                        }
                        .opacity(draggingGoal?.id == goal.id ? 0.5 : 1.0)
                        .animation(.easeInOut(duration: 0.15), value: draggingGoal?.id)
                        .onDrag {
                            guard isCmdHeld else { return NSItemProvider() }
                            draggingGoal = goal
                            return NSItemProvider(object: goal.id.uuidString as NSString)
                        }
                        .onDrop(of: [.text], delegate: GoalDropDelegate(
                            goal: goal,
                            goals: viewModel.goals,
                            draggingGoal: $draggingGoal,
                            onMove: { from, to in
                                viewModel.moveGoal(from: from, to: to)
                            }
                        ))
                    }
                }
            }

            // Add goal button
            Button {
                viewModel.isAddingGoal = true
            } label: {
                HStack {
                    Image(systemName: "plus")
                        .font(.system(size: 10, weight: .medium))
                    Text("Add Goal")
                        .font(.system(size: 11))
                }
                .foregroundColor(.secondary)
                .padding(.vertical, 4)
                .padding(.horizontal, 8)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
    }
}

struct GoalDropDelegate: DropDelegate {
    let goal: Goal
    let goals: [Goal]
    @Binding var draggingGoal: Goal?
    let onMove: (IndexSet, Int) -> Void

    func performDrop(info: DropInfo) -> Bool {
        draggingGoal = nil
        return true
    }

    func dropEntered(info: DropInfo) {
        guard let dragging = draggingGoal,
              dragging.id != goal.id,
              let fromIndex = goals.firstIndex(where: { $0.id == dragging.id }),
              let toIndex = goals.firstIndex(where: { $0.id == goal.id }) else {
            return
        }

        withAnimation(.easeInOut(duration: 0.2)) {
            onMove(IndexSet(integer: fromIndex), toIndex > fromIndex ? toIndex + 1 : toIndex)
        }
    }

    func dropUpdated(info: DropInfo) -> DropProposal? {
        DropProposal(operation: .move)
    }
}
