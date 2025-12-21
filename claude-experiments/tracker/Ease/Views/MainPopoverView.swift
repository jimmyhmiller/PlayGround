import SwiftUI

struct MainPopoverView: View {
    @EnvironmentObject var viewModel: EaseViewModel

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
    }

    private var mainContent: some View {
        VStack(spacing: 16) {
            // Time period picker
            TimePeriodPicker(selection: $viewModel.selectedPeriod)

            // Proportional bars visualization
            if !viewModel.goals.isEmpty {
                ProportionalBarsView()
                    .padding(.horizontal, 8)

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
