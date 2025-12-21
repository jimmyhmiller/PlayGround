import SwiftUI

struct AddGoalView: View {
    @EnvironmentObject var viewModel: EaseViewModel

    @State private var name: String = ""
    @State private var selectedColorHex: String = ""
    @FocusState private var isNameFocused: Bool

    private var canAdd: Bool {
        !name.trimmingCharacters(in: .whitespaces).isEmpty
    }

    private var firstUnusedColor: String {
        let usedColors = Set(viewModel.goals.map { $0.colorHex })
        return Goal.presetColors.first { !usedColors.contains($0) } ?? Goal.presetColors[0]
    }

    private func addGoal() {
        if canAdd {
            viewModel.addGoal(name: name.trimmingCharacters(in: .whitespaces), colorHex: selectedColorHex)
            viewModel.isAddingGoal = false
        }
    }

    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Button {
                    viewModel.isAddingGoal = false
                } label: {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 12, weight: .semibold))
                }
                .buttonStyle(.plain)

                Spacer()

                Text("New Goal")
                    .font(.headline)

                Spacer()

                // Invisible spacer for centering
                Image(systemName: "chevron.left")
                    .font(.system(size: 12, weight: .semibold))
                    .opacity(0)
            }

            TextField("Goal name", text: $name)
                .textFieldStyle(.roundedBorder)
                .focused($isNameFocused)
                .onSubmit {
                    addGoal()
                }
                .onAppear {
                    if selectedColorHex.isEmpty {
                        selectedColorHex = firstUnusedColor
                    }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                        isNameFocused = true
                    }
                }

            VStack(alignment: .leading, spacing: 8) {
                Text("Color")
                    .font(.caption)
                    .foregroundColor(.secondary)

                LazyVGrid(columns: Array(repeating: GridItem(.fixed(28), spacing: 8), count: 5), spacing: 8) {
                    ForEach(Goal.presetColors, id: \.self) { hex in
                        Circle()
                            .fill(Color(hex: hex))
                            .frame(width: 24, height: 24)
                            .overlay(
                                Circle()
                                    .stroke(Color.primary, lineWidth: selectedColorHex == hex ? 2 : 0)
                            )
                            .onTapGesture {
                                selectedColorHex = hex
                            }
                    }
                }
            }

            HStack {
                Button("Cancel") {
                    viewModel.isAddingGoal = false
                }

                Spacer()

                Button("Add") {
                    addGoal()
                }
                .disabled(!canAdd)
            }
        }
    }
}
