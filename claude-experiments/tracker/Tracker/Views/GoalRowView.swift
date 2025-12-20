import SwiftUI

struct GoalRowView: View {
    let goal: Goal
    let onAdd: (Double) -> Void
    let onDelete: () -> Void
    let onColorChange: (String) -> Void

    @State private var dragOffset: CGFloat = 0
    @GestureState private var isPressed: Bool = false
    @State private var showingColorPicker: Bool = false

    private let maxDragDistance: CGFloat = 100
    private let minAmountPerDrag: Double = 1.0
    private let maxAmountPerDrag: Double = 10.0

    private var dragProgress: CGFloat {
        min(1.0, max(0, dragOffset / maxDragDistance))
    }

    private var amountToAdd: Double {
        let progress = Double(dragProgress)
        return minAmountPerDrag + (maxAmountPerDrag - minAmountPerDrag) * progress
    }

    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(goal.color)
                .frame(width: 12, height: 12)
                .onTapGesture {
                    showingColorPicker.toggle()
                }
                .popover(isPresented: $showingColorPicker) {
                    colorPickerContent
                }

            Text(goal.name)
                .font(.system(size: 13))

            Spacer()

            // Pull indicator
            ZStack {
                if isPressed {
                    // Visual feedback during press/drag
                    RoundedRectangle(cornerRadius: 4)
                        .fill(goal.color.opacity(0.3))
                        .frame(width: 40, height: 8 + dragProgress * 20)
                } else {
                    Image(systemName: "arrow.up.circle")
                        .font(.system(size: 14))
                        .foregroundColor(.secondary)
                }
            }
            .frame(width: 40, height: 28)

            Button {
                onDelete()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 14))
                    .foregroundColor(.secondary.opacity(0.5))
            }
            .buttonStyle(.plain)
            .opacity(isPressed ? 0 : 1)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .contentShape(Rectangle())
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isPressed ? goal.color.opacity(0.1) : Color.clear)
        )
        .gesture(
            DragGesture(minimumDistance: 0)
                .updating($isPressed) { _, state, _ in
                    state = true
                }
                .onChanged { value in
                    // Only track upward drag
                    if value.translation.height < 0 {
                        dragOffset = -value.translation.height
                    }
                }
                .onEnded { value in
                    if dragOffset > 20 { // Minimum threshold
                        onAdd(amountToAdd)
                    }
                    withAnimation(.easeOut(duration: 0.2)) {
                        dragOffset = 0
                    }
                }
        )
    }

    private var colorPickerContent: some View {
        VStack(spacing: 8) {
            LazyVGrid(columns: Array(repeating: GridItem(.fixed(28), spacing: 8), count: 5), spacing: 8) {
                ForEach(Goal.presetColors, id: \.self) { hex in
                    Circle()
                        .fill(Color(hex: hex))
                        .frame(width: 24, height: 24)
                        .overlay(
                            Circle()
                                .stroke(Color.primary, lineWidth: goal.colorHex == hex ? 2 : 0)
                        )
                        .onTapGesture {
                            onColorChange(hex)
                            showingColorPicker = false
                        }
                }
            }
        }
        .padding()
    }
}
