import SwiftUI

struct GoalRowView: View {
    let goal: Goal
    var isReorderMode: Bool = false
    let onAdd: (Double) -> Void
    let onDelete: () -> Void
    let onColorChange: (String) -> Void

    @State private var dragOffset: CGFloat = 0
    @GestureState private var isPressed: Bool = false
    @State private var showingColorPicker: Bool = false
    @State private var isAltPressed: Bool = false
    @State private var eventMonitor: Any?

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
            Image(systemName: "line.3.horizontal")
                .font(.system(size: 12))
                .foregroundColor(.secondary)
                .frame(width: 16)
                .opacity(isReorderMode ? 1 : 0)

            Circle()
                .fill(goal.color)
                .frame(width: 12, height: 12)
                .onTapGesture {
                    if !isReorderMode {
                        showingColorPicker.toggle()
                    }
                }
                .popover(isPresented: $showingColorPicker) {
                    colorPickerContent
                }

            Text(goal.name)
                .font(.system(size: 13))

            Spacer()

            // Pull indicator - always present for consistent layout
            ZStack {
                if isPressed {
                    // Visual feedback during press/drag
                    RoundedRectangle(cornerRadius: 4)
                        .fill(isAltPressed ? Color(hex: "#E57373").opacity(0.3) : goal.color.opacity(0.3))
                        .frame(width: 40, height: 8 + dragProgress * 20)
                } else {
                    Image(systemName: isAltPressed ? "arrow.down.circle" : "arrow.up.circle")
                        .font(.system(size: 14))
                        .foregroundColor(isAltPressed ? Color(hex: "#E57373") : .secondary)
                }
            }
            .frame(width: 40, height: 28)
            .opacity(isReorderMode ? 0 : 1)

            Button {
                onDelete()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 14))
                    .foregroundColor(.secondary.opacity(0.5))
            }
            .buttonStyle(.plain)
            .opacity(isReorderMode ? 0 : (isPressed ? 0 : 1))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .contentShape(Rectangle())
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isReorderMode ? goal.color.opacity(0.05) : (isPressed ? goal.color.opacity(0.1) : Color.clear))
        )
        .gesture(
            isReorderMode ? nil : DragGesture(minimumDistance: 0)
                .updating($isPressed) { _, state, _ in
                    state = true
                }
                .onChanged { value in
                    // Check for Alt key
                    isAltPressed = NSEvent.modifierFlags.contains(.option)

                    // Track drag in the appropriate direction
                    if isAltPressed {
                        // Alt held: drag down to decrease
                        if value.translation.height > 0 {
                            dragOffset = value.translation.height
                        }
                    } else {
                        // Normal: drag up to increase
                        if value.translation.height < 0 {
                            dragOffset = -value.translation.height
                        }
                    }
                }
                .onEnded { value in
                    // Always add at least minimum amount on any click/drag
                    let amount = dragOffset > 20 ? amountToAdd : minAmountPerDrag
                    if isAltPressed {
                        onAdd(-amount)
                    } else {
                        onAdd(amount)
                    }
                    withAnimation(.easeOut(duration: 0.2)) {
                        dragOffset = 0
                    }
                }
        )
        .onAppear {
            isAltPressed = NSEvent.modifierFlags.contains(.option)
            eventMonitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { event in
                isAltPressed = event.modifierFlags.contains(.option)
                return event
            }
        }
        .onDisappear {
            isAltPressed = false
            if let monitor = eventMonitor {
                NSEvent.removeMonitor(monitor)
                eventMonitor = nil
            }
        }
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
