import SwiftUI

struct TimePeriodPicker: View {
    @Binding var selection: TimePeriod
    var animate: Bool = false

    var body: some View {
        HStack(spacing: 2) {
            ForEach(TimePeriod.allCases, id: \.self) { period in
                Button {
                    if animate {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            selection = period
                        }
                    } else {
                        selection = period
                    }
                } label: {
                    Text(period.rawValue)
                        .font(.caption)
                        .fontWeight(selection == period ? .semibold : .regular)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(
                            RoundedRectangle(cornerRadius: 4)
                                .fill(selection == period ? Color.accentColor.opacity(0.2) : Color.clear)
                        )
                }
                .buttonStyle(.plain)
            }
        }
        .padding(4)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(Color.gray.opacity(0.1))
        )
    }
}
