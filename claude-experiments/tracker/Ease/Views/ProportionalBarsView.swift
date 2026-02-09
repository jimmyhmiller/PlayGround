import SwiftUI

struct ProportionalBarsView: View {
    @EnvironmentObject var viewModel: EaseViewModel

    var body: some View {
        let proportions = viewModel.proportions(for: viewModel.selectedPeriod)

        VStack(spacing: 4) {
            ForEach(proportions, id: \.goal.id) { item in
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        // Background slot
                        RoundedRectangle(cornerRadius: 4)
                            .fill(item.goal.color.opacity(0.15))
                            .frame(width: geo.size.width)

                        // Filled bar
                        RoundedRectangle(cornerRadius: 4)
                            .fill(item.goal.color)
                            .frame(width: item.proportion > 0 ? max(8, geo.size.width * item.proportion) : 0)
                    }
                }
                .frame(height: 12)
                .help(item.goal.name)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture {
            withAnimation(.easeInOut(duration: 0.2)) {
                viewModel.showCalendarView = true
            }
        }
    }
}
