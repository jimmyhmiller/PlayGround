import SwiftUI
import CalorieModel

struct TrendsView: View {
    @EnvironmentObject var store: AppStore
    private var a: Analysis? { store.analysis }
    private var goal: Goal { store.state.goal }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                Text("Trends").font(.system(size: 32, weight: .bold)).pagePadding().padding(.top, 8)
                goalSection.padding(.top, 24)

                SectionDivider().pagePadding()
                chartSection(title: "CALORIE BANK · VS GOAL PACE",
                             value: bankText, valueColor: (a?.earnedCaloriesKcal ?? 0) >= 0 ? Theme.green : Theme.amber) {
                    if let s = a?.earnedCaloriesSeries, s.count > 2 {
                        DeficitAreaChart(series: s).frame(height: 120)
                    }
                }

                SectionDivider().pagePadding()
                chartSection(title: "TDEE ESTIMATE · CONVERGING",
                             value: tdeeText, valueColor: Theme.text) {
                    if let s = a?.effectiveTDEESeries, s.count > 2 {
                        TDEEBandChart(series: s).frame(height: 96)
                    }
                    Text("Confidence narrows as the scale confirms each week.")
                        .font(.system(size: 12)).foregroundStyle(Theme.textDim(0.45))
                        .pagePadding().padding(.top, 8)
                }

                if let bias = a?.biasSeries, bias.count > 2 {
                    SectionDivider().pagePadding()
                    chartSection(title: "LOGGING BIAS",
                                 value: "\(Fmt.signed((a?.loggingBias.map { ($0.value - 1) * 100 }) ?? 0, 0))%",
                                 valueColor: Theme.amber) {
                        BiasChart(series: bias).frame(height: 80)
                    }
                }
            }
            .padding(.bottom, 110)
        }
        .scrollIndicators(.hidden)
    }

    // MARK: Goal

    private var goalSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                CapsLabel(text: "GOAL · \(Fmt.int(goal.targetWeightLb)) LB")
                Spacer()
                Text(etaText).font(.mono(11)).foregroundStyle(Theme.textDim(0.55))
            }
            HStack(alignment: .firstTextBaseline) {
                Text(a?.startWeightLb.map { Fmt.int($0) } ?? "—").font(.mono(13)).foregroundStyle(Theme.textDim(0.45))
                Spacer()
                Text(a?.trendWeightLb.map { Fmt.f($0, 1) } ?? "—").font(.mono(22, .heavy))
                Spacer()
                Text(Fmt.int(goal.targetWeightLb)).font(.mono(13)).foregroundStyle(Theme.green)
            }
            .padding(.vertical, 12)

            GoalBar(fraction: a?.goalProgress ?? 0)

            Text(toGoText).font(.system(size: 13)).foregroundStyle(Theme.textDim(0.55)).padding(.top, 12)
        }
        .pagePadding()
    }

    private var etaText: String {
        guard let eta = a?.etaDate else { return "ETA —" }
        let f = DateFormatter(); f.dateFormat = "MMM d"
        return "ETA " + f.string(from: eta).uppercased()
    }

    private var toGoText: String {
        guard let toGo = a?.toGoLb else { return "Set a goal to see your projection." }
        let rate = a?.ratePerWeekLb ?? 0
        return "\(Fmt.f(abs(toGo), 1)) lb to target · holding \(Fmt.signed(rate, 2)) lb/wk"
    }

    private var bankText: String {
        let d = a?.earnedCaloriesKcal ?? 0
        return (d > 0 ? "+" : (d < 0 ? "\u{2212}" : "")) + Fmt.int(abs(d))
    }

    private var tdeeText: String {
        guard let e = a?.trueTDEE ?? a?.effectiveTDEE else { return "—" }
        return "\(Fmt.int(e.value)) ±\(Fmt.int(e.standardError))"
    }

    // MARK: Section scaffold

    @ViewBuilder
    private func chartSection<Content: View>(title: String, value: String, valueColor: Color,
                                             @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                CapsLabel(text: title)
                Spacer()
                Text(value).font(.mono(16, .bold)).foregroundStyle(valueColor)
            }
            .pagePadding()
            content()
        }
    }
}

/// Goal progress bar with a position marker.
struct GoalBar: View {
    let fraction: Double
    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let f = max(0, min(1, fraction))
            ZStack(alignment: .leading) {
                Capsule().fill(Color.white.opacity(0.07))
                Capsule().fill(Theme.green).frame(width: f * w)
                RoundedRectangle(cornerRadius: 1).fill(.white)
                    .frame(width: 2, height: 12)
                    .offset(x: min(w - 2, f * w), y: -4)
            }
        }
        .frame(height: 4)
    }
}
