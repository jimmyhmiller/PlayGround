import SwiftUI
import CalorieModel

struct WeightView: View {
    @EnvironmentObject var store: AppStore
    var openWeigh: () -> Void
    private var a: Analysis? { store.analysis }

    private var trendByDay: [Date: Double] {
        var m: [Date: Double] = [:]
        for p in a?.trendSeries ?? [] where p.trend != nil {
            m[Calendar.current.startOfDay(for: p.date)] = p.trend
        }
        return m
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                Text("Weight").font(.system(size: 32, weight: .bold)).pagePadding().padding(.top, 8)

                headline.padding(.top, 24)

                if let series = a?.trendSeries, series.contains(where: { $0.trend != nil }) {
                    ReconciliationChart(series: series).frame(height: 156).padding(.top, 18)
                    legend.pagePadding().padding(.top, 8)
                }

                SectionDivider().pagePadding()
                reconciliation
                logButton.padding(.top, 22)
                SectionDivider().pagePadding()
                recentWeighIns
            }
            .padding(.bottom, 110)
        }
        .scrollIndicators(.hidden)
    }

    private var headline: some View {
        HStack(alignment: .bottom) {
            VStack(alignment: .leading, spacing: 6) {
                CapsLabel(text: "TREND WEIGHT")
                HStack(alignment: .firstTextBaseline, spacing: 5) {
                    Text(a?.trendWeightLb.map { Fmt.f($0, 1) } ?? "—")
                        .font(.mono(54, .heavy)).tracking(-2)
                    Text("lb").font(.system(size: 15)).foregroundStyle(Theme.textDim(0.4))
                }
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text("\(Fmt.signed(a?.totalChangeLb ?? 0)) lb").font(.mono(20, .bold)).foregroundStyle(Theme.green)
                Text("\(Fmt.signed(a?.ratePerWeekLb ?? 0, 2)) lb/wk")
                    .font(.mono(10)).tracking(0.5).foregroundStyle(Theme.textDim(0.4))
            }
        }
        .pagePadding()
    }

    private var legend: some View {
        HStack(spacing: 18) {
            legendItem(color: Theme.green, dashed: false, label: "TREND")
            HStack(spacing: 6) {
                Circle().fill(Theme.textDim(0.4)).frame(width: 6, height: 6)
                Text("SCALE")
            }.font(.mono(10)).foregroundStyle(Theme.textDim(0.5))
            legendItem(color: Theme.amber.opacity(0.85), dashed: true, label: "LOGS-ONLY")
        }
    }

    private func legendItem(color: Color, dashed: Bool, label: String) -> some View {
        HStack(spacing: 6) {
            Rectangle().fill(dashed ? .clear : color)
                .frame(width: 14, height: 2)
                .overlay(dashed ? Rectangle().stroke(color, style: StrokeStyle(lineWidth: 1.5, dash: [3, 2])).frame(height: 0) : nil)
            Text(label)
        }
        .font(.mono(10)).foregroundStyle(Theme.textDim(0.5))
    }

    private var reconciliation: some View {
        VStack(alignment: .leading, spacing: 0) {
            CapsLabel(text: "RECONCILIATION · \(a?.daysCount ?? 0) DAYS").padding(.bottom, 8)
            reconRow("Logs predicted", value: a?.logsPredictedChangeLb.map { "\(Fmt.signed($0)) lb" } ?? "—",
                     color: Theme.amber.opacity(0.95), dashed: true)
            reconRow("Scale shows", value: a?.scaleChangeLb.map { "\(Fmt.signed($0)) lb" } ?? "—",
                     color: Theme.green, dashed: false)
            reconRow("Reconciled deficit", value: deficitText, color: Theme.text, swatch: false)
            if let s = reconSentence {
                Text(s).font(.system(size: 13)).foregroundStyle(Theme.textDim(0.6))
                    .lineSpacing(3).padding(.top, 14)
            }
        }
        .pagePadding()
    }

    private func reconRow(_ label: String, value: String, color: Color, dashed: Bool = false, swatch: Bool = true) -> some View {
        HStack {
            HStack(spacing: 10) {
                if swatch {
                    Rectangle().fill(dashed ? .clear : color).frame(width: 14, height: 2)
                        .overlay(dashed ? Rectangle().stroke(color, style: StrokeStyle(lineWidth: 1.5, dash: [3, 2])).frame(height: 0) : nil)
                }
                Text(label).font(.system(size: 14)).foregroundStyle(Theme.textDim(0.7))
            }
            Spacer()
            Text(value).font(.mono(16, .semibold)).foregroundStyle(color)
        }
        .padding(.vertical, 13)
        .overlay(Rectangle().fill(Theme.hair).frame(height: 0.5), alignment: .top)
    }

    private var deficitText: String {
        let d = a?.cumulativeDeficitKcal ?? 0
        return (d > 0 ? "\u{2212}" : (d < 0 ? "+" : "")) + Fmt.int(abs(d)) + " kcal"
    }

    private var reconSentence: String? {
        guard let logs = a?.logsPredictedChangeLb, let scale = a?.scaleChangeLb else { return nil }
        let gap = abs(logs - scale)
        guard gap > 0.3 else {
            return "Your logs and the scale agree — no systematic logging error detected yet."
        }
        // Logs predicting MORE loss (more negative) than the scale shows ⇒ you ate more than
        // you logged ⇒ under-logging.
        let under = logs < scale
        return "Your logs alone predicted \(Fmt.signed(logs)) lb. The scale moved \(Fmt.signed(scale)) lb — the \(Fmt.f(gap, 1)) lb gap is the \(under ? "under" : "over")-logging the model now corrects for."
    }

    private var logButton: some View {
        Button(action: openWeigh) {
            Text("Log today's weight")
                .font(.system(size: 15, weight: .bold)).foregroundStyle(Theme.onGreen)
                .frame(maxWidth: .infinity).padding(16)
                .background(RoundedRectangle(cornerRadius: 14).fill(Theme.green))
        }
        .pagePadding()
    }

    private var recentWeighIns: some View {
        VStack(alignment: .leading, spacing: 0) {
            CapsLabel(text: "RECENT WEIGH-INS").pagePadding()
            ForEach(Array(store.allWeighIns.prefix(8))) { w in
                let day = Calendar.current.startOfDay(for: w.date)
                let dev = trendByDay[day].map { w.weightLb - $0 }
                HStack(spacing: 14) {
                    Text(dayLabel(w.date)).font(.mono(12)).foregroundStyle(Theme.textDim(0.45)).frame(width: 64, alignment: .leading)
                    Text(devText(dev)).font(.mono(11)).foregroundStyle(devColor(dev))
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Text(Fmt.f(w.weightLb, 1)).font(.mono(17, .semibold))
                }
                .padding(.horizontal, 20).padding(.vertical, 13)
                .overlay(Rectangle().fill(Theme.hairLight).frame(height: 0.5), alignment: .top)
            }
        }
        .padding(.top, 6)
    }

    private func devText(_ dev: Double?) -> String {
        guard let dev else { return "" }
        return "\(Fmt.signed(dev)) vs trend"
    }
    private func devColor(_ dev: Double?) -> Color {
        guard let dev else { return Theme.textDim(0.4) }
        if abs(dev) < 0.5 { return Theme.textDim(0.4) }
        return dev > 0 ? Theme.amber.opacity(0.8) : Theme.green.opacity(0.8)
    }

    private func dayLabel(_ d: Date) -> String {
        let cal = Calendar.current
        if cal.isDateInToday(d) { return "Today" }
        if cal.isDateInYesterday(d) { return "Yesterday" }
        let f = DateFormatter(); f.dateFormat = "EEE d"; return f.string(from: d)
    }
}
