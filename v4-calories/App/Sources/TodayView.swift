import SwiftUI
import CalorieModel

struct TodayView: View {
    @EnvironmentObject var store: AppStore
    var openEntry: () -> Void
    var openSettings: () -> Void
    var onEdit: (CalorieEntry) -> Void

    private var a: Analysis? { store.analysis }
    private var budget: Double { a?.dailyBudgetKcal ?? 2000 }
    private var total: Double { store.todayTotal }
    private var remaining: Double { budget - total }
    private var remainingColor: Color { remaining >= 0 ? Theme.green : Theme.amber }
    private var activeToday: Double? { store.activeByDay[store.today] }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                header
                caloriesIn.padding(.top, 26)
                SectionDivider().pagePadding()
                cumulativeDeficit
                if showModel {
                    SectionDivider().pagePadding()
                    modelSection
                }
                SectionDivider().pagePadding()
                entriesSection
            }
            .padding(.top, 8)
            .padding(.bottom, 110)
        }
        .scrollIndicators(.hidden)
    }

    // MARK: Header

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Today").font(.system(size: 32, weight: .bold))
                Text(dateLabel).font(.mono(11)).tracking(0.5).foregroundStyle(Theme.textDim(0.4))
            }
            Spacer()
            Button(action: openSettings) {
                HStack(spacing: 6) {
                    if store.state.healthKitEnabled {
                        Circle().fill(Theme.green).frame(width: 6, height: 6)
                        Text("HEALTH SYNCED").font(.mono(10)).tracking(1)
                    } else {
                        Image(systemName: "gearshape").font(.system(size: 14))
                    }
                }
                .foregroundStyle(Theme.textDim(0.45))
                .padding(.top, 8)
            }
        }
        .pagePadding()
    }

    private var dateLabel: String {
        let f = DateFormatter(); f.dateFormat = "EEE d MMM"
        let base = f.string(from: Date()).uppercased()
        if let n = a?.daysCount, n > 0 { return "\(base) · DAY \(n)" }
        return base
    }

    // MARK: Calories in

    private var caloriesIn: some View {
        VStack(alignment: .leading, spacing: 0) {
            CapsLabel(text: "CALORIES IN")
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(Fmt.int(total)).font(.mono(64, .heavy)).tracking(-2.5)
                Text("/ \(Fmt.int(budget))").font(.mono(16)).foregroundStyle(Theme.textDim(0.4))
            }
            .padding(.top, 4)
            BarProgress(fraction: total / max(1, budget), color: remainingColor).padding(.top, 18)
            HStack {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(Fmt.int(abs(remaining))).font(.mono(18, .bold)).foregroundStyle(remainingColor)
                    Text("kcal \(remaining >= 0 ? "left" : "over")")
                        .font(.system(size: 12)).foregroundStyle(Theme.textDim(0.45))
                }
                Spacer()
                if let active = activeToday {
                    HStack(spacing: 5) {
                        Text("▲").foregroundStyle(Theme.green)
                        Text("\(Fmt.int(active)) active today")
                    }
                    .font(.mono(11)).foregroundStyle(Theme.textDim(0.5))
                }
            }
            .padding(.top, 10)
        }
        .pagePadding()
    }

    // MARK: Cumulative deficit (the headline)

    private var cumulativeDeficit: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                CapsLabel(text: "CUMULATIVE DEFICIT · \(a?.daysCount ?? 0) DAYS", color: Theme.green.opacity(0.85))
                Spacer()
                if matchesScale {
                    Text("✓ MATCHES SCALE").font(.mono(9)).tracking(0.5)
                        .foregroundStyle(Theme.green.opacity(0.75))
                }
            }
            Text(deficitText).font(.mono(54, .heavy)).tracking(-2).padding(.top, 10)
            Text("KCAL BANKED · ±\(Fmt.int(a?.cumulativeDeficitSE ?? 0))")
                .font(.mono(11)).tracking(0.5).foregroundStyle(Theme.textDim(0.42)).padding(.top, 6)

            if let series = a?.cumulativeDeficitSeries, series.count > 2 {
                DeficitSparkline(series: series).frame(height: 46).padding(.vertical, 12)
            }

            HStack(spacing: 40) {
                StatBlock(label: "PROJECTED LOSS",
                          value: "\(Fmt.signed(-(a?.lossSoFarLb ?? 0))) lb", color: Theme.green)
                StatBlock(label: "VS PLAN",
                          value: "\(Fmt.signed(a?.aheadOfPlanLb ?? 0)) lb",
                          color: (a?.aheadOfPlanLb ?? 0) >= 0 ? Theme.green : Theme.amber)
            }
            .padding(.top, 12)

            Text(planSentence)
                .font(.system(size: 13)).foregroundStyle(Theme.textDim(0.6))
                .lineSpacing(3).padding(.top, 16)
        }
        .pagePadding()
    }

    private var matchesScale: Bool { (a?.daysSinceLastWeighIn ?? 99) <= 1 }

    private var deficitText: String {
        let d = a?.cumulativeDeficitKcal ?? 0
        if d > 0 { return "\u{2212}" + Fmt.int(d) }      // banked deficit shows as removed calories
        if d < 0 { return "+" + Fmt.int(-d) }
        return "0"
    }

    private var planSentence: String {
        guard let a, a.daysCount > 0 else {
            return "Log a few days and weigh in — the scale anchors your real deficit and the running total never resets."
        }
        let banked = Fmt.int(max(0, a.cumulativeDeficitKcal))
        let ahead = a.aheadOfPlanLb
        let dir = ahead >= 0 ? "ahead of" : "behind"
        return "Banked \(banked) kcal over \(a.daysCount) days — about \(Fmt.signed(ahead)) lb \(dir) your \(Fmt.f(store.state.goal.ratePerWeek, 2)) lb/wk plan. Miss a day and the scale fills the gap; the running total never resets."
    }

    // MARK: The model

    private var showModel: Bool { a?.effectiveTDEE != nil || a?.trueTDEE != nil }

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            CapsLabel(text: "THE MODEL · LEARNED FROM YOUR SCALE").padding(.bottom, 18)

            let tdee = a?.trueTDEE ?? a?.effectiveTDEE
            if let tdee {
                HStack(alignment: .firstTextBaseline) {
                    Text(a?.trueTDEE != nil ? "Real maintenance · TDEE" : "Maintenance · as logged")
                        .font(.system(size: 14)).foregroundStyle(Theme.textDim(0.75))
                    Spacer()
                    HStack(spacing: 4) {
                        Text(Fmt.int(tdee.value)).font(.mono(16, .semibold))
                        Text("±\(Fmt.int(tdee.standardError))").font(.mono(12)).foregroundStyle(Theme.textDim(0.4))
                    }
                }
                TDEEBandBar(value: tdee.value, se: tdee.standardError).padding(.top, 10)
            }

            if let bias = a?.loggingBias {
                let pct = (bias.value - 1) * 100
                let under = pct >= 0
                HStack(alignment: .firstTextBaseline) {
                    Text(under ? "You under-log by" : "You over-log by")
                        .font(.system(size: 14)).foregroundStyle(Theme.textDim(0.75))
                    Spacer()
                    Text("\(Fmt.signed(pct, 0))%  ±\(Fmt.int(bias.standardError * 100))")
                        .font(.mono(16, .semibold)).foregroundStyle(Theme.amber)
                }
                .padding(.top, 24)
                Text("True intake runs ~\(Fmt.int(abs(pct)))% \(under ? "above" : "below") what you enter. Your budget already corrects for it.")
                    .font(.system(size: 12)).foregroundStyle(Theme.textDim(0.45))
                    .lineSpacing(2).padding(.top, 8)
            } else {
                Text("Connect Apple Health (active + resting energy) to learn whether you systematically under- or over-log.")
                    .font(.system(size: 12)).foregroundStyle(Theme.textDim(0.45))
                    .lineSpacing(2).padding(.top, 16)
            }
        }
        .pagePadding()
    }

    // MARK: Entries

    private var entriesSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                CapsLabel(text: entryCountLabel)
                Spacer()
                Button("+ Add", action: openEntry).font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Theme.green)
            }
            .pagePadding()

            ForEach(store.todayEntries) { e in
                HStack(spacing: 14) {
                    Text(timeText(e.date)).font(.mono(11)).foregroundStyle(Theme.textDim(0.35)).frame(width: 42, alignment: .leading)
                    Text(e.label?.isEmpty == false ? e.label! : "—")
                        .font(.system(size: 14)).foregroundStyle(Theme.textDim(0.85))
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Text(Fmt.int(e.kcal)).font(.mono(17, .semibold))
                    Image(systemName: "chevron.right").font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Theme.textDim(0.25))
                }
                .padding(.horizontal, 20).padding(.vertical, 15)
                .contentShape(Rectangle())
                .overlay(Rectangle().fill(Theme.hairLight).frame(height: 0.5), alignment: .top)
                .onTapGesture { onEdit(e) }
            }

            Button(action: openEntry) {
                HStack(spacing: 10) {
                    Image(systemName: "plus")
                        .font(.system(size: 14)).foregroundStyle(Theme.green)
                        .frame(width: 22, height: 22)
                        .overlay(Circle().stroke(Theme.green.opacity(0.5), lineWidth: 1))
                    Text("Log calories").font(.system(size: 14, weight: .medium)).foregroundStyle(Theme.green)
                    Spacer()
                }
                .padding(.horizontal, 20).padding(.vertical, 15)
                .overlay(Rectangle().fill(Theme.hairLight).frame(height: 0.5), alignment: .top)
            }
        }
        .padding(.top, 6)
    }

    private var entryCountLabel: String {
        let c = store.todayEntries.count
        return "\(c) \(c == 1 ? "ENTRY" : "ENTRIES") · \(Fmt.int(total)) KCAL"
    }

    private func timeText(_ d: Date) -> String {
        let f = DateFormatter(); f.dateFormat = "HH:mm"; return f.string(from: d)
    }
}

/// The small TDEE confidence bar with a marker, like the design.
struct TDEEBandBar: View {
    let value: Double
    let se: Double
    private let axMin = 1600.0, axMax = 3200.0
    private func pct(_ v: Double) -> Double { (v - axMin) / (axMax - axMin) }
    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            ZStack(alignment: .leading) {
                Capsule().fill(Color.white.opacity(0.06))
                Capsule().fill(Theme.green.opacity(0.25))
                    .frame(width: max(2, (pct(value + 2*se) - pct(value - 2*se)) * w))
                    .offset(x: max(0, pct(value - 2*se) * w))
                RoundedRectangle(cornerRadius: 1).fill(Theme.green)
                    .frame(width: 2, height: 11)
                    .offset(x: min(w - 2, max(0, pct(value) * w)), y: -3)
            }
        }
        .frame(height: 5)
    }
}
