import SwiftUI
import Charts
import CalorieModel

/// Calories burned, from Apple Health (active move energy + resting/basal energy).
struct BurnView: View {
    @EnvironmentObject var store: AppStore
    var openSettings: () -> Void

    private var today: AppStore.BurnDay { store.todayBurn }
    private var series: [AppStore.BurnDay] { store.burnSeries() }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                header.padding(.top, 8)
                sections
            }
            .padding(.bottom, 120)
        }
        .scrollIndicators(.hidden)
        .task { if store.state.healthKitEnabled { await store.syncHealth() } }
    }

    @ViewBuilder private var sections: some View {
        if store.hasBurnData {
            burnedToday.padding(.top, 24)
            SectionDivider().pagePadding()
            netToday
            SectionDivider().pagePadding()
            history
            SectionDivider().pagePadding()
            averages
        } else {
            emptyState
        }
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Burn").font(.system(size: 32, weight: .bold))
                Text(dateLabel).font(.mono(11)).tracking(0.5).foregroundStyle(Theme.textDim(0.4))
            }
            Spacer()
            if store.state.healthKitEnabled {
                HStack(spacing: 6) {
                    Circle().fill(Theme.green).frame(width: 6, height: 6)
                    Text("HEALTH SYNCED").font(.mono(10)).tracking(1)
                }
                .foregroundStyle(Theme.textDim(0.45)).padding(.top, 8)
            }
        }
        .pagePadding()
    }

    private var dateLabel: String {
        let f = DateFormatter(); f.dateFormat = "EEE d MMM"; return f.string(from: Date()).uppercased()
    }

    private var burnedToday: some View {
        let total = today.total ?? 0
        let activeFrac = total > 0 ? (today.active ?? 0) / total : 0
        return VStack(alignment: .leading, spacing: 0) {
            CapsLabel(text: "BURNED TODAY")
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(today.total.map { Fmt.int($0) } ?? "—").font(.mono(64, .heavy)).tracking(-2.5)
                Text("kcal").font(.mono(16)).foregroundStyle(Theme.textDim(0.4))
            }
            .padding(.top, 4)
            BarProgress(fraction: activeFrac, color: Theme.green, height: 6).padding(.top, 18)
            HStack(spacing: 28) {
                legendValue(color: Theme.green, label: "ACTIVE", value: today.active)
                legendValue(color: Theme.textDim(0.3), label: "RESTING", value: today.resting)
            }
            .padding(.top, 14)
        }
        .pagePadding()
    }

    private func legendValue(color: Color, label: String, value: Double?) -> some View {
        HStack(spacing: 8) {
            Circle().fill(color).frame(width: 7, height: 7)
            VStack(alignment: .leading, spacing: 2) {
                Text(label).font(.mono(9)).tracking(1).foregroundStyle(Theme.textDim(0.4))
                Text(value.map { Fmt.int($0) } ?? "—").font(.mono(18, .bold))
            }
        }
    }

    private var netToday: some View {
        VStack(alignment: .leading, spacing: 0) {
            CapsLabel(text: "TODAY · IN VS OUT").padding(.bottom, 4)
            burnRow("Eaten", Fmt.int(store.todayTotal), Theme.text)
            burnRow("Burned", today.total.map { Fmt.int($0) } ?? "—", Theme.green)
            netRow
        }
        .pagePadding()
    }

    @ViewBuilder private var netRow: some View {
        if let burned = today.total {
            let net = burned - store.todayTotal
            burnRow(net >= 0 ? "Net deficit" : "Net surplus",
                    "\(Fmt.signedInt(net)) kcal", net >= 0 ? Theme.green : Theme.amber, bold: true)
        }
    }

    private func burnRow(_ label: String, _ value: String, _ color: Color, bold: Bool = false) -> some View {
        HStack {
            Text(label).font(.system(size: 14)).foregroundStyle(Theme.textDim(0.7))
            Spacer()
            Text(value).font(.mono(16, bold ? .bold : .semibold)).foregroundStyle(color)
        }
        .padding(.vertical, 12)
        .overlay(Rectangle().fill(Theme.hair).frame(height: 0.5), alignment: .top)
    }

    private var history: some View {
        let last = Array(series.suffix(14))
        return VStack(alignment: .leading, spacing: 10) {
            CapsLabel(text: "DAILY BURN · LAST \(last.count) DAYS").pagePadding()
            Chart(last) { d in
                if let r = d.resting {
                    BarMark(x: .value("day", d.date, unit: .day), y: .value("kcal", r))
                        .foregroundStyle(Theme.textDim(0.2))
                }
                if let a = d.active {
                    BarMark(x: .value("day", d.date, unit: .day), y: .value("kcal", a))
                        .foregroundStyle(Theme.green)
                }
            }
            .chartXAxis(.hidden)
            .chartYAxis {
                AxisMarks(position: .trailing, values: .automatic(desiredCount: 3)) { _ in
                    AxisGridLine().foregroundStyle(Color.white.opacity(0.05))
                    AxisValueLabel().foregroundStyle(Theme.textDim(0.35)).font(.mono(9))
                }
            }
            .frame(height: 130).padding(.horizontal, 14)
            HStack(spacing: 18) {
                legendChip(Theme.green, "ACTIVE")
                legendChip(Theme.textDim(0.2), "RESTING")
            }
            .pagePadding()
        }
    }

    private func legendChip(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 2).fill(color).frame(width: 10, height: 10)
            Text(label).font(.mono(10)).foregroundStyle(Theme.textDim(0.5))
        }
    }

    private var averages: some View {
        let totals = series.compactMap(\.total)
        let actives = series.compactMap(\.active)
        let avgTotal = totals.isEmpty ? nil : totals.reduce(0, +) / Double(totals.count)
        let avgActive = actives.isEmpty ? nil : actives.reduce(0, +) / Double(actives.count)
        return HStack(spacing: 40) {
            StatBlock(label: "AVG DAILY BURN", value: avgTotal.map { Fmt.int($0) } ?? "—")
            StatBlock(label: "AVG ACTIVE", value: avgActive.map { Fmt.int($0) } ?? "—", color: Theme.green)
        }
        .pagePadding()
    }

    private var emptyState: some View {
        let connectedNoEnergy = store.state.healthKitEnabled && (store.lastSync?.hasEnergy == false)
        return VStack(spacing: 14) {
            Image(systemName: "flame").font(.system(size: 40)).foregroundStyle(Theme.textDim(0.3))
            Text(connectedNoEnergy ? "Health is on, but no energy yet" : "No burn data yet")
                .font(.system(size: 17, weight: .semibold))
            Text(connectedNoEnergy
                 ? "Active & Resting Energy read access is most likely off — it defaults to OFF in Health's permission sheet. Open Settings to turn it on and re-sync."
                 : "Connect Apple Health to see your active and resting energy here. The app uses it to learn your true calorie burn.")
                .multilineTextAlignment(.center).font(.system(size: 14)).foregroundStyle(Theme.textDim(0.5))
            Button(action: openSettings) {
                Text(connectedNoEnergy ? "Fix in settings" : "Open settings")
                    .font(.system(size: 15, weight: .bold)).foregroundStyle(Theme.onGreen)
                    .padding(.horizontal, 24).padding(.vertical, 14)
                    .background(RoundedRectangle(cornerRadius: 14).fill(Theme.green))
            }
            .padding(.top, 6)
        }
        .frame(maxWidth: .infinity).padding(.top, 80).padding(.horizontal, 30)
    }
}
