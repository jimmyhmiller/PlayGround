import SwiftUI
import CalorieModel

enum Tab { case today, weight, burn, trends }

struct RootView: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.scenePhase) private var scenePhase
    @State private var tab: Tab = .today
    @State private var entryOpen = false
    @State private var weighOpen = false
    @State private var settingsOpen = false
    @State private var editingEntry: CalorieEntry?

    private var hasData: Bool {
        !store.state.entries.isEmpty || !store.allWeighIns.isEmpty || store.state.hasOnboarded
    }

    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()
            if !hasData {
                OnboardingView()
            } else {
                content
                    .safeAreaInset(edge: .bottom, spacing: 0) { tabBar }
            }
        }
        .sheet(isPresented: $entryOpen) { EntrySheet().presentationDetents([.large]) }
        .sheet(item: $editingEntry) { EntrySheet(editing: $0).presentationDetents([.large]) }
        .sheet(isPresented: $weighOpen) { WeighInSheet().presentationDetents([.medium, .large]) }
        .sheet(isPresented: $settingsOpen) { SettingsView() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active, store.state.healthKitEnabled { Task { await store.syncHealth() } }
        }
        .onAppear {
            let args = ProcessInfo.processInfo.arguments
            if args.contains("--tab=weight") { tab = .weight }
            else if args.contains("--tab=burn") { tab = .burn }
            else if args.contains("--tab=trends") { tab = .trends }
            if args.contains("--entry") { entryOpen = true }
        }
    }

    @ViewBuilder private var content: some View {
        switch tab {
        case .today:  TodayView(openEntry: { entryOpen = true },
                                openSettings: { settingsOpen = true },
                                onEdit: { editingEntry = $0 })
        case .weight: WeightView(openWeigh: { weighOpen = true })
        case .burn:   BurnView(openSettings: { settingsOpen = true })
        case .trends: TrendsView()
        }
    }

    private var tabBar: some View {
        HStack(spacing: 0) {
            tabButton(.today, system: "circle.circle", label: "Today")
            tabButton(.weight, system: "scalemass", label: "Weight")
            Spacer().frame(width: 64)
            tabButton(.burn, system: "flame", label: "Burn")
            tabButton(.trends, system: "chart.bar.fill", label: "Trends")
        }
        .overlay(alignment: .center) {
            Button { entryOpen = true } label: {
                Image(systemName: "plus")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundStyle(Theme.onGreen)
                    .frame(width: 54, height: 54)
                    .background(Circle().fill(Theme.green))
                    .shadow(color: Theme.green.opacity(0.35), radius: 11, y: 6)
            }
            .offset(y: -2)
        }
        .padding(.horizontal, 20)
        .padding(.top, 10)
        .background(
            LinearGradient(colors: [Theme.bg.opacity(0), Theme.bg],
                           startPoint: .top, endPoint: .init(x: 0.5, y: 0.4))
                .ignoresSafeArea()
        )
    }

    private func tabButton(_ t: Tab, system: String, label: String) -> some View {
        let active = tab == t
        return Button {
            withAnimation(.easeInOut(duration: 0.15)) { tab = t }
        } label: {
            VStack(spacing: 5) {
                Image(systemName: system).font(.system(size: 18, weight: .medium))
                Text(label).font(.system(size: 10, weight: .semibold))
            }
            .foregroundStyle(active ? Theme.green : Theme.textDim(0.4))
            .frame(maxWidth: .infinity)
        }
    }
}

/// Shown until the user has any data — choose demo or a fresh start.
struct OnboardingView: View {
    @EnvironmentObject var store: AppStore
    var body: some View {
        VStack(spacing: 14) {
            Spacer()
            Text("Cumulative")
                .font(.system(size: 40, weight: .bold)).foregroundStyle(Theme.text)
            Text("Calories and weight, reconciled.\nThe scale fills the gaps.")
                .multilineTextAlignment(.center)
                .font(.system(size: 15)).foregroundStyle(Theme.textDim(0.55))
            Spacer()
            Button { store.loadDemoData() } label: {
                Text("Explore with demo data")
                    .font(.system(size: 16, weight: .bold)).foregroundStyle(Theme.onGreen)
                    .frame(maxWidth: .infinity).padding(16)
                    .background(RoundedRectangle(cornerRadius: 14).fill(Theme.green))
            }
            Button { store.markOnboarded() } label: {
                Text("Start fresh")
                    .font(.system(size: 16, weight: .semibold)).foregroundStyle(Theme.text)
                    .frame(maxWidth: .infinity).padding(16)
                    .background(RoundedRectangle(cornerRadius: 14).fill(Theme.key))
            }
        }
        .padding(24)
    }
}
