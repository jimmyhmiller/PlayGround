import SwiftUI
import CalorieModel

struct SettingsView: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.dismiss) private var dismiss
    @Environment(\.openURL) private var openURL
    @State private var goal: Goal = Goal(targetWeightLb: 178, ratePerWeek: 0.85)
    @State private var hasProfile = false
    @State private var newShortcutName = ""
    @State private var newShortcutKcal = ""
    @State private var aiKeyDraft = ""

    var body: some View {
        NavigationStack {
            Form {
                goalSection
                profileSection
                healthSection
                assistantSection
                shortcutsSection
                dataSection
            }
            .scrollContentBackground(.hidden)
            .background(Theme.bg)
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
        .preferredColorScheme(.dark)
        .tint(Theme.green)
        .onAppear {
            goal = store.state.goal
            hasProfile = goal.sex != nil
        }
        .onChange(of: goal) { _, newGoal in store.updateGoal(newGoal) }
    }

    private var goalSection: some View {
        Section("Goal") {
            Stepper(value: $goal.targetWeightLb, in: 80...400, step: 1) {
                LabeledContent("Target weight", value: "\(Fmt.int(goal.targetWeightLb)) lb")
            }
            VStack(alignment: .leading) {
                LabeledContent("Loss rate", value: "\(Fmt.f(goal.ratePerWeek, 2)) lb/wk")
                Slider(value: $goal.ratePerWeek, in: 0...2, step: 0.05)
            }
            Text("Your daily budget is derived from this rate and your learned maintenance, in your own logged units.")
                .font(.footnote).foregroundStyle(Theme.textDim(0.5))
        }
    }

    private var profileSection: some View {
        Section("Body (improves TDEE & bias without an Apple Watch)") {
            Toggle("Use a body profile", isOn: $hasProfile)
                .onChange(of: hasProfile) { _, on in
                    if on {
                        goal.sex = goal.sex ?? .male
                        goal.ageYears = goal.ageYears ?? 38
                        goal.heightCm = goal.heightCm ?? 178
                    } else {
                        goal.sex = nil; goal.ageYears = nil; goal.heightCm = nil
                    }
                }
            if hasProfile {
                Picker("Sex", selection: Binding(
                    get: { goal.sex ?? .male },
                    set: { goal.sex = $0 })) {
                    Text("Male").tag(Goal.Sex.male)
                    Text("Female").tag(Goal.Sex.female)
                }
                Stepper(value: Binding(get: { goal.ageYears ?? 38 }, set: { goal.ageYears = $0 }),
                        in: 14...100, step: 1) {
                    LabeledContent("Age", value: "\(Int(goal.ageYears ?? 38))")
                }
                Stepper(value: Binding(get: { goal.heightCm ?? 178 }, set: { goal.heightCm = $0 }),
                        in: 120...220, step: 1) {
                    LabeledContent("Height", value: "\(Int(goal.heightCm ?? 178)) cm")
                }
            }
        }
    }

    private var healthSection: some View {
        Section("Apple Health") {
            Toggle("Sync activity & weight", isOn: Binding(
                get: { store.state.healthKitEnabled },
                set: { store.setHealthKitEnabled($0) }))
            Text("Reads active + resting energy (your true burn) and imports smart-scale weigh-ins. The app works fully without it.")
                .font(.footnote).foregroundStyle(Theme.textDim(0.5))

            if store.state.healthKitEnabled {
                Button {
                    Task { await store.syncHealth() }
                } label: {
                    HStack {
                        Text("Sync now")
                        Spacer()
                        if store.isSyncing { ProgressView() }
                    }
                }
                .disabled(store.isSyncing)

                if let r = store.lastSync { healthDiagnostics(r) }
            }
        }
    }

    @ViewBuilder
    private func healthDiagnostics(_ r: AppStore.SyncReport) -> some View {
        let dim = Theme.textDim(0.55)
        VStack(alignment: .leading, spacing: 6) {
            diagRow("Active energy", "\(r.activeDays) days", r.activeDays > 0)
            diagRow("Resting energy", "\(r.basalDays) days", r.basalDays > 0)
            diagRow("Weigh-ins", "\(r.weighIns)", r.weighIns > 0)
        }
        .font(.mono(13))
        .padding(.vertical, 2)

        if !r.available {
            Text("Health data isn't available on this device.")
                .font(.footnote).foregroundStyle(Theme.amber)
        } else if !r.hasEnergy {
            VStack(alignment: .leading, spacing: 10) {
                Text("Health is connected but no energy is coming through. Read access for **Active Energy** and **Resting Energy** is most likely off — those default to OFF in the permission sheet and have to be turned on by hand.")
                    .font(.footnote).foregroundStyle(dim)
                Text("Apple Health → Sharing → Apps → Cumulative → turn on **Active Energy** + **Resting Energy** (or “Turn On All”), then come back and tap Sync now.")
                    .font(.footnote).foregroundStyle(dim)
                Button("Open Apple Health") {
                    if let url = URL(string: "x-apple-health://") { openURL(url) }
                }
                .font(.system(size: 14, weight: .semibold))
            }
            .padding(.top, 4)
        }
    }

    private func diagRow(_ label: String, _ value: String, _ ok: Bool) -> some View {
        HStack(spacing: 8) {
            Image(systemName: ok ? "checkmark.circle.fill" : "xmark.circle")
                .foregroundStyle(ok ? Theme.green : Theme.textDim(0.4))
            Text(label).foregroundStyle(Theme.textDim(0.7))
            Spacer()
            Text(value).foregroundStyle(ok ? Theme.text : Theme.textDim(0.4))
        }
    }

    private var assistantSection: some View {
        Section("Assistant") {
            HStack {
                Image(systemName: store.hasAIKey ? "checkmark.circle.fill" : "xmark.circle")
                    .foregroundStyle(store.hasAIKey ? Theme.green : Theme.textDim(0.4))
                Text(store.hasAIKey
                     ? (store.aiKeyIsFromBuild ? "DeepSeek key: built in" : "DeepSeek key: custom")
                     : "No DeepSeek key set")
                Spacer()
            }
            SecureField("DeepSeek API key (sk-…)", text: $aiKeyDraft)
                .font(.mono(13))
            HStack {
                Button("Save key") { store.setAIKey(aiKeyDraft); aiKeyDraft = "" }
                    .disabled(aiKeyDraft.trimmingCharacters(in: .whitespaces).isEmpty)
                if !store.state.aiKey.isEmpty {
                    Spacer()
                    Button("Use built-in", role: .destructive) { store.setAIKey("") }
                }
            }
            Text("Powers “Ask AI” on the calorie entry screen — type a meal in plain language and it estimates the calories. Leave blank to use the key built into this build.")
                .font(.footnote).foregroundStyle(Theme.textDim(0.5))
        }
    }

    private var shortcutsSection: some View {
        Section("Shortcuts") {
            ForEach(store.state.shortcuts) { s in
                HStack {
                    Text(s.label)
                    Spacer()
                    Text("\(Fmt.int(s.kcal)) kcal").foregroundStyle(Theme.textDim(0.5)).font(.mono(13))
                }
            }
            .onDelete { idx in idx.map { store.state.shortcuts[$0].id }.forEach(store.deleteShortcut) }

            HStack(spacing: 8) {
                TextField("Name", text: $newShortcutName)
                TextField("kcal", text: $newShortcutKcal)
                    .keyboardType(.numberPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 70)
                    .font(.mono(14))
                Button {
                    if let kcal = Double(newShortcutKcal), kcal > 0 {
                        store.addShortcut(label: newShortcutName.isEmpty ? "Shortcut" : newShortcutName, kcal: kcal)
                        newShortcutName = ""; newShortcutKcal = ""
                    }
                } label: {
                    Image(systemName: "plus.circle.fill")
                        .foregroundStyle(Double(newShortcutKcal) ?? 0 > 0 ? Theme.green : Theme.textDim(0.3))
                }
                .disabled(!(Double(newShortcutKcal) ?? 0 > 0))
            }
            Text("Shortcuts appear as one-tap chips on the calorie entry screen. You can also save one there with the amount you just typed. Swipe to delete.")
                .font(.footnote).foregroundStyle(Theme.textDim(0.5))
        }
    }

    private var dataSection: some View {
        Section("Data") {
            Button("Load demo data") { store.loadDemoData(); dismiss() }
            Button("Reset everything", role: .destructive) { store.resetAll(); dismiss() }
        }
    }
}
