import SwiftUI
import CalorieModel

struct SettingsView: View {
    @EnvironmentObject var store: AppStore
    @Environment(\.dismiss) private var dismiss
    @State private var goal: Goal = Goal(targetWeightLb: 178, ratePerWeek: 0.85)
    @State private var hasProfile = false

    var body: some View {
        NavigationStack {
            Form {
                goalSection
                profileSection
                healthSection
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
            if store.state.shortcuts.isEmpty {
                Text("Add shortcuts from the calorie entry screen via “+ save”.")
                    .font(.footnote).foregroundStyle(Theme.textDim(0.5))
            }
        }
    }

    private var dataSection: some View {
        Section("Data") {
            Button("Load demo data") { store.loadDemoData(); dismiss() }
            Button("Reset everything", role: .destructive) { store.resetAll(); dismiss() }
        }
    }
}
