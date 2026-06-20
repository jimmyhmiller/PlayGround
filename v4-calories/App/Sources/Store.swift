import Foundation
import CalorieModel
import SwiftUI

/// The full persisted state of the app. Domain types come straight from CalorieModel.
struct AppState: Codable {
    var entries: [CalorieEntry] = []
    var weighIns: [WeighIn] = []
    var shortcuts: [Shortcut] = []
    var goal = Goal(targetWeightLb: 178, ratePerWeek: 0.85)
    var startDate: Date? = nil
    var healthKitEnabled = false
    var hasOnboarded = false
    /// Optional in-app override for the DeepSeek key; falls back to the build-injected one.
    var aiKey: String = ""
}

@MainActor
final class AppStore: ObservableObject {
    @Published private(set) var state: AppState
    @Published private(set) var analysis: Analysis?

    /// HealthKit-derived daily energy, keyed by start-of-day. Not persisted (re-fetched).
    @Published var activeByDay: [Date: Double] = [:]
    @Published var basalByDay: [Date: Double] = [:]
    /// Weigh-ins imported from HealthKit (e.g. a smart scale), merged with manual ones.
    @Published var healthWeighIns: [WeighIn] = []

    /// Result of the last Health sync, so the UI can explain *why* there is (or isn't) data
    /// instead of silently showing an empty screen.
    struct SyncReport: Equatable {
        var available: Bool
        var requestApproved: Bool
        var alreadyDetermined: Bool
        var activeDays: Int
        var basalDays: Int
        var weighIns: Int
        var hasEnergy: Bool { activeDays > 0 || basalDays > 0 }
    }
    @Published var lastSync: SyncReport?
    @Published var isSyncing = false

    let health = HealthKitManager()
    private let cal = Calendar.current
    private let storeURL: URL

    init() {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        storeURL = dir.appendingPathComponent("cumulative-tracker.json")
        if let data = try? Data(contentsOf: storeURL),
           let loaded = try? JSONDecoder().decode(AppState.self, from: data) {
            state = loaded
        } else {
            state = AppState()
        }
        recompute()
        if state.healthKitEnabled { Task { await syncHealth() } }

        // Debug launch hooks for screenshotting / manual QA.
        let args = ProcessInfo.processInfo.arguments
        if args.contains("--reset") { resetAll() }
        if args.contains("--demo") { loadDemoData() }
    }

    // MARK: - Derived

    var today: Date { cal.startOfDay(for: Date()) }

    var todayEntries: [CalorieEntry] {
        state.entries
            .filter { cal.isDate($0.date, inSameDayAs: today) }
            .sorted { $0.date < $1.date }
    }

    var todayTotal: Double { todayEntries.reduce(0) { $0 + $1.kcal } }

    var allWeighIns: [WeighIn] {
        (state.weighIns + healthWeighIns).sorted { $0.date > $1.date }
    }

    // MARK: - AI assistant

    /// Resolved DeepSeek key: in-app override first, then the build-injected Info.plist value.
    var aiKey: String? {
        let override = state.aiKey.trimmingCharacters(in: .whitespaces)
        if !override.isEmpty { return override }
        if let k = Bundle.main.object(forInfoDictionaryKey: "DEEPSEEK_API_KEY") as? String {
            let trimmed = k.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty, !trimmed.hasPrefix("$(") { return trimmed }
        }
        return nil
    }

    var hasAIKey: Bool { aiKey != nil }

    /// True when the key came from the build (not a user override) — for display only.
    var aiKeyIsFromBuild: Bool {
        state.aiKey.trimmingCharacters(in: .whitespaces).isEmpty && hasAIKey
    }

    func setAIKey(_ key: String) {
        state.aiKey = key.trimmingCharacters(in: .whitespaces)
        persist()
    }

    // MARK: - Burn (energy expenditure)

    struct BurnDay: Identifiable {
        var date: Date
        var active: Double?
        var resting: Double?
        var id: Date { date }
        var total: Double? {
            if active == nil && resting == nil { return nil }
            return (active ?? 0) + (resting ?? 0)
        }
    }

    /// Resting energy for a day: Apple Health basal if present, else a Mifflin estimate from
    /// trend weight + profile (nil if neither basal nor a profile is available).
    func restingEstimate(on day: Date) -> Double? {
        if let b = basalByDay[cal.startOfDay(for: day)] { return b }
        let w = analysis?.trendWeightLb ?? allWeighIns.first?.weightLb
        guard let w else { return nil }
        return Estimator.restingMifflin(weightLb: w, goal: state.goal)
    }

    /// Per-day burn over the tracked range, for days that have any Health energy data.
    func burnSeries() -> [BurnDay] {
        let start = state.startDate.map { cal.startOfDay(for: $0) } ?? today
        var out: [BurnDay] = []
        var day = start
        while day <= today {
            let active = activeByDay[day]
            let resting = basalByDay[day] ?? (active != nil ? restingEstimate(on: day) : nil)
            if active != nil || basalByDay[day] != nil {
                out.append(BurnDay(date: day, active: active, resting: resting))
            }
            guard let next = cal.date(byAdding: .day, value: 1, to: day) else { break }
            day = next
        }
        return out
    }

    var todayBurn: BurnDay {
        BurnDay(date: today, active: activeByDay[today],
                resting: basalByDay[today] ?? restingEstimate(on: today))
    }

    var hasBurnData: Bool { !activeByDay.isEmpty || !basalByDay.isEmpty }

    /// Contiguous per-day records spanning the tracked period, fed to the analyzer.
    func buildRecords() -> [DailyRecord] {
        let start = state.startDate.map { cal.startOfDay(for: $0) } ?? today
        return DailyRecord.contiguous(
            from: start, to: today, calendar: cal,
            entries: state.entries,
            weighIns: state.weighIns + healthWeighIns,
            activeByDay: activeByDay,
            basalByDay: basalByDay
        )
    }

    func recompute() {
        analysis = Analyzer.analyze(records: buildRecords(), goal: state.goal)
    }

    // MARK: - Mutations

    private func persist() {
        if let data = try? JSONEncoder().encode(state) { try? data.write(to: storeURL) }
        recompute()
    }

    private func ensureStartDate(_ date: Date) {
        let d = cal.startOfDay(for: date)
        if let s = state.startDate { if d < s { state.startDate = d } } else { state.startDate = d }
    }

    func addEntry(kcal: Double, label: String? = nil, date: Date = Date()) {
        guard kcal > 0 else { return }
        state.entries.append(CalorieEntry(date: date, kcal: kcal, label: label))
        ensureStartDate(date)
        if state.healthKitEnabled { Task { await health.saveDietaryEnergy(kcal: kcal, date: date) } }
        persist()
    }

    func updateEntry(_ id: UUID, kcal: Double, label: String?) {
        guard let i = state.entries.firstIndex(where: { $0.id == id }) else { return }
        guard kcal > 0 else { return }
        state.entries[i].kcal = kcal
        state.entries[i].label = (label?.isEmpty == true) ? nil : label
        persist()
    }

    func deleteEntry(_ id: UUID) {
        state.entries.removeAll { $0.id == id }
        persist()
    }

    func addWeighIn(_ lb: Double, date: Date = Date()) {
        guard lb > 0 else { return }
        state.weighIns.append(WeighIn(date: date, weightLb: lb))
        ensureStartDate(date)
        if state.healthKitEnabled { Task { await health.saveBodyMass(lb: lb, date: date) } }
        persist()
    }

    func addShortcut(label: String, kcal: Double) {
        guard kcal > 0 else { return }
        state.shortcuts.insert(Shortcut(label: label, kcal: kcal), at: 0)
        persist()
    }

    func deleteShortcut(_ id: UUID) {
        state.shortcuts.removeAll { $0.id == id }
        persist()
    }

    func updateGoal(_ goal: Goal) { state.goal = goal; persist() }
    func markOnboarded() {
        state.hasOnboarded = true
        if state.shortcuts.isEmpty { state.shortcuts = Self.defaultShortcuts }
        persist()
    }

    // MARK: - HealthKit

    func setHealthKitEnabled(_ on: Bool) {
        state.healthKitEnabled = on
        persist()
        if on { Task { await syncHealth() } } else {
            activeByDay = [:]; basalByDay = [:]; healthWeighIns = []; recompute()
        }
    }

    func syncHealth() async {
        guard state.healthKitEnabled else { return }
        guard health.isAvailable else {
            lastSync = SyncReport(available: false, requestApproved: false,
                                  alreadyDetermined: false, activeDays: 0, basalDays: 0, weighIns: 0)
            return
        }
        isSyncing = true
        defer { isSyncing = false }
        let approved = await health.requestAuthorization()
        let status = await health.requestStatus()
        let start = state.startDate.map { cal.startOfDay(for: $0) }
            ?? cal.date(byAdding: .day, value: -90, to: today)!
        let active = await health.dailyEnergy(.active, from: start, to: today)
        let basal = await health.dailyEnergy(.basal, from: start, to: today)
        let weights = await health.bodyMass(from: start, to: today)
        // syncHealth is @MainActor-isolated, so these assignments are already on the main actor.
        self.activeByDay = active
        self.basalByDay = basal
        self.healthWeighIns = weights
        self.lastSync = SyncReport(available: true,
                                   requestApproved: approved,
                                   alreadyDetermined: status == .alreadyDetermined,
                                   activeDays: active.count, basalDays: basal.count,
                                   weighIns: weights.count)
        self.recompute()
    }

    // MARK: - Demo / reset

    /// Seed a realistic 56-day history using the proven Simulator, ending today.
    func loadDemoData() {
        var p = ScenarioParams()
        p.days = 56
        let s = Simulator.generate(p, seed: 20260620)
        let n = s.records.count
        var entries: [CalorieEntry] = []
        var weighIns: [WeighIn] = []
        var active: [Date: Double] = [:]
        var basal: [Date: Double] = [:]

        for (i, rec) in s.records.enumerated() {
            let day = cal.date(byAdding: .day, value: -(n - 1 - i), to: today)!
            let dayStart = cal.startOfDay(for: day)
            if let logged = rec.loggedKcal, logged > 0 {
                if i == n - 1 {
                    // Today mirrors the design's example entries.
                    entries.append(CalorieEntry(date: at(dayStart, 8, 12), kcal: 310, label: "Coffee + oats"))
                    entries.append(CalorieEntry(date: at(dayStart, 12, 40), kcal: max(0, logged - 310)))
                } else {
                    entries.append(CalorieEntry(date: at(dayStart, 9, 0), kcal: (logged * 0.4).rounded()))
                    entries.append(CalorieEntry(date: at(dayStart, 13, 30), kcal: (logged * 0.6).rounded()))
                }
            }
            if let w = rec.weightLb { weighIns.append(WeighIn(date: at(dayStart, 7, 30), weightLb: w)) }
            if let a = rec.activeKcal { active[dayStart] = a }
            if let b = rec.basalKcal { basal[dayStart] = b }
        }

        var newState = AppState()
        newState.entries = entries
        newState.weighIns = weighIns
        newState.shortcuts = Self.defaultShortcuts
        newState.goal = Goal(targetWeightLb: 178, ratePerWeek: 0.85,
                             sex: .male, ageYears: 38, heightCm: 180)
        newState.startDate = cal.date(byAdding: .day, value: -(n - 1), to: today)
        newState.hasOnboarded = true
        state = newState
        activeByDay = active
        basalByDay = basal
        healthWeighIns = []
        persist()
    }

    func resetAll() {
        state = AppState()
        activeByDay = [:]; basalByDay = [:]; healthWeighIns = []
        persist()
    }

    private func at(_ day: Date, _ h: Int, _ m: Int) -> Date {
        cal.date(bySettingHour: h, minute: m, second: 0, of: day) ?? day
    }

    static let defaultShortcuts: [Shortcut] = [
        Shortcut(label: "Coffee", kcal: 5),
        Shortcut(label: "Banana", kcal: 105),
        Shortcut(label: "Usual lunch", kcal: 640),
        Shortcut(label: "Shake", kcal: 160),
        Shortcut(label: "Eggs ×3", kcal: 230),
        Shortcut(label: "Beer", kcal: 200),
    ]
}
