import Foundation

/// Energy in one pound of body-mass change. The classic value is 3500 kcal/lb.
/// Real tissue loss is a mix of fat (~3500) and lean/water, but over a multi-week
/// trend the scale-anchored math self-corrects, so this constant only sets the
/// *units* of the deficit, not its accuracy.
public let kcalPerLb: Double = 3500.0

/// One calendar day's worth of inputs. The model works on a *contiguous* array of
/// these (one per day, gaps included as `nil` fields) so the filters and cumulative
/// sums have a well-defined time axis. Use `DailyRecord.contiguous(...)` to build it
/// from sparse entries.
public struct DailyRecord: Sendable, Equatable {
    /// Start-of-day (local midnight). Days must be consecutive and unique.
    public var date: Date
    /// Sum of calorie entries logged for this day. `nil` means the user logged nothing
    /// (genuinely unknown intake), which is different from `0` (a real fast).
    public var loggedKcal: Double?
    /// Active energy from HealthKit (move ring), if available.
    public var activeKcal: Double?
    /// Basal/resting energy from HealthKit, if available (needs Apple Watch usually).
    public var basalKcal: Double?
    /// A scale reading for this day (manual entry or HealthKit body-mass), if any.
    public var weightLb: Double?

    public init(date: Date,
                loggedKcal: Double? = nil,
                activeKcal: Double? = nil,
                basalKcal: Double? = nil,
                weightLb: Double? = nil) {
        self.date = date
        self.loggedKcal = loggedKcal
        self.activeKcal = activeKcal
        self.basalKcal = basalKcal
        self.weightLb = weightLb
    }
}

/// A single weigh-in, before aggregation into days.
public struct WeighIn: Sendable, Equatable, Identifiable, Codable {
    public var id: UUID
    public var date: Date
    public var weightLb: Double
    /// True when imported from HealthKit (e.g. a smart scale) rather than typed in.
    public var fromHealthKit: Bool
    public init(id: UUID = UUID(), date: Date, weightLb: Double, fromHealthKit: Bool = false) {
        self.id = id; self.date = date; self.weightLb = weightLb; self.fromHealthKit = fromHealthKit
    }
}

/// A single calorie entry. Name is optional by design — the fast path is just a number.
public struct CalorieEntry: Sendable, Equatable, Identifiable, Codable {
    public var id: UUID
    public var date: Date
    public var kcal: Double
    public var label: String?
    public init(id: UUID = UUID(), date: Date, kcal: Double, label: String? = nil) {
        self.id = id; self.date = date; self.kcal = kcal; self.label = label
    }
}

/// A reusable one-tap shortcut for a common food/amount.
public struct Shortcut: Sendable, Equatable, Identifiable, Codable {
    public var id: UUID
    public var label: String
    public var kcal: Double
    public init(id: UUID = UUID(), label: String, kcal: Double) {
        self.id = id; self.label = label; self.kcal = kcal
    }
}

/// The user's objective and physiology. Goal fields drive the budget and ETA; the
/// optional body fields let us estimate resting metabolic rate when HealthKit has no
/// basal data, which is what makes logging-bias identifiable without an Apple Watch.
public struct Goal: Sendable, Equatable, Codable {
    public var targetWeightLb: Double
    /// Desired loss rate, lb/week. Positive = lose. Negative = gain.
    public var ratePerWeek: Double

    // Optional profile for a Mifflin–St Jeor resting-energy fallback.
    public enum Sex: String, Sendable, Equatable, Codable { case male, female }
    public var sex: Sex?
    public var ageYears: Double?
    public var heightCm: Double?

    public init(targetWeightLb: Double,
                ratePerWeek: Double,
                sex: Sex? = nil,
                ageYears: Double? = nil,
                heightCm: Double? = nil) {
        self.targetWeightLb = targetWeightLb
        self.ratePerWeek = ratePerWeek
        self.sex = sex
        self.ageYears = ageYears
        self.heightCm = heightCm
    }
}

// MARK: - Tolerant decoding
//
// Auto-synthesized `Decodable` throws on a missing key even when the property has a
// default. That means adding ANY new stored field silently breaks decoding of older
// saved data. These hand-written decoders use defaults for missing/!decodable fields
// so persisted data survives across app versions. Encoding stays synthesized.

public extension WeighIn {
    enum CodingKeys: String, CodingKey { case id, date, weightLb, fromHealthKit }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            id: (try? c.decode(UUID.self, forKey: .id)) ?? UUID(),
            date: try c.decode(Date.self, forKey: .date),
            weightLb: try c.decode(Double.self, forKey: .weightLb),
            fromHealthKit: (try? c.decode(Bool.self, forKey: .fromHealthKit)) ?? false)
    }
}

public extension CalorieEntry {
    enum CodingKeys: String, CodingKey { case id, date, kcal, label }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            id: (try? c.decode(UUID.self, forKey: .id)) ?? UUID(),
            date: try c.decode(Date.self, forKey: .date),
            kcal: try c.decode(Double.self, forKey: .kcal),
            label: try? c.decode(String.self, forKey: .label))
    }
}

public extension Shortcut {
    enum CodingKeys: String, CodingKey { case id, label, kcal }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            id: (try? c.decode(UUID.self, forKey: .id)) ?? UUID(),
            label: (try? c.decode(String.self, forKey: .label)) ?? "Shortcut",
            kcal: try c.decode(Double.self, forKey: .kcal))
    }
}

public extension Goal {
    enum CodingKeys: String, CodingKey { case targetWeightLb, ratePerWeek, sex, ageYears, heightCm }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            targetWeightLb: (try? c.decode(Double.self, forKey: .targetWeightLb)) ?? 178,
            ratePerWeek: (try? c.decode(Double.self, forKey: .ratePerWeek)) ?? 0.85,
            sex: try? c.decode(Sex.self, forKey: .sex),
            ageYears: try? c.decode(Double.self, forKey: .ageYears),
            heightCm: try? c.decode(Double.self, forKey: .heightCm))
    }
}

public extension DailyRecord {
    /// Expand sparse weigh-ins + calorie entries + HealthKit samples into a contiguous
    /// per-day array spanning `[start, end]`. Days with no data become all-`nil` records,
    /// which the model treats as "missing" (the scale fills the gap later).
    static func contiguous(
        from start: Date,
        to end: Date,
        calendar: Calendar = .current,
        entries: [CalorieEntry] = [],
        weighIns: [WeighIn] = [],
        activeByDay: [Date: Double] = [:],
        basalByDay: [Date: Double] = [:]
    ) -> [DailyRecord] {
        let startDay = calendar.startOfDay(for: start)
        let endDay = calendar.startOfDay(for: end)
        guard endDay >= startDay else { return [] }

        // Aggregate entries by day.
        var loggedByDay: [Date: Double] = [:]
        for e in entries {
            let d = calendar.startOfDay(for: e.date)
            loggedByDay[d, default: 0] += e.kcal
        }
        // Latest weigh-in of the day wins (most recent reading).
        var weightByDay: [Date: (Date, Double)] = [:]
        for w in weighIns {
            let d = calendar.startOfDay(for: w.date)
            if let existing = weightByDay[d], existing.0 >= w.date { continue }
            weightByDay[d] = (w.date, w.weightLb)
        }

        var out: [DailyRecord] = []
        var day = startDay
        while day <= endDay {
            out.append(DailyRecord(
                date: day,
                loggedKcal: loggedByDay[day],
                activeKcal: activeByDay[calendar.startOfDay(for: day)],
                basalKcal: basalByDay[calendar.startOfDay(for: day)],
                weightLb: weightByDay[day]?.1
            ))
            guard let next = calendar.date(byAdding: .day, value: 1, to: day) else { break }
            day = next
        }
        return out
    }
}
