import Foundation

/// Deterministic SplitMix64 PRNG so scenarios are reproducible across runs/machines.
public struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    public init(seed: UInt64) { state = seed == 0 ? 0x9E3779B97F4A7C15 : seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

public extension SeededGenerator {
    mutating func uniform() -> Double { Double(next() >> 11) * (1.0 / 9007199254740992.0) }
    /// Standard normal via Box–Muller.
    mutating func gauss() -> Double {
        let u1 = max(1e-12, uniform()), u2 = uniform()
        return (-2 * Foundation.log(u1)).squareRoot() * Foundation.cos(2 * .pi * u2)
    }
}

/// Knobs for one simulated dieter.
public struct ScenarioParams: Sendable {
    public var days: Int = 56
    public var startWeightLb: Double = 196
    public var restingTDEE: Double = 2050
    public var activeMean: Double = 420
    public var biasAlpha: Double = 1.08        // true under-logging factor (>1)
    public var ratePerWeek: Double = 0.85       // intended loss rate
    public var intakeNoise: Double = 150
    public var waterAmp: Double = 1.6           // lb scale of hydration swings
    public var pMissWeigh: Double = 0.30        // chance of skipping a weigh-in on a day
    public var pMissLog: Double = 0.08          // chance of not logging that day
    public var hasHealthKit: Bool = true        // active energy available
    public var hasBasal: Bool = true            // basal energy available (accurate anchor)
    public var hasProfile: Bool = true          // sex/age/height for Mifflin fallback
    public init() {}
}

public struct ScenarioTruth: Sendable {
    public var trueWeight: [Double]
    public var trueTDEE: [Double]
    public var trueIntake: [Double]
    public var biasAlpha: Double
    /// Logged-unit maintenance the estimator should converge to: mean(trueTDEE)/alpha.
    public var effectiveMaintenance: Double
}

public struct Scenario: Sendable {
    public var records: [DailyRecord]
    public var goal: Goal
    public var truth: ScenarioTruth
}

public enum Simulator {
    /// Build one scenario: ground-truth physics, then realistically corrupted observations.
    public static func generate(_ p: ScenarioParams, seed: UInt64) -> Scenario {
        var rng = SeededGenerator(seed: seed)
        let n = p.days
        let base = Date(timeIntervalSince1970: 1_700_000_000)
        let day: (Int) -> Date = { Calendar.current.startOfDay(for: base.addingTimeInterval(Double($0) * 86400)) }

        let deficitPerDay = p.ratePerWeek * kcalPerLb / 7.0
        let meanIntake = p.restingTDEE + p.activeMean - deficitPerDay

        var trueWeight = [Double](repeating: 0, count: n)
        var trueTDEE = [Double](repeating: 0, count: n)
        var trueIntake = [Double](repeating: 0, count: n)
        var active = [Double](repeating: 0, count: n)

        var w = p.startWeightLb
        for i in 0..<n {
            let ac = max(100, p.activeMean + rng.gauss() * 120)
            active[i] = ac
            trueTDEE[i] = p.restingTDEE + ac
            let intake = max(800, meanIntake + rng.gauss() * p.intakeNoise)
            trueIntake[i] = intake
            if i > 0 { w += (intake - trueTDEE[i]) / kcalPerLb }
            trueWeight[i] = w
        }

        // Observation corruption: water noise (two sinusoids + AR(1) + occasional spike).
        // Phases are randomized per scenario so day 0 isn't systematically high/low water —
        // a fixed phase would inject a spurious start-anchor bias that no real user has.
        let phase1 = rng.uniform() * 2 * .pi
        let phase2 = rng.uniform() * 2 * .pi
        let period1 = 6.0 + rng.uniform() * 4      // ~6–10 day glycogen/sodium cycle
        let period2 = 2.0 + rng.uniform() * 1.5
        var arState = rng.gauss() * p.waterAmp * 0.45   // random initial hydration state
        var scaleObs = [Double](repeating: 0, count: n)
        for i in 0..<n {
            arState = 0.5 * arState + rng.gauss() * p.waterAmp * 0.45
            var water = p.waterAmp * (0.6 * Foundation.sin(Double(i) / period1 + phase1)
                                      + 0.4 * Foundation.sin(Double(i) / period2 + phase2)) + arState
            if rng.uniform() < 0.05 { water += (1.5 + rng.uniform() * 2.5) }   // sodium/carb spike
            scaleObs[i] = trueWeight[i] + water
        }

        // Build records: drop weigh-ins/logs at random; force first & last weigh-in present.
        var records: [DailyRecord] = []
        records.reserveCapacity(n)
        for i in 0..<n {
            let hasWeigh = (i == 0 || i == n - 1) ? true : rng.uniform() > p.pMissWeigh
            let hasLog = rng.uniform() > p.pMissLog
            var rec = DailyRecord(date: day(i))
            if hasWeigh { rec.weightLb = (scaleObs[i] * 10).rounded() / 10 }
            if hasLog {
                let noisy = (trueIntake[i] / p.biasAlpha) * (1 + rng.gauss() * 0.03)
                rec.loggedKcal = (max(0, noisy)).rounded()
            }
            if p.hasHealthKit { rec.activeKcal = (active[i] + rng.gauss() * 40).rounded() }
            if p.hasBasal { rec.basalKcal = (p.restingTDEE + rng.gauss() * 30).rounded() }
            records.append(rec)
        }

        var goal = Goal(targetWeightLb: p.startWeightLb - 18, ratePerWeek: p.ratePerWeek)
        if p.hasProfile {
            // A plausible profile whose Mifflin resting lands near restingTDEE.
            goal.sex = .male
            goal.ageYears = 38
            // Invert Mifflin (male) for height given start weight so the anchor is realistic.
            let kg = p.startWeightLb * 0.45359237
            goal.heightCm = (p.restingTDEE - 5 - 10 * kg + 5 * 38) / 6.25
        }

        let meanTDEE = trueTDEE.reduce(0, +) / Double(n)
        let truth = ScenarioTruth(
            trueWeight: trueWeight, trueTDEE: trueTDEE, trueIntake: trueIntake,
            biasAlpha: p.biasAlpha, effectiveMaintenance: meanTDEE / p.biasAlpha
        )
        return Scenario(records: records, goal: goal, truth: truth)
    }
}
