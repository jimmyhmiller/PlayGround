import Foundation

/// A point estimate with a 1-sigma standard error. `nil`-returning estimators mean
/// "not enough data to say" — we never fabricate a number.
public struct Estimate: Sendable, Equatable {
    public var value: Double
    public var standardError: Double
    public init(_ value: Double, se: Double) { self.value = value; self.standardError = se }
}

public enum Estimator {

    /// Mifflin–St Jeor resting energy (kcal/day), used only as a fallback anchor for true
    /// TDEE when HealthKit has no basal series. Requires sex/age/height in the goal.
    public static func restingMifflin(weightLb: Double, goal: Goal) -> Double? {
        guard let sex = goal.sex, let age = goal.ageYears, let h = goal.heightCm else { return nil }
        let kg = weightLb * 0.45359237
        let base = 10 * kg + 6.25 * h - 5 * age
        switch sex {
        case .male:   return base + 5
        case .female: return base - 161
        }
    }

    /// `effectiveTDEE` — the intake level, **in the user's own (possibly biased) logged
    /// units**, that holds trend weight flat over the window. Budget math is built on this,
    /// so it self-corrects for systematic over/under-logging without knowing the bias.
    ///
    /// effectiveTDEE = meanLoggedIntake − (trendEnergyChange / windowDays)
    ///
    /// `points` are the filter output; window is `[hi-windowDays, hi]` clamped to data.
    public static func effectiveTDEE(
        records: [DailyRecord],
        points: [TrendFilter.Point],
        windowDays: Int = 28
    ) -> Estimate? {
        guard let hi = TrendFilter.lastTrendIndex(points),
              let firstTrend = TrendFilter.firstTrendIndex(points) else { return nil }
        let lo = max(firstTrend, hi - windowDays)
        let span = hi - lo
        guard span >= 7 else { return nil }   // need at least a week of trend to anchor

        guard let tHi = points[hi].trend, let tLo = points[lo].trend,
              let vHi = points[hi].variance, let vLo = points[lo].variance else { return nil }

        // Mean & variance of logged intake over logged days in the window.
        var logs: [Double] = []
        for i in lo...hi {
            if let l = records[i].loggedKcal, l.isFinite { logs.append(l) }
        }
        guard logs.count >= 5 else { return nil }   // need enough logged days to mean over
        let meanLog = logs.reduce(0, +) / Double(logs.count)
        let varLog = logs.count > 1
            ? logs.reduce(0) { $0 + ($1 - meanLog) * ($1 - meanLog) } / Double(logs.count - 1)
            : 0

        // Average daily energy balance from the filter's slope (lower variance than
        // differencing two noisy endpoints); fall back to endpoint difference if needed.
        let slopes = (lo...hi).compactMap { points[$0].slopePerDay }
        let meanSlope = slopes.isEmpty
            ? (tHi - tLo) / Double(span)
            : slopes.reduce(0, +) / Double(slopes.count)
        let energyChangePerDay = meanSlope * kcalPerLb         // <0 when losing
        let tdee = meanLog - energyChangePerDay

        // SE: trend endpoints uncertainty propagated to per-day energy, plus intake SEM.
        let trendSE = sqrt(vHi + vLo) * kcalPerLb / Double(span)
        let intakeSE = sqrt(varLog / Double(logs.count))
        let se = sqrt(trendSE * trendSE + intakeSE * intakeSE)
        return Estimate(tdee, se: se)
    }

    /// `trueTDEE` — actual daily burn from HealthKit (active + basal). Falls back to
    /// active + a Mifflin resting estimate when basal isn't recorded. Returns `nil` when
    /// there's no external signal at all (then logging bias is genuinely unidentifiable).
    public static func trueTDEE(
        records: [DailyRecord],
        points: [TrendFilter.Point],
        goal: Goal,
        windowDays: Int = 28
    ) -> Estimate? {
        guard let hi = TrendFilter.lastTrendIndex(points),
              let firstTrend = TrendFilter.firstTrendIndex(points) else { return nil }
        let lo = max(firstTrend, hi - windowDays)
        guard hi - lo >= 7 else { return nil }

        var totals: [Double] = []
        for i in lo...hi {
            guard let active = records[i].activeKcal, active.isFinite else { continue }
            if let basal = records[i].basalKcal, basal.isFinite {
                totals.append(active + basal)
            } else if let trend = points[i].trend,
                      let resting = restingMifflin(weightLb: trend, goal: goal) {
                totals.append(active + resting)
            }
        }
        guard totals.count >= 5 else { return nil }
        let mean = totals.reduce(0, +) / Double(totals.count)
        let v = totals.count > 1
            ? totals.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(totals.count - 1)
            : 0
        return Estimate(mean, se: sqrt(v / Double(totals.count)))
    }

    /// Logging bias as a multiplicative factor `α = trueTDEE / effectiveTDEE`.
    /// `α > 1` ⇒ you under-log (true intake runs above what you enter). Needs both TDEEs.
    public static func loggingBias(effective: Estimate?, truth: Estimate?) -> Estimate? {
        guard let e = effective, let t = truth, e.value > 0 else { return nil }
        let alpha = t.value / e.value
        // First-order error propagation for a ratio.
        let rel = sqrt(pow(t.standardError / t.value, 2) + pow(e.standardError / e.value, 2))
        return Estimate(alpha, se: alpha * rel)
    }
}
