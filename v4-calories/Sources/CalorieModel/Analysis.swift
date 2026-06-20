import Foundation

public struct WeightPoint: Sendable, Equatable {
    public var date: Date
    public var trend: Double?
    public var observed: Double?
    public var logsOnly: Double?   // where the logs-at-face-value line would put you
}

public struct DatedValue: Sendable, Equatable {
    public var date: Date
    public var value: Double
}

public struct DatedBand: Sendable, Equatable {
    public var date: Date
    public var value: Double
    public var se: Double
}

/// Everything the UI needs, computed from the raw daily records + goal. Optionals are
/// honest: a field is `nil` until there's enough data to estimate it.
public struct Analysis: Sendable, Equatable {
    // Cumulative deficit (the headline) — scale-anchored, projected past the last weigh-in.
    public var cumulativeDeficitKcal: Double
    public var cumulativeDeficitSE: Double
    public var daysCount: Int
    public var lossSoFarLb: Double        // positive = lost (from trend; projected after last weigh-in)
    public var aheadOfPlanLb: Double      // positive = ahead of the goal rate

    // Weight
    public var trendWeightLb: Double?
    public var startWeightLb: Double?
    public var totalChangeLb: Double?     // trend - start (negative = lost)
    public var ratePerWeekLb: Double?     // current trend slope, lb/week (negative = losing)
    public var daysSinceLastWeighIn: Int?

    // The model
    public var effectiveTDEE: Estimate?
    public var trueTDEE: Estimate?
    public var loggingBias: Estimate?     // alpha; >1 means under-logging

    // Budget (in the user's logged units, so they log straight against it)
    public var dailyBudgetKcal: Double?
    public var budgetIsCalibrated: Bool   // false = provisional (not yet learned from scale)

    // Goal
    public var goalProgress: Double       // 0...1
    public var toGoLb: Double?
    public var etaDate: Date?

    // Reconciliation (Weight screen)
    public var logsPredictedChangeLb: Double?
    public var scaleChangeLb: Double?

    // Series for charts
    public var trendSeries: [WeightPoint]
    public var cumulativeDeficitSeries: [DatedValue]
    public var effectiveTDEESeries: [DatedBand]
    public var biasSeries: [DatedValue]
}

public struct AnalysisConfig: Sendable {
    public var trend: TrendFilter
    public var tdeeWindowDays: Int        // trailing window for TDEE/bias
    public var rateWindowDays: Int        // trailing window for the current weight slope
    public init(trend: TrendFilter = TrendFilter(),
                tdeeWindowDays: Int = 28,
                rateWindowDays: Int = 21) {
        self.trend = trend
        self.tdeeWindowDays = tdeeWindowDays
        self.rateWindowDays = rateWindowDays
    }
}

public enum Analyzer {

    public static func analyze(records: [DailyRecord],
                               goal: Goal,
                               config: AnalysisConfig = AnalysisConfig()) -> Analysis {
        let n = records.count
        let points = config.trend.run(records)

        let firstIdx = TrendFilter.firstTrendIndex(points)
        let lastIdx = TrendFilter.lastTrendIndex(points)

        // --- Model estimates (trailing window) ---
        let effective = Estimator.effectiveTDEE(records: records, points: points,
                                                windowDays: config.tdeeWindowDays)
        let truth = Estimator.trueTDEE(records: records, points: points, goal: goal,
                                       windowDays: config.tdeeWindowDays)
        let bias = Estimator.loggingBias(effective: effective, truth: truth)

        // --- Weight figures ---
        let trendWeight = lastIdx.flatMap { points[$0].trend }
        let startWeight = firstIdx.flatMap { points[$0].trend }
        let totalChange: Double? = {
            guard let t = trendWeight, let s = startWeight else { return nil }
            return t - s
        }()
        let rate = currentRate(points: points, windowDays: config.rateWindowDays)
        let daysSinceWeigh: Int? = {
            guard let li = lastIdx else { return nil }
            return (n - 1) - li
        }()

        // --- Cumulative deficit: scale-anchored up to last weigh-in, projected after. ---
        let dailyDeficitTarget = goal.ratePerWeek * kcalPerLb / 7.0
        // Each endpoint's true weight can't be known better than ~one marginal water-std,
        // because water is *correlated* (the Kalman, assuming independent readings, averages
        // it down too optimistically). Floor each endpoint variance at half the water variance
        // so the reported band is honest. This is the core "don't over-claim" guard.
        let waterVar = config.trend.estimateWaterVar(records) ?? config.trend.measurementVar
        let endpointFloor = waterVar * 0.5
        var cumDeficit = 0.0
        var cumDeficitVar = 0.0
        if let fi = firstIdx, let li = lastIdx,
           let tF = points[fi].trend, let tL = points[li].trend,
           let vF = points[fi].variance, let vL = points[li].variance {
            cumDeficit = (tF - tL) * kcalPerLb           // banked energy from the scale
            cumDeficitVar = (max(vF, endpointFloor) + max(vL, endpointFloor)) * kcalPerLb * kcalPerLb
            // Project the days since the last weigh-in using the calibrated budget frame.
            if let e = effective, li < n - 1 {
                for i in (li + 1)..<n {
                    if let logged = records[i].loggedKcal {
                        cumDeficit += (e.value - logged)
                    } else {
                        // No log that day: assume we hit plan (neutral, reconciles at next weigh-in).
                        cumDeficit += dailyDeficitTarget
                    }
                    cumDeficitVar += e.standardError * e.standardError   // projection adds uncertainty
                }
            }
        }
        let cumDeficitSE = sqrt(cumDeficitVar)
        let lossSoFar = cumDeficit / kcalPerLb

        // Ahead/behind plan: actual loss vs what the goal rate prescribes over elapsed days.
        let elapsedDays: Double = {
            guard let fi = firstIdx, let li = lastIdx else { return 0 }
            return Double(li - fi) + Double(max(0, (n - 1) - li))
        }()
        let plannedLoss = goal.ratePerWeek * (elapsedDays / 7.0)
        let aheadOfPlan = lossSoFar - plannedLoss

        // --- Budget ---
        let (budget, calibrated) = budgetKcal(effective: effective, truth: truth, goal: goal,
                                              trendWeight: trendWeight,
                                              dailyDeficitTarget: dailyDeficitTarget)

        // --- Goal / ETA ---
        var goalProgress = 0.0
        var toGo: Double? = nil
        var eta: Date? = nil
        if let cur = trendWeight, let start = startWeight {
            let span = start - goal.targetWeightLb
            if abs(span) > 0.001 {
                goalProgress = min(1, max(0, (start - cur) / span))
            }
            toGo = cur - goal.targetWeightLb
            // ETA from the *actual* trend rate, adjusting over time; fall back to plan rate.
            let usableRate: Double? = {
                if let r = rate, abs(r) > 0.05, (toGo ?? 0) * r < 0 { return r } // moving toward goal
                if abs(goal.ratePerWeek) > 0.01 { return -copysign(goal.ratePerWeek, toGo ?? 0) }
                return nil
            }()
            if let r = usableRate, let tg = toGo, abs(r) > 0.001 {
                // Time to close the gap: need weight to change by (target - current) = -tg at rate r.
                let weeks = -tg / r
                if weeks > 0, weeks.isFinite, let last = records.last?.date {
                    eta = Calendar.current.date(byAdding: .day, value: Int((weeks * 7).rounded()), to: last)
                }
            }
        }

        // --- Reconciliation (logs-only vs scale) ---
        let (logsPredicted, scaleChange, trendSeries) =
            reconciliation(records: records, points: points, truth: truth, effective: effective,
                           firstIdx: firstIdx, lastIdx: lastIdx)

        // --- Chart series ---
        let cumSeries = cumulativeDeficitSeries(records: records, points: points,
                                                effective: effective,
                                                dailyDeficitTarget: dailyDeficitTarget,
                                                firstIdx: firstIdx, lastIdx: lastIdx)
        let (tdeeSeries, biasSeries) = rollingModelSeries(records: records, points: points,
                                                          goal: goal, config: config)

        return Analysis(
            cumulativeDeficitKcal: cumDeficit,
            cumulativeDeficitSE: cumDeficitSE,
            daysCount: { if let fi = firstIdx { return n - fi } else { return 0 } }(),
            lossSoFarLb: lossSoFar,
            aheadOfPlanLb: aheadOfPlan,
            trendWeightLb: trendWeight,
            startWeightLb: startWeight,
            totalChangeLb: totalChange,
            ratePerWeekLb: rate,
            daysSinceLastWeighIn: daysSinceWeigh,
            effectiveTDEE: effective,
            trueTDEE: truth,
            loggingBias: bias,
            dailyBudgetKcal: budget,
            budgetIsCalibrated: calibrated,
            goalProgress: goalProgress,
            toGoLb: toGo,
            etaDate: eta,
            logsPredictedChangeLb: logsPredicted,
            scaleChangeLb: scaleChange,
            trendSeries: trendSeries,
            cumulativeDeficitSeries: cumSeries,
            effectiveTDEESeries: tdeeSeries,
            biasSeries: biasSeries
        )
    }

    // MARK: - Pieces

    /// Current trend slope in lb/week, from the filter's smoothed slope averaged over a
    /// trailing window (falls back to endpoint difference if slope is unavailable).
    static func currentRate(points: [TrendFilter.Point], windowDays: Int) -> Double? {
        guard let hi = TrendFilter.lastTrendIndex(points),
              let firstTrend = TrendFilter.firstTrendIndex(points) else { return nil }
        let lo = max(firstTrend, hi - windowDays)
        let span = hi - lo
        guard span >= 3 else { return nil }
        let slopes = (lo...hi).compactMap { points[$0].slopePerDay }
        if !slopes.isEmpty { return slopes.reduce(0, +) / Double(slopes.count) * 7.0 }
        guard let tHi = points[hi].trend, let tLo = points[lo].trend else { return nil }
        return (tHi - tLo) / Double(span) * 7.0
    }

    static func budgetKcal(effective: Estimate?, truth: Estimate?, goal: Goal,
                           trendWeight: Double?, dailyDeficitTarget: Double) -> (Double?, Bool) {
        if let e = effective {
            return (max(800, e.value - dailyDeficitTarget), true)   // calibrated to the scale
        }
        // Provisional bootstrap while still learning.
        if let t = truth {
            return (max(800, t.value - dailyDeficitTarget), false)
        }
        if let w = trendWeight, let resting = Estimator.restingMifflin(weightLb: w, goal: goal) {
            let provisionalTDEE = resting * 1.45   // light-activity multiplier
            return (max(800, provisionalTDEE - dailyDeficitTarget), false)
        }
        return (nil, false)
    }

    static func reconciliation(records: [DailyRecord], points: [TrendFilter.Point],
                               truth: Estimate?, effective: Estimate?,
                               firstIdx: Int?, lastIdx: Int?)
        -> (logsPredicted: Double?, scaleChange: Double?, series: [WeightPoint]) {

        var series: [WeightPoint] = []
        series.reserveCapacity(records.count)

        // The logs-only line: integrate (logged - referenceTDEE) at face value. Using the
        // external (HealthKit) TDEE makes the gap to the scale equal the under-logging.
        // Without an external anchor we fall back to effectiveTDEE (gap ≈ 0 → "can't tell").
        let refTDEE = truth?.value ?? effective?.value
        var logsOnlyEnergy = 0.0
        let anchor = firstIdx.flatMap { points[$0].trend }

        for i in records.indices {
            var logsOnly: Double? = nil
            if let a = anchor, let ref = refTDEE, let fi = firstIdx, i >= fi {
                if i > fi, let logged = records[i].loggedKcal {
                    logsOnlyEnergy += (logged - ref)
                }
                logsOnly = a + logsOnlyEnergy / kcalPerLb
            }
            series.append(WeightPoint(date: records[i].date,
                                      trend: points[i].trend,
                                      observed: records[i].weightLb,
                                      logsOnly: logsOnly))
        }

        var logsPredicted: Double? = nil
        var scaleChange: Double? = nil
        if let fi = firstIdx, let li = lastIdx,
           let lastLogsOnly = series[li].logsOnly, let a = anchor,
           let tF = points[fi].trend, let tL = points[li].trend {
            logsPredicted = lastLogsOnly - a
            scaleChange = tL - tF
        }
        return (logsPredicted, scaleChange, series)
    }

    static func cumulativeDeficitSeries(records: [DailyRecord], points: [TrendFilter.Point],
                                        effective: Estimate?, dailyDeficitTarget: Double,
                                        firstIdx: Int?, lastIdx: Int?) -> [DatedValue] {
        guard let fi = firstIdx, let li = lastIdx, let tF = points[fi].trend else { return [] }
        var out: [DatedValue] = []
        var projected = 0.0
        for i in fi..<records.count {
            let value: Double
            if i <= li, let t = points[i].trend {
                value = (tF - t) * kcalPerLb
            } else if let lastTrend = points[li].trend {
                // Project beyond the last weigh-in.
                if let e = effective, let logged = records[i].loggedKcal {
                    projected += (e.value - logged)
                } else {
                    projected += dailyDeficitTarget
                }
                value = (tF - lastTrend) * kcalPerLb + projected
            } else {
                value = 0
            }
            out.append(DatedValue(date: records[i].date, value: value))
        }
        return out
    }

    /// Re-run TDEE/bias as of each day to get the "converging" series the Trends screen shows.
    static func rollingModelSeries(records: [DailyRecord], points: [TrendFilter.Point],
                                   goal: Goal, config: AnalysisConfig)
        -> (tdee: [DatedBand], bias: [DatedValue]) {
        var tdee: [DatedBand] = []
        var bias: [DatedValue] = []
        guard let firstTrend = TrendFilter.firstTrendIndex(points) else { return (tdee, bias) }
        let start = firstTrend + 7
        guard start < records.count else { return (tdee, bias) }
        for end in start..<records.count {
            let subRecords = Array(records[0...end])
            let subPoints = Array(points[0...end])
            let e = Estimator.effectiveTDEE(records: subRecords, points: subPoints,
                                            windowDays: config.tdeeWindowDays)
            let t = Estimator.trueTDEE(records: subRecords, points: subPoints, goal: goal,
                                       windowDays: config.tdeeWindowDays)
            // Plot the best available TDEE so the chart matches the headline (true burn when
            // HealthKit is connected, otherwise as-logged maintenance).
            if let best = t ?? e {
                tdee.append(DatedBand(date: records[end].date, value: best.value, se: best.standardError))
            }
            if let e, let b = Estimator.loggingBias(effective: e, truth: t) {
                bias.append(DatedValue(date: records[end].date, value: (b.value - 1) * 100))
            }
        }
        return (tdee, bias)
    }
}
