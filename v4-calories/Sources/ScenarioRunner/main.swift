import CalorieModel
import Foundation

// Robustness harness: thousands of randomized realistic dieters, plus targeted stress
// tests. Verifies the model tracks ground truth and never "goes crazy" (NaN, absurd
// budgets, runaway deficit) under water noise, missing weigh-ins, and under-logging.

struct Metrics {
    var trendMaxErr = 0.0
    var trendRMSE = 0.0
    var cumDeficitErr = 0.0    // kcal
    var cumDeficitErrLb = 0.0
    var cumDeficitSE = 0.0     // reported kcal SE
    var cumWithin2SE = false   // is the true value inside the reported ±2σ band?
    var effErr: Double? = nil
    var biasErr: Double? = nil
    var budget: Double? = nil
    var sane = true            // no NaN/Inf anywhere we read
    /// Worst lb by which the trend sits BELOW every reading taken so far. This is the bug the
    /// scale floor exists to prevent ("the app says I weigh less than the scale has ever
    /// shown"), so it must be 0.
    var trendBelowAllReadings = 0.0
    /// Worst lb by which the trend sits ABOVE every reading taken so far — the same
    /// over-extrapolation in the other direction. NOT fixed by the floor (which is one-sided);
    /// reported so the residual is visible rather than silently unmeasured.
    var trendAboveAllReadings = 0.0
    /// The final trend vs the lowest reading ever — the number the user actually reads.
    var lastTrendBelowMinObs = 0.0
    /// How much work the scale floor had to do: days floored, and the worst lift in lb. These
    /// measure the raw filter's over-extrapolation that the floor is covering up.
    var flooredDays = 0
    var flooredMaxLb = 0.0
}

func percentile(_ xs: [Double], _ p: Double) -> Double {
    guard !xs.isEmpty else { return .nan }
    let s = xs.sorted()
    let idx = min(s.count - 1, max(0, Int((p / 100) * Double(s.count - 1))))
    return s[idx]
}

func finite(_ xs: Double...) -> Bool { xs.allSatisfy { $0.isFinite } }

// Filter tuning can be overridden from the command line so the process-noise parameters can be
// swept against this oracle instead of hand-picked:
//   swift run -c release ScenarioRunner --level 0.01 --slope 4e-4 --trials 1500
let cliArgs = CommandLine.arguments
func cliValue(_ name: String) -> Double? {
    guard let i = cliArgs.firstIndex(of: name), i + 1 < cliArgs.count else { return nil }
    return Double(cliArgs[i + 1])
}
let baseFilter = TrendFilter()
let tunedConfig = AnalysisConfig(trend: TrendFilter(
    measurementVar: baseFilter.measurementVar,
    levelVar: cliValue("--level") ?? baseFilter.levelVar,
    slopeVar: cliValue("--slope") ?? baseFilter.slopeVar))

func evaluate(_ s: Scenario) -> Metrics {
    var m = Metrics()
    let a = Analyzer.analyze(records: s.records, goal: s.goal, config: tunedConfig)
    let n = s.truth.trueWeight.count

    // Trend vs true weight (after a 7-day burn-in).
    var sumSq = 0.0, cnt = 0.0
    for (i, pt) in a.trendSeries.enumerated() where i >= 7 {
        guard let tr = pt.trend else { continue }
        let err = abs(tr - s.truth.trueWeight[i])
        m.trendMaxErr = max(m.trendMaxErr, err)
        sumSq += err * err; cnt += 1
        if !err.isFinite { m.sane = false }
    }
    m.trendRMSE = cnt > 0 ? (sumSq / cnt).squareRoot() : 0

    // Invariant: the trend must stay within the readings the user had actually taken by that
    // day (running min/max, matching how the floor is defined). Compared against a *running*
    // bound, not a global one, so a later low reading cannot retroactively excuse an excursion.
    let obs = s.records.compactMap(\.weightLb)
    var runMin = Double.infinity, runMax = -Double.infinity
    for i in s.records.indices {
        if let z = s.records[i].weightLb, z.isFinite { runMin = min(runMin, z); runMax = max(runMax, z) }
        guard runMin.isFinite, let tr = a.trendSeries[i].trend else { continue }
        m.trendBelowAllReadings = max(m.trendBelowAllReadings, runMin - tr)
        m.trendAboveAllReadings = max(m.trendAboveAllReadings, tr - runMax)
    }
    if let minObs = obs.min(), let lastW = s.records.lastIndex(where: { $0.weightLb != nil }),
       let tr = a.trendSeries[lastW].trend {
        m.lastTrendBelowMinObs = max(0, minObs - tr)
    }

    // Cumulative deficit vs true banked energy over the whole span.
    let trueBanked = (s.truth.trueWeight[0] - s.truth.trueWeight[n - 1]) * kcalPerLb
    m.cumDeficitErr = abs(a.cumulativeDeficitKcal - trueBanked)
    m.cumDeficitErrLb = (a.cumulativeDeficitKcal - trueBanked) / kcalPerLb   // signed
    m.cumDeficitSE = a.cumulativeDeficitSE
    m.cumWithin2SE = a.cumulativeDeficitSE > 0 && m.cumDeficitErr <= 2 * a.cumulativeDeficitSE
    if !finite(a.cumulativeDeficitKcal, trueBanked, a.cumulativeDeficitSE) { m.sane = false }

    if let e = a.effectiveTDEE { m.effErr = abs(e.value - s.truth.effectiveMaintenance); if !e.value.isFinite { m.sane = false } }
    if let b = a.loggingBias { m.biasErr = abs(b.value - s.truth.biasAlpha); if !b.value.isFinite { m.sane = false } }
    if let bud = a.dailyBudgetKcal { m.budget = bud; if !bud.isFinite { m.sane = false } }

    m.flooredDays = a.trendFlooredDays
    m.flooredMaxLb = a.trendFlooredMaxLb

    // Sanity across exposed optional numbers.
    for v in [a.trendWeightLb, a.ratePerWeekLb, a.toGoLb, a.totalChangeLb] {
        if let v, !v.isFinite { m.sane = false }
    }
    return m
}

func randomParams(_ rng: inout SeededGenerator, realism: Bool) -> ScenarioParams {
    func u(_ lo: Double, _ hi: Double) -> Double { lo + rng.uniform() * (hi - lo) }
    var p = ScenarioParams()
    p.days = Int(u(40, 90))
    p.startWeightLb = u(150, 235)
    p.restingTDEE = u(1450, 2400)
    p.activeMean = u(120, 720)
    p.biasAlpha = u(0.95, 1.28)
    p.ratePerWeek = u(0.3, 1.6)
    p.intakeNoise = u(90, 220)
    p.waterAmp = u(1.0, 3.0)
    p.pMissWeigh = u(0.1, 0.55)
    p.pMissLog = u(0.0, 0.25)
    p.hasHealthKit = rng.uniform() > 0.2
    p.hasBasal = p.hasHealthKit && rng.uniform() > 0.4
    p.hasProfile = rng.uniform() > 0.3
    // Non-linear reality: drifting rate, an early glycogen whoosh, diet breaks, and a long
    // weigh-in gap. These are what make the true weight curve bend, and they are what the
    // constant-rate suite is missing. Draw them only for the realistic suite.
    p.rateDriftLbPerWeek = realism ? u(0.1, 1.2) : 0
    p.whooshLb = realism ? u(0, 3.0) : 0
    p.pDietBreak = realism ? u(0, 0.025) : 0
    p.maxWeighGapDays = realism ? Int(u(0, 14)) : 0
    return p
}

/// Pass/fail thresholds for one Monte Carlo suite.
struct Thresholds {
    var maxBudgetBad: Int
    var minCoverage: Double
    var maxCumBias: Double
    var maxTrendRMSEp95: Double
    var maxCumP95: Double
    var maxEffP95: Double
    var maxBiasFailRate: Double
}

/// The historical bar. The model was tuned against the constant-rate suite, so these are the
/// numbers it has always been held to and they are pure regression gates.
let calibrationThresholds = Thresholds(
    maxBudgetBad: 0, minCoverage: 0.90, maxCumBias: 0.2, maxTrendRMSEp95: 1.4,
    maxCumP95: 4.0, maxEffP95: 320, maxBiasFailRate: 0.25)

// The realistic suite is a genuinely harder estimation problem: when a dieter's loss rate
// wanders, `effectiveTDEE` (and therefore logging bias, and therefore the budget) is harder to
// pin down, and a glycogen whoosh moves real mass with no matching energy deficit, so part of
// the logging bias is not identifiable at all.
//
// These accuracy numbers come out worse than the constant-rate suite's *with the model
// unchanged* — run both suites and compare: the same estimator scores effTDEE p95 249 kcal /
// bias 19.0% on constant-rate vs 328 kcal / 33.3% here. That gap is the benchmark being harder,
// not a regression. So these are RECORDED CURRENT PERFORMANCE, not a standard the model meets:
// they exist to catch regressions, and closing the gap is open work on the TDEE/bias estimator,
// not on the trend filter.
let realisticThresholds = Thresholds(
    maxBudgetBad: 3, minCoverage: 0.90, maxCumBias: 0.2, maxTrendRMSEp95: 1.4,
    maxCumP95: 4.3, maxEffP95: 345, maxBiasFailRate: 0.36)

// MARK: - Monte Carlo

let trials = Int(cliValue("--trials") ?? 4000)

func line(_ label: String, _ value: String) {
    print("  \(label.padding(toLength: 30, withPad: " ", startingAt: 0)) \(value)")
}
func stat(_ label: String, _ xs: [Double], unit: String) {
    guard !xs.isEmpty else { line(label, "n/a"); return }
    let mean = xs.reduce(0, +) / Double(xs.count)
    line(label, String(format: "mean %.3f  p95 %.3f  max %.3f %@", mean, percentile(xs, 95), xs.max() ?? 0, unit))
}

/// Run `trials` randomized dieters and report. Returns true if every gate passed.
func runSuite(name: String, realism: Bool, thresholds th: Thresholds) -> Bool {
    var rng = SeededGenerator(seed: 0xCA10C1E5)
    var trendMax: [Double] = [], trendRMSE: [Double] = [], cumLb: [Double] = []
    var effErrs: [Double] = [], biasErrs: [Double] = []
    var insane = 0, budgetBad = 0, cumCovered = 0
    var trendFail = 0, cumFail = 0, effFail = 0, biasFail = 0
    var effApplicable = 0, biasApplicable = 0
    var belowAll: [Double] = [], aboveAll: [Double] = [], lastBelowMin: [Double] = []
    var belowFail = 0, lastBelowFail = 0
    var flooredLift: [Double] = []
    var scenariosFloored = 0, flooredDayTotal = 0, dayTotal = 0

    for t in 0..<trials {
        var localRng = SeededGenerator(seed: rng.next())
        let p = randomParams(&localRng, realism: realism)
        let s = Simulator.generate(p, seed: rng.next())
        let m = evaluate(s)
        let meanTDEE = s.truth.trueTDEE.reduce(0, +) / Double(s.truth.trueTDEE.count)
        let spanOK = p.days >= 22

        trendMax.append(m.trendMaxErr); trendRMSE.append(m.trendRMSE); cumLb.append(m.cumDeficitErrLb)
        if !m.sane { insane += 1 }
        if m.cumWithin2SE { cumCovered += 1 }
        if let b = m.budget, !(b >= 800 && b <= meanTDEE + 300) { budgetBad += 1 }
        if m.trendMaxErr >= 2.0 || m.trendRMSE >= 0.9 { trendFail += 1 }
        belowAll.append(m.trendBelowAllReadings); aboveAll.append(m.trendAboveAllReadings)
        lastBelowMin.append(m.lastTrendBelowMinObs)
        if m.trendBelowAllReadings > 0.001 { belowFail += 1 }
        if m.lastTrendBelowMinObs > 0.001 { lastBelowFail += 1 }
        flooredLift.append(m.flooredMaxLb)
        if m.flooredDays > 0 { scenariosFloored += 1 }
        flooredDayTotal += m.flooredDays; dayTotal += p.days
        if abs(m.cumDeficitErrLb) >= 2.0 { cumFail += 1 }   // absolute: deficit within ~2 lb of truth
        if spanOK, let e = m.effErr { effApplicable += 1; effErrs.append(e); if e >= 250 { effFail += 1 } }
        if spanOK, p.hasBasal, let b = m.biasErr { biasApplicable += 1; biasErrs.append(b); if b >= 0.08 { biasFail += 1 } }
        _ = t
    }
    let cumCoverage = Double(cumCovered) / Double(trials)


    print("\n========== \(name) (\(trials) randomized dieters) ==========")
    stat("Trend vs true weight (max)", trendMax, unit: "lb")
    stat("Trend vs true weight (RMSE)", trendRMSE, unit: "lb")
    stat("Cumulative deficit |error|", cumLb.map { abs($0) }, unit: "lb")
    let cumBias = cumLb.reduce(0, +) / Double(cumLb.count)
    line("Cumulative deficit bias", String(format: "%+.3f lb (signed mean — want ~0)", cumBias))
    stat("effectiveTDEE error", effErrs, unit: "kcal")
    stat("logging-bias error (HK basal)", biasErrs, unit: "α")
    print("  ----------------------------------------------------------")
    line("NaN/Inf scenarios", "\(insane) / \(trials)")
    line("Implausible budget", "\(budgetBad) / \(trials)")
    line("Deficit band honest (±2σ)", String(format: "%.1f%% covered (want ≥90%%)", cumCoverage * 100))
    line("Trend track failures", "\(trendFail) / \(trials)  (<2.0lb max & <0.9 RMSE)")
    line("Cum-deficit failures", "\(cumFail) / \(trials)  (within 2.0 lb of truth)")
    line("effectiveTDEE failures", "\(effFail) / \(effApplicable)  (<250 kcal, span≥22d)")
    line("logging-bias failures", "\(biasFail) / \(biasApplicable)  (<0.08 α, HK basal, span≥22d)")
    stat("Trend below ALL readings", belowAll, unit: "lb")
    stat("Trend above ALL readings", aboveAll, unit: "lb")
    stat("Final trend below min reading", lastBelowMin, unit: "lb")
    line("Scenarios trend<all readings", "\(belowFail) / \(trials)")
    line("Final trend < min reading", "\(lastBelowFail) / \(trials)")
    line("Scale floor engaged", String(format: "%d / %d scenarios  (%.2f%% of all days)",
         scenariosFloored, trials, Double(flooredDayTotal) / Double(max(1, dayTotal)) * 100))
    stat("Scale floor lift", flooredLift, unit: "lb")

    // Gate on what the model can actually control: honesty (calibrated band), no systematic
    // drift, no NaN/absurd output, bounded error. Absolute accuracy is scale-noise-limited and
    // is *reported* via the band rather than gated per-scenario.
    let trendRMSEp95 = percentile(trendRMSE, 95)
    let cumP95 = percentile(cumLb.map { abs($0) }, 95)
    let effP95 = percentile(effErrs, 95)
    let belowRate = Double(belowFail) / Double(trials)
    let lastBelowRate = Double(lastBelowFail) / Double(trials)
    let belowMax = belowAll.max() ?? 0
    let aboveMax = aboveAll.max() ?? 0
    let biasRate = biasApplicable == 0 ? 0 : Double(biasFail) / Double(biasApplicable)

    print("\n---------- GATES: \(name) ----------")
    func gate(_ label: String, _ ok: Bool, _ value: String) -> Bool {
        line(label, "\(ok ? "PASS" : "FAIL") (\(value))")
        return ok
    }
    var ok = true
    ok = gate("no NaN / absurd budget", insane == 0 && budgetBad <= th.maxBudgetBad,
              "\(insane) NaN, \(budgetBad) budget") && ok
    ok = gate("deficit band honest", cumCoverage >= th.minCoverage,
              String(format: "%.1f%% >= %.0f%%", cumCoverage * 100, th.minCoverage * 100)) && ok
    ok = gate("deficit bias ~0", abs(cumBias) < th.maxCumBias,
              String(format: "%+.3f lb", cumBias)) && ok
    ok = gate("trend RMSE p95", trendRMSEp95 < th.maxTrendRMSEp95,
              String(format: "%.3f < %.2f lb", trendRMSEp95, th.maxTrendRMSEp95)) && ok
    ok = gate("deficit |err| p95", cumP95 < th.maxCumP95,
              String(format: "%.3f < %.2f lb", cumP95, th.maxCumP95)) && ok
    ok = gate("effTDEE p95", effP95 < th.maxEffP95,
              String(format: "%.0f < %.0f kcal", effP95, th.maxEffP95)) && ok
    ok = gate("bias fail rate", biasRate < th.maxBiasFailRate,
              String(format: "%.1f%% < %.0f%%", biasRate * 100, th.maxBiasFailRate * 100)) && ok
    // The bug this release fixes: the trend must never sit below every reading taken so far.
    // These are absolute — 0 tolerance — on both suites.
    ok = gate("trend never < all readings", belowRate == 0,
              String(format: "%.2f%%", belowRate * 100)) && ok
    ok = gate("final trend >= min reading", lastBelowRate == 0,
              String(format: "%.2f%%", lastBelowRate * 100)) && ok
    ok = gate("worst below-excursion == 0", belowMax < 0.001,
              String(format: "%.3f lb", belowMax)) && ok
    // Not gated: the floor is one-sided, so the mirror-image excursion (trend above every
    // reading) is still possible. Tracked so it stays visible rather than silently unmeasured.
    line("(ungated) worst above-excursion", String(format: "%.3f lb", aboveMax))
    print("  ---> \(name): \(ok ? "PASS ✅" : "FAIL ❌")")
    return ok
}

// The constant-rate suite is the historical benchmark the model was tuned on — pure regression
// gates. The realistic suite adds the non-linear reality (drifting rate, glycogen whoosh, diet
// breaks, long weigh-in gaps) that a straight-line simulator cannot produce, and which is what
// let the "trend below every reading" bug survive to the field.
let calibrationPass = runSuite(name: "CONSTANT-RATE SUITE (historical calibration)",
                               realism: false, thresholds: calibrationThresholds)
let realisticPass = runSuite(name: "REALISTIC SUITE (non-linear dieters)",
                             realism: true, thresholds: realisticThresholds)
let pass = calibrationPass && realisticPass

// MARK: - Targeted stress tests

print("\n========== STRESS TESTS ==========")

func report(_ name: String, _ s: Scenario, focus: (Analysis) -> String) {
    let a = Analyzer.analyze(records: s.records, goal: s.goal, config: tunedConfig)
    print("\n• \(name)")
    print("  " + focus(a))
}

// 1) Two-week logging gap mid-diet, then resume — does the deficit reconcile to the scale?
do {
    var p = ScenarioParams(); p.days = 60; p.pMissLog = 0.05
    var s = Simulator.generate(p, seed: 42)
    for i in 25..<39 { s.records[i].loggedKcal = nil }   // wipe two weeks of logs
    let trueBanked = (s.truth.trueWeight[0] - s.truth.trueWeight[59]) * kcalPerLb
    report("14-day logging blackout (scale fills the gap)", s) {
        String(format: "cumДeficit %.0f kcal vs true banked %.0f kcal  (err %.1f%%)  loss %.2f lb",
               $0.cumulativeDeficitKcal, trueBanked,
               abs($0.cumulativeDeficitKcal - trueBanked) / abs(trueBanked) * 100, $0.lossSoFarLb)
    }
}

// 2) Massive water spike — trend must stay calm.
do {
    var p = ScenarioParams(); p.days = 50; p.waterAmp = 1.2
    var s = Simulator.generate(p, seed: 7)
    let before = Analyzer.analyze(records: s.records, goal: s.goal, config: tunedConfig).trendWeightLb ?? 0
    if let w = s.records[48].weightLb { s.records[48].weightLb = w + 5.0 }  // +5 lb salt bomb
    let after = Analyzer.analyze(records: s.records, goal: s.goal, config: tunedConfig).trendWeightLb ?? 0
    print("\n• +5 lb single-day water spike")
    print(String(format: "  trend moved %.2f lb (raw reading moved +5.0 lb) — filtered %.0f%%",
                 after - before, (1 - (after - before) / 5.0) * 100))
}

// 3) No HealthKit, no profile — bias must be reported as unknown, not fabricated.
do {
    var p = ScenarioParams(); p.days = 56; p.hasHealthKit = false; p.hasBasal = false; p.hasProfile = false
    let s = Simulator.generate(p, seed: 99)
    report("No HealthKit / no profile (bias unidentifiable)", s) {
        "loggingBias = \($0.loggingBias.map { String(format: "%.3f", $0.value) } ?? "nil (correct)")  " +
        "budget calibrated = \($0.budgetIsCalibrated)  budget = \($0.dailyBudgetKcal.map { String(Int($0)) } ?? "nil")"
    }
}

// 4) Brand-new user, 5 days only — must not over-claim.
do {
    var p = ScenarioParams(); p.days = 5
    let s = Simulator.generate(p, seed: 3)
    report("Cold start (5 days)", s) {
        "effectiveTDEE = \($0.effectiveTDEE.map { String(Int($0.value)) } ?? "nil")  " +
        "budget = \($0.dailyBudgetKcal.map { String(Int($0)) } ?? "nil") (calibrated=\($0.budgetIsCalibrated))  " +
        "cumDeficit = \(Int($0.cumulativeDeficitKcal)) kcal"
    }
}

// 5) Whoosh-then-plateau + a 10-day weigh-in gap: the real-world shape that exposed the
//    over-stiff filter. After the gap the single new reading must dominate the projection —
//    the trend has to snap back to the scale, not sail on below every reading ever taken.
do {
    var p = ScenarioParams()
    p.days = 34; p.whooshLb = 3.0; p.rateDriftLbPerWeek = 1.0; p.pMissWeigh = 0.05
    var s = Simulator.generate(p, seed: 2026)
    for i in 24..<33 { s.records[i].weightLb = nil }        // 9 blank days before the last reading
    let a = Analyzer.analyze(records: s.records, goal: s.goal, config: tunedConfig)
    let obs = s.records.compactMap(\.weightLb)
    let lastReading = obs.last ?? .nan
    print("\n• Whoosh + plateau + 10-day weigh-in gap")
    print(String(format: "  trend %.2f  vs last reading %.2f (gap %+.2f)  min reading %.2f  %@",
                 a.trendWeightLb ?? .nan, lastReading, (a.trendWeightLb ?? .nan) - lastReading,
                 obs.min() ?? .nan,
                 (a.trendWeightLb ?? .nan) >= (obs.min() ?? .nan) ? "inside range ✅" : "BELOW EVERY READING ❌"))
}

// 6) Plateau eater (maintenance, alpha=1) — deficit must hover near zero, not drift.
do {
    var p = ScenarioParams(); p.days = 56; p.ratePerWeek = 0.0; p.biasAlpha = 1.0
    // A true maintenance eater: no glycogen whoosh, no rate drift, no diet breaks — otherwise
    // this scenario has real weight change in it and "deficit ≈ 0" is the wrong expectation.
    p.whooshLb = 0; p.rateDriftLbPerWeek = 0; p.pDietBreak = 0
    let s = Simulator.generate(p, seed: 123)
    report("True maintenance (should read ~0 deficit)", s) {
        String(format: "cumDeficit %.0f kcal  loss %.2f lb  rate %.2f lb/wk",
               $0.cumulativeDeficitKcal, $0.lossSoFarLb, $0.ratePerWeekLb ?? .nan)
    }
}



print("\n========== \(pass ? "PASS ✅" : "FAIL ❌") ==========\n")
exit(pass ? 0 : 1)
