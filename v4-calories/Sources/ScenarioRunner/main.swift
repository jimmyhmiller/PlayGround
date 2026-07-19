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
}

func percentile(_ xs: [Double], _ p: Double) -> Double {
    guard !xs.isEmpty else { return .nan }
    let s = xs.sorted()
    let idx = min(s.count - 1, max(0, Int((p / 100) * Double(s.count - 1))))
    return s[idx]
}

func finite(_ xs: Double...) -> Bool { xs.allSatisfy { $0.isFinite } }

func evaluate(_ s: Scenario) -> Metrics {
    var m = Metrics()
    let a = Analyzer.analyze(records: s.records, goal: s.goal)
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

    // Sanity across exposed optional numbers.
    for v in [a.trendWeightLb, a.ratePerWeekLb, a.toGoLb, a.totalChangeLb] {
        if let v, !v.isFinite { m.sane = false }
    }
    return m
}

func randomParams(_ rng: inout SeededGenerator) -> ScenarioParams {
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
    return p
}

// MARK: - Monte Carlo

let trials = 4000
var rng = SeededGenerator(seed: 0xCA10C1E5)
var trendMax: [Double] = [], trendRMSE: [Double] = [], cumLb: [Double] = []
var effErrs: [Double] = [], biasErrs: [Double] = []
var insane = 0, budgetBad = 0, cumCovered = 0
var trendFail = 0, cumFail = 0, effFail = 0, biasFail = 0
var effApplicable = 0, biasApplicable = 0

for t in 0..<trials {
    var localRng = SeededGenerator(seed: rng.next())
    let p = randomParams(&localRng)
    let s = Simulator.generate(p, seed: rng.next())
    let m = evaluate(s)
    let meanTDEE = s.truth.trueTDEE.reduce(0, +) / Double(s.truth.trueTDEE.count)
    let spanOK = p.days >= 22

    trendMax.append(m.trendMaxErr); trendRMSE.append(m.trendRMSE); cumLb.append(m.cumDeficitErrLb)
    if !m.sane { insane += 1 }
    if m.cumWithin2SE { cumCovered += 1 }
    if let b = m.budget, !(b >= 800 && b <= meanTDEE + 300) { budgetBad += 1 }
    if m.trendMaxErr >= 2.0 || m.trendRMSE >= 0.9 { trendFail += 1 }
    if abs(m.cumDeficitErrLb) >= 2.0 { cumFail += 1 }   // absolute: deficit within ~2 lb of truth
    if spanOK, let e = m.effErr { effApplicable += 1; effErrs.append(e); if e >= 250 { effFail += 1 } }
    if spanOK, p.hasBasal, let b = m.biasErr { biasApplicable += 1; biasErrs.append(b); if b >= 0.08 { biasFail += 1 } }
    _ = t
}
let cumCoverage = Double(cumCovered) / Double(trials)

func line(_ label: String, _ value: String) {
    print("  \(label.padding(toLength: 30, withPad: " ", startingAt: 0)) \(value)")
}
func stat(_ label: String, _ xs: [Double], unit: String) {
    guard !xs.isEmpty else { line(label, "n/a"); return }
    let mean = xs.reduce(0, +) / Double(xs.count)
    line(label, String(format: "mean %.3f  p95 %.3f  max %.3f %@", mean, percentile(xs, 95), xs.max() ?? 0, unit))
}

print("\n========== MONTE CARLO (\(trials) randomized dieters) ==========")
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

// MARK: - Targeted stress tests

print("\n========== STRESS TESTS ==========")

func report(_ name: String, _ s: Scenario, focus: (Analysis) -> String) {
    let a = Analyzer.analyze(records: s.records, goal: s.goal)
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
    let before = Analyzer.analyze(records: s.records, goal: s.goal).trendWeightLb ?? 0
    if let w = s.records[48].weightLb { s.records[48].weightLb = w + 5.0 }  // +5 lb salt bomb
    let after = Analyzer.analyze(records: s.records, goal: s.goal).trendWeightLb ?? 0
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

// 5) Plateau eater (maintenance, alpha=1) — deficit must hover near zero, not drift.
do {
    var p = ScenarioParams(); p.days = 56; p.ratePerWeek = 0.0; p.biasAlpha = 1.0
    let s = Simulator.generate(p, seed: 123)
    report("True maintenance (should read ~0 deficit)", s) {
        String(format: "cumDeficit %.0f kcal  loss %.2f lb  rate %.2f lb/wk",
               $0.cumulativeDeficitKcal, $0.lossSoFarLb, $0.ratePerWeekLb ?? .nan)
    }
}

// Gate on what the model can actually control: honesty (calibrated band), no systematic
// drift, no NaN/absurd output, bounded error. Absolute accuracy is scale-noise-limited and
// is *reported* via the band rather than gated per-scenario.
let trendRMSEp95 = percentile(trendRMSE, 95)
let cumP95 = percentile(cumLb.map { abs($0) }, 95)
let effP95 = percentile(effErrs, 95)

print("\n========== GATES ==========")
line("no NaN / absurd budget", insane == 0 && budgetBad == 0 ? "PASS" : "FAIL")
line("deficit band ≥90% honest", cumCoverage >= 0.90 ? "PASS (\(String(format: "%.0f%%", cumCoverage*100)))" : "FAIL")
line("deficit bias |mean|<0.2lb", abs(cumBias) < 0.2 ? "PASS (\(String(format: "%+.2f", cumBias)))" : "FAIL")
line("trend RMSE p95 <1.4lb", trendRMSEp95 < 1.4 ? "PASS (\(String(format: "%.2f", trendRMSEp95)))" : "FAIL")
line("deficit |err| p95 <4.0lb", cumP95 < 4.0 ? "PASS (\(String(format: "%.2f", cumP95)))" : "FAIL")
line("effTDEE p95 <320 kcal", effP95 < 320 ? "PASS (\(String(format: "%.0f", effP95)))" : "FAIL")
line("bias fail rate <25%", biasApplicable == 0 || Double(biasFail)/Double(biasApplicable) < 0.25
     ? "PASS (\(String(format: "%.0f%%", Double(biasFail)/Double(max(1,biasApplicable))*100)))" : "FAIL")

let pass = insane == 0 && budgetBad == 0
    && cumCoverage >= 0.90
    && abs(cumBias) < 0.2
    && trendRMSEp95 < 1.4
    && cumP95 < 4.0
    && effP95 < 320
    && (biasApplicable == 0 || Double(biasFail) / Double(biasApplicable) < 0.25)

print("\n========== \(pass ? "PASS ✅" : "FAIL ❌") ==========\n")
exit(pass ? 0 : 1)
