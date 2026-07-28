import XCTest
@testable import CalorieModel

/// Persisted-data must survive new fields being added to the models. Auto-synthesized
/// Decodable throws on a missing key even with a default; these prove our hand-written
/// decoders tolerate older JSON (the bug that wiped real data once).
final class BackwardCompatCodingTests: XCTestCase {

    private func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(json.utf8))
    }

    func testCalorieEntryMissingLabel() throws {
        // Oldest format: no `label`, no `id`.
        let e = try decode(CalorieEntry.self, #"{"date": 0, "kcal": 540}"#)
        XCTAssertEqual(e.kcal, 540)
        XCTAssertNil(e.label)
    }

    func testWeighInMissingFromHealthKit() throws {
        // `fromHealthKit` was added later; old rows omit it.
        let w = try decode(WeighIn.self, #"{"id": "5B7C0E2A-0000-0000-0000-000000000001", "date": 0, "weightLb": 183.4}"#)
        XCTAssertEqual(w.weightLb, 183.4, accuracy: 0.0001)
        XCTAssertFalse(w.fromHealthKit)
    }

    func testGoalMissingProfileFields() throws {
        let g = try decode(Goal.self, #"{"targetWeightLb": 178, "ratePerWeek": 1.0}"#)
        XCTAssertEqual(g.targetWeightLb, 178)
        XCTAssertNil(g.sex)
        XCTAssertNil(g.heightCm)
    }

    func testRoundTripStillWorks() throws {
        let e = CalorieEntry(date: Date(timeIntervalSince1970: 100), kcal: 321, label: "Lunch")
        let data = try JSONEncoder().encode(e)
        let back = try JSONDecoder().decode(CalorieEntry.self, from: data)
        XCTAssertEqual(back, e)
    }
}

final class TrendFilterTests: XCTestCase {

    func testEmptyAndNoWeighInsAreSafe() {
        XCTAssertTrue(TrendFilter().run([]).isEmpty)
        let days = DailyRecord.contiguous(from: Date(timeIntervalSince1970: 0),
                                          to: Date(timeIntervalSince1970: 5 * 86400))
        let pts = TrendFilter().run(days)
        XCTAssertTrue(pts.allSatisfy { $0.trend == nil })   // nothing to anchor to
    }

    func testWaterSpikeIsFiltered() {
        var p = ScenarioParams(); p.days = 50; p.waterAmp = 1.2; p.pMissWeigh = 0
        var s = Simulator.generate(p, seed: 7)
        let before = TrendFilter().run(s.records)
        let beforeTrend = before.last?.trend ?? 0
        // Inject a +5 lb single-day salt bomb on the penultimate day.
        s.records[48].weightLb = (s.records[48].weightLb ?? 0) + 5
        let after = TrendFilter().run(s.records)
        let moved = abs((after.last?.trend ?? 0) - beforeTrend)
        XCTAssertLessThan(moved, 1.5, "A 5 lb water spike must not move the trend more than ~1.5 lb")
    }

    func testGapGrowsUncertainty() {
        var p = ScenarioParams(); p.days = 40; p.pMissWeigh = 0
        var s = Simulator.generate(p, seed: 11)
        for i in 20..<39 { s.records[i].weightLb = nil }   // long gap, last day still has a reading
        let pts = TrendFilter().run(s.records)
        // Variance mid-gap should exceed variance right after a reading.
        let vAfterReading = pts[19].variance ?? 0
        let vMidGap = pts[30].variance ?? 0
        XCTAssertGreaterThan(vMidGap, vAfterReading)
    }
}

/// Regression tests for the scale floor: the trend must never claim a weight lower than every
/// reading the scale has produced so far.
///
/// The bug (reported 2026-07-28 from real device data): a fast first week followed by a plateau
/// and then a 10-day weigh-in gap left the local-linear-trend filter extending the early rate in
/// a straight line, so it reported 209.5 lb when the lowest number the scale had ever shown was
/// 210.3 lb — and the fresh 210.8 lb reading barely moved it.
final class ScaleFloorTests: XCTestCase {

    /// Jimmy's actual weigh-ins from the device container, 2026-06-26 .. 2026-07-28.
    /// Day index (from the first weigh-in) → lb. Note the 9-day hole before the last reading.
    private static let realSeries: [(day: Int, lb: Double)] = [
        (0, 216.3), (1, 215.6), (2, 214.7), (3, 213.9), (4, 212.5), (5, 212.3), (6, 213.0),
        (7, 211.7), (8, 213.6), (9, 213.2), (10, 213.1), (11, 212.4), (12, 212.9), (13, 213.1),
        (14, 211.9), (15, 211.8), (17, 211.2), (18, 210.9), (19, 211.0), (20, 211.7),
        (21, 210.9), (22, 210.3), (32, 210.8),
    ]

    private func realRecords() -> [DailyRecord] {
        let base = Date(timeIntervalSince1970: 1_700_000_000)
        let cal = Calendar.current
        let day: (Int) -> Date = { cal.startOfDay(for: base.addingTimeInterval(Double($0) * 86400)) }
        let byDay = Dictionary(uniqueKeysWithValues: Self.realSeries.map { ($0.day, $0.lb) })
        return (0...32).map { DailyRecord(date: day($0), weightLb: byDay[$0]) }
    }

    func testRealSeriesTrendIsNeverBelowEveryReading() {
        let records = realRecords()
        let pts = TrendFilter().run(records)
        let minReading = records.compactMap(\.weightLb).min()!
        let trend = pts.compactMap(\.trend).last!
        XCTAssertGreaterThanOrEqual(
            trend, minReading - 1e-9,
            "Trend weight \(trend) must not be below the lowest scale reading \(minReading)")
    }

    /// The floor is the *running* minimum, so it must hold at every point in the series, not just
    /// at the end — the chart line must never dip under the readings drawn beside it either.
    func testRealSeriesTrendIsNeverBelowRunningMinimum() {
        let records = realRecords()
        let pts = TrendFilter().run(records)
        var runningMin = Double.infinity
        for (i, p) in pts.enumerated() {
            if let z = p.observed { runningMin = min(runningMin, z) }
            guard let t = p.trend, runningMin.isFinite else { continue }
            XCTAssertGreaterThanOrEqual(t, runningMin - 1e-9,
                                        "day \(i): trend \(t) dipped below running min \(runningMin)")
        }
    }

    /// Without the fix this series produced 209.52 lb. Pin the corrected value so a future
    /// retune cannot silently walk it back under the readings.
    func testRealSeriesTrendMatchesTheFloor() {
        let pts = TrendFilter().run(realRecords())
        XCTAssertEqual(pts.compactMap(\.trend).last!, 210.3, accuracy: 0.01)
    }

    /// How much of the trend is the floor actually responsible for?
    ///
    /// Not a no-op, and it is worth being precise about that: even for the easiest possible user
    /// — weighing in every single day, losing at a constant rate — the raw filter dips under the
    /// running minimum on ~7% of days (measured over 200 seeds: p95 20%, worst 29%), lifting the
    /// trend by up to ~2.1 lb. That is the filter's own over-extrapolation showing through, most
    /// often in the first few days and during any run of readings the smoother cuts under.
    ///
    /// So this test does NOT claim the floor is rare. It pins the scale of its involvement, so a
    /// future retune that makes the floor take over the trend entirely fails loudly.
    func testFloorInvolvementStaysBounded() {
        var fractions: [Double] = []
        var lifts: [Double] = []
        for seed in UInt64(1)...200 {
            var p = ScenarioParams(); p.days = 45; p.pMissWeigh = 0
            p.whooshLb = 0; p.rateDriftLbPerWeek = 0; p.pDietBreak = 0
            let s = Simulator.generate(p, seed: seed)
            let a = Analyzer.analyze(records: s.records, goal: s.goal)
            fractions.append(Double(a.trendFlooredDays) / Double(p.days))
            lifts.append(a.trendFlooredMaxLb)
        }
        let meanFraction = fractions.reduce(0, +) / Double(fractions.count)
        XCTAssertLessThan(meanFraction, 0.15,
                          "the floor should be correcting the trend, not defining it")
        XCTAssertLessThan(fractions.max()!, 0.40,
                          "no single steady dieter should have the floor own most of their trend")
        XCTAssertLessThan(lifts.max()!, 3.0,
                          "a lift this large means the filter is badly out, not being nudged")
    }

    /// A gainer must not be pinned up to an old low: the floor uses the running minimum, so a
    /// weight the user has since regained past cannot drag the trend with it.
    func testFloorDoesNotPinAGainerToAnOldLow() {
        let base = Date(timeIntervalSince1970: 1_700_000_000)
        let cal = Calendar.current
        // Dip to 180, then climb steadily to 195.
        let weights: [Double] = [190, 186, 183, 180, 182, 185, 187, 189, 191, 193, 195]
        let records = weights.enumerated().map { i, w in
            DailyRecord(date: cal.startOfDay(for: base.addingTimeInterval(Double(i) * 86400)),
                        weightLb: w)
        }
        let pts = TrendFilter().run(records)
        let last = pts.compactMap(\.trend).last!
        XCTAssertGreaterThan(last, 185, "trend should follow the regain, not sit at the old low of 180")
    }

    /// The reported cumulative loss must not exceed what the scale can support: with the floor
    /// engaged, start − trend can be no larger than start − (lowest reading).
    func testCumulativeLossIsCappedByTheScale() {
        let records = realRecords()
        let a = Analyzer.analyze(records: records, goal: Goal(targetWeightLb: 178, ratePerWeek: 1.25))
        let minReading = records.compactMap(\.weightLb).min()!
        let start = a.startWeightLb!
        XCTAssertLessThanOrEqual(-a.totalChangeLb!, start - minReading + 1e-9,
                                 "reported loss must not exceed start minus the lowest reading")
    }
}

final class EstimatorTests: XCTestCase {

    func testColdStartDoesNotOverclaim() {
        var p = ScenarioParams(); p.days = 5
        let s = Simulator.generate(p, seed: 3)
        let a = Analyzer.analyze(records: s.records, goal: s.goal)
        XCTAssertNil(a.effectiveTDEE, "5 days is too little to estimate TDEE")
        XCTAssertFalse(a.budgetIsCalibrated, "budget must be flagged provisional")
        XCTAssertNotNil(a.dailyBudgetKcal, "but a provisional budget should still exist")
    }

    func testBiasUnknownWithoutAnchor() {
        var p = ScenarioParams(); p.days = 56
        p.hasHealthKit = false; p.hasBasal = false; p.hasProfile = false
        let s = Simulator.generate(p, seed: 99)
        let a = Analyzer.analyze(records: s.records, goal: s.goal)
        XCTAssertNil(a.loggingBias, "bias is unidentifiable without HealthKit or a profile — must be nil")
    }

    func testEffectiveTDEERecoversTruth() {
        var p = ScenarioParams(); p.days = 70; p.biasAlpha = 1.12
        let s = Simulator.generate(p, seed: 21)
        let a = Analyzer.analyze(records: s.records, goal: s.goal)
        let e = try! XCTUnwrap(a.effectiveTDEE)
        XCTAssertEqual(e.value, s.truth.effectiveMaintenance, accuracy: 220,
                       "effectiveTDEE should land near true logged-unit maintenance")
    }
}

final class ReconciliationTests: XCTestCase {

    func testScaleFillsLoggingGap() {
        var p = ScenarioParams(); p.days = 60; p.pMissWeigh = 0.2
        var s = Simulator.generate(p, seed: 42)
        for i in 25..<39 { s.records[i].loggedKcal = nil }   // two weeks of no logging
        let a = Analyzer.analyze(records: s.records, goal: s.goal)
        let trueBanked = (s.truth.trueWeight[0] - s.truth.trueWeight[59]) * kcalPerLb
        let errLb = abs(a.cumulativeDeficitKcal - trueBanked) / kcalPerLb
        XCTAssertLessThan(errLb, 2.5, "the scale should reconcile the deficit despite missing logs")
    }

    func testNoNaNsAcrossManyScenarios() {
        var rng = SeededGenerator(seed: 0xABCDEF)
        for _ in 0..<300 {
            var p = ScenarioParams()
            p.days = 30 + Int(rng.uniform() * 60)
            p.waterAmp = 1 + rng.uniform() * 2
            p.pMissWeigh = rng.uniform() * 0.5
            p.hasHealthKit = rng.uniform() > 0.3
            let s = Simulator.generate(p, seed: rng.next())
            let a = Analyzer.analyze(records: s.records, goal: s.goal)
            XCTAssertTrue(a.cumulativeDeficitKcal.isFinite)
            XCTAssertTrue(a.cumulativeDeficitSE.isFinite && a.cumulativeDeficitSE >= 0)
            if let b = a.dailyBudgetKcal { XCTAssertTrue(b.isFinite && b >= 800) }
            for v in [a.trendWeightLb, a.ratePerWeekLb, a.toGoLb] where v != nil {
                XCTAssertTrue(v!.isFinite)
            }
        }
    }
}

/// Monte-Carlo guardrails: a smaller mirror of ScenarioRunner so `swift test` enforces the
/// core invariants (honest band, no bias, no NaN) in CI.
final class MonteCarloGuardrailTests: XCTestCase {
    func testHonestBandAndNoBias() {
        var rng = SeededGenerator(seed: 0xCA1)
        var covered = 0, total = 0
        var signedSum = 0.0
        for _ in 0..<800 {
            var p = ScenarioParams()
            p.days = 40 + Int(rng.uniform() * 50)
            p.startWeightLb = 150 + rng.uniform() * 80
            p.restingTDEE = 1500 + rng.uniform() * 900
            p.biasAlpha = 0.95 + rng.uniform() * 0.3
            p.ratePerWeek = 0.3 + rng.uniform() * 1.3
            p.waterAmp = 1 + rng.uniform() * 2
            p.pMissWeigh = 0.1 + rng.uniform() * 0.4
            let s = Simulator.generate(p, seed: rng.next())
            let a = Analyzer.analyze(records: s.records, goal: s.goal)
            let trueBanked = (s.truth.trueWeight[0] - s.truth.trueWeight[p.days - 1]) * kcalPerLb
            let err = a.cumulativeDeficitKcal - trueBanked
            signedSum += err / kcalPerLb
            if a.cumulativeDeficitSE > 0 && abs(err) <= 2 * a.cumulativeDeficitSE { covered += 1 }
            total += 1
        }
        let coverage = Double(covered) / Double(total)
        let bias = signedSum / Double(total)
        XCTAssertGreaterThanOrEqual(coverage, 0.88, "reported ±2σ band must cover the truth")
        XCTAssertLessThan(abs(bias), 0.25, "cumulative deficit must be unbiased")
    }
}
