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
