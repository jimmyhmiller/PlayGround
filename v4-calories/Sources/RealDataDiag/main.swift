import Foundation
import CalorieModel

// Diagnose the model against a REAL app data container, pulled off the device with:
//
//   xcrun devicectl device copy from --device <UDID> --domain-type appDataContainer \
//     --domain-identifier com.jimmyhmiller.CumulativeTracker \
//     --source "Library/Application Support/cumulative-tracker.json" --destination <file>
//
// Usage: swift run RealDataDiag <cumulative-tracker.json> [--level V] [--slope V] [--sweep]
//
// Prints the per-day trend next to the observed readings plus the headline figures, so a
// "the app says X but my scale says Y" report can be reproduced outside the phone.

struct SavedState: Decodable {
    var startDate: Date?
    var weighIns: [WeighIn]
    var goal: Goal
    var entries: [CalorieEntry]
}

let args = CommandLine.arguments
guard args.count > 1 else {
    FileHandle.standardError.write(Data("usage: RealDataDiag <cumulative-tracker.json> [--level V] [--slope V] [--sweep]\n".utf8))
    exit(2)
}

func flag(_ name: String) -> Double? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return Double(args[i + 1])
}
let sweep = args.contains("--sweep")

let url = URL(fileURLWithPath: args[1])
let data = try Data(contentsOf: url)
let state = try JSONDecoder().decode(SavedState.self, from: data)

let cal = Calendar.current
let start = state.startDate.map { cal.startOfDay(for: $0) } ?? cal.startOfDay(for: Date())
// End the series at the last weigh-in's day, so results are reproducible whenever this is run.
let end = cal.startOfDay(for: state.weighIns.map(\.date).max() ?? start)

let records = DailyRecord.contiguous(from: start, to: end, calendar: cal,
                                     entries: state.entries, weighIns: state.weighIns)

let defaults = TrendFilter()
func makeFilter(level: Double, slope: Double) -> TrendFilter {
    TrendFilter(measurementVar: defaults.measurementVar, levelVar: level, slopeVar: slope)
}

let df = DateFormatter()
df.dateFormat = "yyyy-MM-dd"
let obs = records.compactMap(\.weightLb)
let minObs = obs.min() ?? .nan
let maxObs = obs.max() ?? .nan

func run(_ filter: TrendFilter) -> Analysis {
    Analyzer.analyze(records: records, goal: state.goal,
                     config: AnalysisConfig(trend: filter))
}

if sweep {
    // How far outside the observed range does the smoothed trend land, as a function of how
    // much freedom the level/slope states are given? A well-specified smoother stays inside.
    print("levelVar  slopeVar   trend@last  minObs  belowMin  maxDeviationOutsideRange  rate/wk")
    for level in [0.004, 0.01, 0.02, 0.04] {
        for slope in [8e-6, 5e-5, 2e-4, 5e-4, 1e-3, 4e-3] {
            let a = run(makeFilter(level: level, slope: slope))
            let t = a.trendWeightLb ?? .nan
            var worst = 0.0
            for p in a.trendSeries {
                guard let tr = p.trend else { continue }
                worst = max(worst, max(0, minObs - tr), max(0, tr - maxObs))
            }
            print(String(format: "%7.4f  %8.1e  %10.2f  %6.2f  %8.2f  %22.2f  %+.2f",
                         level, slope, t, minObs, t - minObs, worst, a.ratePerWeekLb ?? .nan))
        }
    }
    exit(0)
}

let filter = makeFilter(level: flag("--level") ?? defaults.levelVar,
                        slope: flag("--slope") ?? defaults.slopeVar)
let analysis = run(filter)

print("levelVar = \(filter.levelVar)  slopeVar = \(filter.slopeVar)")
print("waterVar R = \(String(format: "%.3f", filter.estimateWaterVar(records) ?? -1)) lb^2")
print("")
print("date         observed   trend    slope/wk")
for (i, p) in analysis.trendSeries.enumerated() {
    let ob = p.observed.map { String(format: "%7.1f", $0) } ?? "      -"
    let tr = p.trend.map { String(format: "%8.2f", $0) } ?? "       -"
    let sl = analysis.trendSeries.indices.contains(i) ? "" : ""
    print("\(df.string(from: p.date))  \(ob)  \(tr)\(sl)")
}

print("")
print(String(format: "min observed reading : %.2f lb", minObs))
print(String(format: "max observed reading : %.2f lb", maxObs))
print(String(format: "last observed reading: %.2f lb", obs.last ?? .nan))
print(String(format: "trend weight         : %.2f lb", analysis.trendWeightLb ?? .nan))
print(String(format: "start weight         : %.2f lb", analysis.startWeightLb ?? .nan))
print(String(format: "total change         : %+.2f lb", analysis.totalChangeLb ?? .nan))
print(String(format: "rate                 : %+.2f lb/wk", analysis.ratePerWeekLb ?? .nan))
print(String(format: "days since weigh-in  : %d", analysis.daysSinceLastWeighIn ?? -1))

if let t = analysis.trendWeightLb, t < minObs - 0.001 {
    print(String(format: "\n*** trend (%.2f) is BELOW every scale reading (min %.2f) ***", t, minObs))
}
