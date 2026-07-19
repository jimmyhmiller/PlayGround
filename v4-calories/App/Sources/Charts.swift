import SwiftUI
import Charts
import CalorieModel

/// Small filled sparkline of the cumulative deficit (Today screen).
struct DeficitSparkline: View {
    let series: [DatedValue]
    var body: some View {
        Chart(series, id: \.date) { p in
            AreaMark(x: .value("d", p.date), y: .value("kcal", p.value))
                .foregroundStyle(Theme.green.opacity(0.12))
            LineMark(x: .value("d", p.date), y: .value("kcal", p.value))
                .foregroundStyle(Theme.green)
                .lineStyle(StrokeStyle(lineWidth: 1.6, lineJoin: .round))
        }
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartLegend(.hidden)
    }
}

/// Reconciliation: trend (green), raw scale points (faint), logs-only (amber dashed).
struct ReconciliationChart: View {
    let series: [WeightPoint]

    private var yDomain: ClosedRange<Double> {
        let vals = series.flatMap { [$0.trend, $0.observed, $0.logsOnly].compactMap { $0 } }
        guard let lo = vals.min(), let hi = vals.max(), hi > lo else { return 0...1 }
        let pad = max(0.8, (hi - lo) * 0.08)
        return (lo - pad)...(hi + pad)
    }

    var body: some View {
        Chart {
            ForEach(series.filter { $0.logsOnly != nil }, id: \.date) { p in
                LineMark(x: .value("d", p.date), y: .value("lb", p.logsOnly!),
                         series: .value("s", "logs"))
                    .foregroundStyle(Theme.amber.opacity(0.85))
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [3, 3]))
            }
            ForEach(series.filter { $0.observed != nil }, id: \.date) { p in
                PointMark(x: .value("d", p.date), y: .value("lb", p.observed!))
                    .foregroundStyle(Theme.textDim(0.38))
                    .symbolSize(10)
            }
            ForEach(series.filter { $0.trend != nil }, id: \.date) { p in
                LineMark(x: .value("d", p.date), y: .value("lb", p.trend!),
                         series: .value("s", "trend"))
                    .foregroundStyle(Theme.green)
                    .lineStyle(StrokeStyle(lineWidth: 2.2, lineCap: .round, lineJoin: .round))
            }
        }
        .chartYScale(domain: yDomain)
        .chartXAxis(.hidden)
        .chartYAxis {
            AxisMarks(position: .trailing, values: .automatic(desiredCount: 3)) { _ in
                AxisGridLine().foregroundStyle(Color.white.opacity(0.05))
                AxisValueLabel().foregroundStyle(Theme.textDim(0.35)).font(.mono(9))
            }
        }
        .chartLegend(.hidden)
    }
}

/// Cumulative deficit area chart with a zero baseline (Trends).
struct DeficitAreaChart: View {
    let series: [DatedValue]
    var body: some View {
        Chart {
            RuleMark(y: .value("zero", 0)).foregroundStyle(Color.white.opacity(0.08))
                .lineStyle(StrokeStyle(lineWidth: 1))
            ForEach(series, id: \.date) { p in
                AreaMark(x: .value("d", p.date), y: .value("kcal", p.value))
                    .foregroundStyle(Theme.green.opacity(0.10))
                LineMark(x: .value("d", p.date), y: .value("kcal", p.value))
                    .foregroundStyle(Theme.green)
                    .lineStyle(StrokeStyle(lineWidth: 2, lineJoin: .round))
            }
        }
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartLegend(.hidden)
    }
}

/// Converging TDEE estimate with a ±2σ confidence band (Trends).
struct TDEEBandChart: View {
    let series: [DatedBand]

    /// Center the line and keep the band visible without letting an early wide band pin the
    /// line to the top. Domain tracks the values, padded by a typical band width.
    private var yDomain: ClosedRange<Double> {
        let vals = series.map(\.value)
        guard let lo = vals.min(), let hi = vals.max() else { return 0...1 }
        let typicalBand = (series.map(\.se).sorted().last ?? 60) * 1.6
        let pad = max(120, typicalBand)
        return (lo - pad)...(hi + pad)
    }

    var body: some View {
        Chart {
            ForEach(series, id: \.date) { p in
                AreaMark(x: .value("d", p.date),
                         yStart: .value("lo", p.value - 2 * p.se),
                         yEnd: .value("hi", p.value + 2 * p.se))
                    .foregroundStyle(Theme.green.opacity(0.12))
            }
            ForEach(series, id: \.date) { p in
                LineMark(x: .value("d", p.date), y: .value("kcal", p.value))
                    .foregroundStyle(Color(hex: 0xCFEEDE))
                    .lineStyle(StrokeStyle(lineWidth: 2, lineJoin: .round))
            }
        }
        .chartYScale(domain: yDomain)
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartLegend(.hidden)
    }
}

/// Logging-bias percentage over time (Trends).
struct BiasChart: View {
    let series: [DatedValue]
    var body: some View {
        Chart(series, id: \.date) { p in
            LineMark(x: .value("d", p.date), y: .value("%", p.value))
                .foregroundStyle(Theme.amber)
                .lineStyle(StrokeStyle(lineWidth: 2, lineJoin: .round))
        }
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartLegend(.hidden)
    }
}
