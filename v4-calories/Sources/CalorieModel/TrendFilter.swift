import Foundation

/// 2×2 linear algebra for the local-linear-trend Kalman filter. Small and explicit so the
/// filter has no external dependencies and runs identically on macOS and iOS.
struct Mat2: Equatable {
    var a, b, c, d: Double   // [[a, b], [c, d]]
    static let identity = Mat2(a: 1, b: 0, c: 0, d: 1)
    static func + (l: Mat2, r: Mat2) -> Mat2 { Mat2(a: l.a + r.a, b: l.b + r.b, c: l.c + r.c, d: l.d + r.d) }
    static func - (l: Mat2, r: Mat2) -> Mat2 { Mat2(a: l.a - r.a, b: l.b - r.b, c: l.c - r.c, d: l.d - r.d) }
    static func * (l: Mat2, r: Mat2) -> Mat2 {
        Mat2(a: l.a * r.a + l.b * r.c, b: l.a * r.b + l.b * r.d,
             c: l.c * r.a + l.d * r.c, d: l.c * r.b + l.d * r.d)
    }
    var transpose: Mat2 { Mat2(a: a, b: c, c: b, d: d) }
    var inverse: Mat2 {
        let det = a * d - b * c
        let s = abs(det) < 1e-12 ? 0 : 1 / det
        return Mat2(a: d * s, b: -b * s, c: -c * s, d: a * s)
    }
}
struct Vec2: Equatable {
    var x, y: Double
    static func + (l: Vec2, r: Vec2) -> Vec2 { Vec2(x: l.x + r.x, y: l.y + r.y) }
    static func - (l: Vec2, r: Vec2) -> Vec2 { Vec2(x: l.x - r.x, y: l.y - r.y) }
}
func * (m: Mat2, v: Vec2) -> Vec2 { Vec2(x: m.a * v.x + m.b * v.y, y: m.c * v.x + m.d * v.y) }

/// A local-linear-trend (level + slope) Kalman model for body weight, run as an
/// RTS smoother.
///
/// State = [trend weight, daily slope]. True weight moves along its current slope; the slope
/// itself drifts slowly (metabolic adaptation, life changes). Each scale reading is the level
/// plus `water` noise. Because slope is modelled explicitly, a steady diet is tracked with **no
/// lag**, endpoint levels are estimated using the whole series, and we get the loss *rate* for
/// free. The backward pass denoises the start anchor so a start→end cumulative deficit is honest.
///
/// At the most recent day the smoother equals the causal filter, so "current weight" never peeks
/// ahead. During weigh-in gaps the level extrapolates along the (small, stable) slope — which is
/// exactly "the scale will fill the gap" — with variance that grows until the next reading.
public struct TrendFilter: Sendable {
    /// Water-noise variance of one scale reading, lb².
    public var measurementVar: Double
    /// Process noise on the level (unmodelled persistent shifts), lb²/day.
    public var levelVar: Double
    /// Process noise on the slope (how fast the true rate can change), (lb/day)²/day.
    public var slopeVar: Double

    public struct Point: Sendable, Equatable {
        public var date: Date
        public var trend: Double?        // smoothed level (lb)
        public var variance: Double?     // smoothed level variance (lb²)
        public var slopePerDay: Double?  // smoothed slope (lb/day)
        public var observed: Double?
    }

    /// When true, the water-noise level `R` is estimated from each user's own reading
    /// innovations (a second pass), so the uncertainty band self-calibrates instead of
    /// trusting a global guess. This is what keeps the reported ±band honest.
    public var adaptiveMeasurement: Bool

    public init(measurementVar: Double = 3.0, levelVar: Double = 0.004, slopeVar: Double = 8e-6,
                adaptiveMeasurement: Bool = true) {
        self.measurementVar = measurementVar
        self.levelVar = levelVar
        self.slopeVar = slopeVar
        self.adaptiveMeasurement = adaptiveMeasurement
    }

    public func run(_ records: [DailyRecord]) -> [Point] {
        guard adaptiveMeasurement, let r = estimateWaterVar(records) else {
            return smooth(records, measurementVar: measurementVar).points
        }
        return smooth(records, measurementVar: r).points
    }

    /// Estimate the **marginal** water-noise variance `R` for this user.
    ///
    /// The hard part: a slow multi-day hydration cycle is aliased with real weight change, so
    /// neither filter innovations nor day-to-day differences (which only see the *fast*
    /// component) capture it — and underestimating it makes the uncertainty band lie.
    ///
    /// We get the full marginal variance from residuals against a **stiff** trend (high R, tiny
    /// slope noise → a near-linear trend that does *not* bend to follow a water sinusoid), so
    /// the slow swings show up in the residuals where we can measure them. We take the max with
    /// the day-to-day-difference estimate (a floor for the fast component) and clamp.
    public func estimateWaterVar(_ records: [DailyRecord]) -> Double? {
        let robust: ([Double]) -> Double? = { xs in
            guard xs.count >= 5 else { return nil }
            let s = xs.sorted()
            let med = s[s.count / 2]
            let ad = xs.map { abs($0 - med) }.sorted()
            let sigma = 1.4826 * ad[ad.count / 2]
            return sigma * sigma
        }

        // Marginal: residuals vs a stiff trend.
        let stiff = smooth(records, measurementVar: 8).points
        var resid: [Double] = []
        for i in records.indices {
            if let z = records[i].weightLb, z.isFinite, let t = stiff[i].trend { resid.append(z - t) }
        }
        let marginal = robust(resid)

        // Fast floor: ½·Var(consecutive-day difference) ≈ R.
        var diffs: [Double] = []
        for i in 1..<max(1, records.count) {
            if let a = records[i - 1].weightLb, let b = records[i].weightLb, a.isFinite, b.isFinite {
                diffs.append(b - a)
            }
        }
        let fastR = robust(diffs).map { $0 / 2.0 }

        guard let m = marginal ?? fastR else { return nil }
        return min(36, max(0.6, max(m, fastR ?? 0)))
    }

    /// One forward+backward RTS pass at a given measurement-noise level `R`.
    private func smooth(_ records: [DailyRecord], measurementVar R: Double) -> (points: [Point], Void) {
        let n = records.count
        var out = records.map {
            Point(date: $0.date, trend: nil, variance: nil, slopePerDay: nil, observed: $0.weightLb)
        }
        guard let first = records.firstIndex(where: { ($0.weightLb ?? .nan).isFinite }) else {
            return (out, ())
        }

        let F = Mat2(a: 1, b: 1, c: 0, d: 1)
        let Q = Mat2(a: levelVar, b: 0, c: 0, d: slopeVar)

        var mPred = [Vec2](repeating: Vec2(x: 0, y: 0), count: n)
        var pPred = [Mat2](repeating: .identity, count: n)
        var mFilt = [Vec2](repeating: Vec2(x: 0, y: 0), count: n)
        var pFilt = [Mat2](repeating: .identity, count: n)

        // Seed: level = first reading, slope = 0, with diffuse slope prior.
        var m = Vec2(x: records[first].weightLb!, y: 0)
        var P = Mat2(a: R, b: 0, c: 0, d: 0.02)
        mPred[first] = m; pPred[first] = P; mFilt[first] = m; pFilt[first] = P

        if first + 1 < n {
            for i in (first + 1)..<n {
                // Predict.
                m = F * m
                P = F * P * F.transpose + Q
                mPred[i] = m; pPred[i] = P
                // Update (H = [1, 0]).
                if let z = records[i].weightLb, z.isFinite {
                    let s = P.a + R
                    let k = Vec2(x: P.a / s, y: P.c / s)               // Kalman gain (2×1)
                    let innov = z - m.x
                    m = Vec2(x: m.x + k.x * innov, y: m.y + k.y * innov)
                    // P = (I - K H) P,  K H = [[k.x, 0], [k.y, 0]]
                    let KH = Mat2(a: k.x, b: 0, c: k.y, d: 0)
                    P = (.identity - KH) * P
                }
                mFilt[i] = m; pFilt[i] = P
            }
        }

        // Backward RTS smoother over [first, n-1].
        var mSm = mFilt
        var pSm = pFilt
        if first < n - 1 {
            for i in stride(from: n - 2, through: first, by: -1) {
                let C = pFilt[i] * F.transpose * pPred[i + 1].inverse
                mSm[i] = mFilt[i] + C * (mSm[i + 1] - mPred[i + 1])
                pSm[i] = pFilt[i] + C * (pSm[i + 1] - pPred[i + 1]) * C.transpose
            }
        }

        for i in first..<n {
            out[i].trend = mSm[i].x
            out[i].slopePerDay = mSm[i].y
            out[i].variance = max(0, pSm[i].a)
        }
        return (out, ())
    }

    public static func lastTrendIndex(_ points: [Point]) -> Int? {
        for i in stride(from: points.count - 1, through: 0, by: -1) where points[i].trend != nil {
            return i
        }
        return nil
    }

    public static func firstTrendIndex(_ points: [Point]) -> Int? {
        points.firstIndex { $0.trend != nil }
    }
}
