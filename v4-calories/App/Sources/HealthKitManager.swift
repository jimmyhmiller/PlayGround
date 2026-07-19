import Foundation
import CalorieModel
#if canImport(HealthKit)
import HealthKit
#endif

/// Thin wrapper around HealthKit. Everything degrades to a no-op when Health data is
/// unavailable (e.g. on a simulator without seeded data), so the app never depends on it.
final class HealthKitManager {
    enum EnergyKind { case active, basal }
    /// Whether the system would still show a permission sheet for our types.
    enum AuthStatus { case unavailable, unknown, shouldRequest, alreadyDetermined }

    #if canImport(HealthKit)
    private let store = HKHealthStore()
    private let kcal = HKUnit.kilocalorie()
    private let lb = HKUnit.pound()

    private var activeType: HKQuantityType { HKQuantityType(.activeEnergyBurned) }
    private var basalType: HKQuantityType { HKQuantityType(.basalEnergyBurned) }
    private var massType: HKQuantityType { HKQuantityType(.bodyMass) }
    private var dietType: HKQuantityType { HKQuantityType(.dietaryEnergyConsumed) }

    var isAvailable: Bool { HKHealthStore.isHealthDataAvailable() }

    func requestAuthorization() async -> Bool {
        guard isAvailable else { return false }
        let read: Set<HKObjectType> = [activeType, basalType, massType]
        let write: Set<HKSampleType> = [massType, dietType]
        return await withCheckedContinuation { cont in
            store.requestAuthorization(toShare: write, read: read) { ok, _ in cont.resume(returning: ok) }
        }
    }

    /// HealthKit hides whether *read* access was granted, but it will tell us whether a
    /// permission sheet would still appear. `.alreadyDetermined` + no data ≈ read denied.
    func requestStatus() async -> AuthStatus {
        guard isAvailable else { return .unavailable }
        let read: Set<HKObjectType> = [activeType, basalType, massType]
        let write: Set<HKSampleType> = [massType, dietType]
        return await withCheckedContinuation { cont in
            store.getRequestStatusForAuthorization(toShare: write, read: read) { status, _ in
                switch status {
                case .shouldRequest: cont.resume(returning: .shouldRequest)
                case .unnecessary:   cont.resume(returning: .alreadyDetermined)
                default:             cont.resume(returning: .unknown)
                }
            }
        }
    }

    func dailyEnergy(_ kind: EnergyKind, from start: Date, to end: Date) async -> [Date: Double] {
        guard isAvailable else { return [:] }
        let type = kind == .active ? activeType : basalType
        let cal = Calendar.current
        let anchor = cal.startOfDay(for: start)
        let endExclusive = cal.date(byAdding: .day, value: 1, to: cal.startOfDay(for: end)) ?? end
        let predicate = HKQuery.predicateForSamples(withStart: anchor, end: endExclusive)

        return await withCheckedContinuation { cont in
            let q = HKStatisticsCollectionQuery(
                quantityType: type, quantitySamplePredicate: predicate,
                options: .cumulativeSum, anchorDate: anchor,
                intervalComponents: DateComponents(day: 1))
            q.initialResultsHandler = { [kcal] _, results, _ in
                var out: [Date: Double] = [:]
                results?.enumerateStatistics(from: anchor, to: endExclusive) { stat, _ in
                    if let sum = stat.sumQuantity() {
                        out[cal.startOfDay(for: stat.startDate)] = sum.doubleValue(for: kcal)
                    }
                }
                cont.resume(returning: out)
            }
            store.execute(q)
        }
    }

    func bodyMass(from start: Date, to end: Date) async -> [WeighIn] {
        guard isAvailable else { return [] }
        let cal = Calendar.current
        let endExclusive = cal.date(byAdding: .day, value: 1, to: cal.startOfDay(for: end)) ?? end
        let predicate = HKQuery.predicateForSamples(withStart: cal.startOfDay(for: start), end: endExclusive)
        let sort = [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)]
        return await withCheckedContinuation { cont in
            let q = HKSampleQuery(sampleType: massType, predicate: predicate,
                                  limit: HKObjectQueryNoLimit, sortDescriptors: sort) { [lb] _, samples, _ in
                let weighIns: [WeighIn] = (samples as? [HKQuantitySample] ?? []).map {
                    WeighIn(date: $0.startDate, weightLb: $0.quantity.doubleValue(for: lb), fromHealthKit: true)
                }
                cont.resume(returning: weighIns)
            }
            store.execute(q)
        }
    }

    func saveBodyMass(lb value: Double, date: Date) async {
        guard isAvailable else { return }
        let sample = HKQuantitySample(type: massType,
                                      quantity: HKQuantity(unit: lb, doubleValue: value),
                                      start: date, end: date)
        try? await store.save(sample)
    }

    func saveDietaryEnergy(kcal value: Double, date: Date) async {
        guard isAvailable else { return }
        let sample = HKQuantitySample(type: dietType,
                                      quantity: HKQuantity(unit: kcal, doubleValue: value),
                                      start: date, end: date)
        try? await store.save(sample)
    }
    #else
    var isAvailable: Bool { false }
    func requestAuthorization() async -> Bool { false }
    func requestStatus() async -> AuthStatus { .unavailable }
    func dailyEnergy(_ kind: EnergyKind, from start: Date, to end: Date) async -> [Date: Double] { [:] }
    func bodyMass(from start: Date, to end: Date) async -> [WeighIn] { [] }
    func saveBodyMass(lb value: Double, date: Date) async {}
    func saveDietaryEnergy(kcal value: Double, date: Date) async {}
    #endif
}
