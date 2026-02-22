import Foundation
import CloudKit

@MainActor
class CloudKitManager {
    private let container: CKContainer
    private let privateDB: CKDatabase
    private let zoneID: CKRecordZone.ID

    private let goalRecordType = "Goal"
    private let entryRecordType = "Entry"
    private let zoneName = "EaseZone"

    private let changeTokenKey = "cloudKitServerChangeToken"
    private let zoneCreatedKey = "cloudKitZoneCreated"
    private let subscriptionCreatedKey = "cloudKitSubscriptionCreated"

    var isAvailable: Bool {
        FileManager.default.ubiquityIdentityToken != nil
    }

    init() {
        container = CKContainer(identifier: "iCloud.com.jimmyhmiller.Ease")
        privateDB = container.privateCloudDatabase
        zoneID = CKRecordZone.ID(zoneName: zoneName, ownerName: CKCurrentUserDefaultName)
    }

    // MARK: - Zone & Subscription Setup

    func setupZoneAndSubscription() async {
        guard isAvailable else { return }
        await createZoneIfNeeded()
        await createSubscriptionIfNeeded()
    }

    private func createZoneIfNeeded() async {
        guard !UserDefaults.standard.bool(forKey: zoneCreatedKey) else { return }
        let zone = CKRecordZone(zoneID: zoneID)
        do {
            _ = try await privateDB.save(zone)
            UserDefaults.standard.set(true, forKey: zoneCreatedKey)
        } catch {
            print("CloudKit: Failed to create zone: \(error)")
        }
    }

    private func createSubscriptionIfNeeded() async {
        guard !UserDefaults.standard.bool(forKey: subscriptionCreatedKey) else { return }
        let subscription = CKDatabaseSubscription(subscriptionID: "ease-all-changes")
        let notificationInfo = CKSubscription.NotificationInfo()
        notificationInfo.shouldSendContentAvailable = true
        subscription.notificationInfo = notificationInfo
        do {
            _ = try await privateDB.save(subscription)
            UserDefaults.standard.set(true, forKey: subscriptionCreatedKey)
        } catch {
            print("CloudKit: Failed to create subscription: \(error)")
        }
    }

    // MARK: - Push Changes

    func pushChanges(goals: [Goal], entries: [Entry], deletedGoalIds: Set<UUID>, deletedEntryIds: Set<UUID>) async {
        guard isAvailable else { return }

        var recordsToSave: [CKRecord] = []
        let recordIDsToDelete: [CKRecord.ID] = []

        // Convert active goals to records
        for goal in goals {
            let record = goalToCKRecord(goal)
            recordsToSave.append(record)
        }

        // Convert active entries to records
        for entry in entries {
            let record = entryToCKRecord(entry)
            recordsToSave.append(record)
        }

        // Mark deleted goals
        for goalId in deletedGoalIds {
            let recordID = CKRecord.ID(recordName: goalId.uuidString, zoneID: zoneID)
            // Push a tombstone record rather than deleting, so other devices see the deletion
            let record = CKRecord(recordType: goalRecordType, recordID: recordID)
            record["isDeleted"] = true as CKRecordValue
            record["name"] = "" as CKRecordValue
            record["colorHex"] = "" as CKRecordValue
            record["modifiedAt"] = Date() as CKRecordValue
            recordsToSave.append(record)
        }

        // Mark deleted entries
        for entryId in deletedEntryIds {
            let recordID = CKRecord.ID(recordName: entryId.uuidString, zoneID: zoneID)
            let record = CKRecord(recordType: entryRecordType, recordID: recordID)
            record["isDeleted"] = true as CKRecordValue
            record["goalId"] = "" as CKRecordValue
            record["amount"] = 0.0 as CKRecordValue
            record["timestamp"] = Date() as CKRecordValue
            recordsToSave.append(record)
        }

        guard !recordsToSave.isEmpty || !recordIDsToDelete.isEmpty else { return }

        // Push in batches of 400 (CloudKit limit is 400 per operation)
        let batchSize = 400
        for startIndex in stride(from: 0, to: recordsToSave.count, by: batchSize) {
            let endIndex = min(startIndex + batchSize, recordsToSave.count)
            let batch = Array(recordsToSave[startIndex..<endIndex])

            let operation = CKModifyRecordsOperation(recordsToSave: batch, recordIDsToDelete: startIndex == 0 ? recordIDsToDelete : nil)
            operation.savePolicy = .changedKeys
            operation.isAtomic = false
            operation.qualityOfService = .userInitiated

            do {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                    operation.modifyRecordsResultBlock = { result in
                        switch result {
                        case .success:
                            continuation.resume()
                        case .failure(let error):
                            continuation.resume(throwing: error)
                        }
                    }
                    self.privateDB.add(operation)
                }
            } catch {
                print("CloudKit: Push failed: \(error)")
            }
        }
    }

    // MARK: - Fetch Changes

    struct FetchResult {
        var goals: [Goal] = []
        var entries: [Entry] = []
        var deletedGoalIds: Set<UUID> = []
        var deletedEntryIds: Set<UUID> = []
    }

    func fetchChanges() async -> FetchResult {
        guard isAvailable else { return FetchResult() }

        var result = FetchResult()
        let token = loadChangeToken()

        let config = CKFetchRecordZoneChangesOperation.ZoneConfiguration()
        config.previousServerChangeToken = token

        let operation = CKFetchRecordZoneChangesOperation(recordZoneIDs: [zoneID], configurationsByRecordZoneID: [zoneID: config])

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            operation.recordWasChangedBlock = { recordID, recordResult in
                switch recordResult {
                case .success(let record):
                    if record.recordType == self.goalRecordType {
                        let goal = self.ckRecordToGoal(record)
                        if goal.isDeleted {
                            result.deletedGoalIds.insert(goal.id)
                        } else {
                            result.goals.append(goal)
                        }
                    } else if record.recordType == self.entryRecordType {
                        let entry = self.ckRecordToEntry(record)
                        if entry.isDeleted {
                            result.deletedEntryIds.insert(entry.id)
                        } else {
                            result.entries.append(entry)
                        }
                    }
                case .failure(let error):
                    print("CloudKit: Record fetch error: \(error)")
                }
            }

            operation.recordZoneChangeTokensUpdatedBlock = { zoneID, newToken, _ in
                if let newToken = newToken {
                    self.saveChangeToken(newToken)
                }
            }

            operation.recordZoneFetchResultBlock = { zoneID, result in
                switch result {
                case .success(let (newToken, _, _)):
                    self.saveChangeToken(newToken)
                case .failure(let error):
                    let ckError = error as? CKError
                    if ckError?.code == .changeTokenExpired {
                        // Reset token and retry on next poll
                        self.clearChangeToken()
                    }
                    print("CloudKit: Zone fetch error: \(error)")
                }
            }

            operation.fetchRecordZoneChangesResultBlock = { _ in
                continuation.resume()
            }

            operation.qualityOfService = .userInitiated
            self.privateDB.add(operation)
        }

        return result
    }

    // MARK: - Merge

    func merge(local: StoredData, remote: FetchResult) -> StoredData {
        var mergedGoals = local.goals
        var mergedEntries = local.entries
        let deletedGoalIds = local.deletedGoalIds.union(remote.deletedGoalIds)
        let deletedEntryIds = local.deletedEntryIds.union(remote.deletedEntryIds)

        // Merge remote goals
        for remoteGoal in remote.goals {
            if deletedGoalIds.contains(remoteGoal.id) {
                continue // Skip deleted goals
            }
            if let localIndex = mergedGoals.firstIndex(where: { $0.id == remoteGoal.id }) {
                // Same UUID — keep the one with later modifiedAt
                if remoteGoal.modifiedAt > mergedGoals[localIndex].modifiedAt {
                    mergedGoals[localIndex] = remoteGoal
                }
            } else {
                // New goal from remote — append
                mergedGoals.append(remoteGoal)
            }
        }

        // Remove locally deleted goals
        mergedGoals.removeAll { deletedGoalIds.contains($0.id) }

        // Merge remote entries
        let localEntryIds = Set(mergedEntries.map { $0.id })
        for remoteEntry in remote.entries {
            if deletedEntryIds.contains(remoteEntry.id) {
                continue
            }
            if !localEntryIds.contains(remoteEntry.id) {
                mergedEntries.append(remoteEntry)
            }
        }

        // Remove locally deleted entries
        mergedEntries.removeAll { deletedEntryIds.contains($0.id) }

        // Also remove entries for deleted goals
        mergedEntries.removeAll { deletedGoalIds.contains($0.goalId) }

        return StoredData(
            goals: mergedGoals,
            entries: mergedEntries,
            deletedGoalIds: deletedGoalIds,
            deletedEntryIds: deletedEntryIds
        )
    }

    // MARK: - CKRecord Conversions

    private func goalToCKRecord(_ goal: Goal) -> CKRecord {
        let recordID = CKRecord.ID(recordName: goal.id.uuidString, zoneID: zoneID)
        let record = CKRecord(recordType: goalRecordType, recordID: recordID)
        record["name"] = goal.name as CKRecordValue
        record["colorHex"] = goal.colorHex as CKRecordValue
        record["modifiedAt"] = goal.modifiedAt as CKRecordValue
        record["isDeleted"] = goal.isDeleted as CKRecordValue
        return record
    }

    private func ckRecordToGoal(_ record: CKRecord) -> Goal {
        let id = UUID(uuidString: record.recordID.recordName) ?? UUID()
        let name = record["name"] as? String ?? ""
        let colorHex = record["colorHex"] as? String ?? "#888888"
        let modifiedAt = record["modifiedAt"] as? Date ?? Date()
        let isDeleted = record["isDeleted"] as? Bool ?? false
        return Goal(id: id, name: name, colorHex: colorHex, modifiedAt: modifiedAt, isDeleted: isDeleted)
    }

    private func entryToCKRecord(_ entry: Entry) -> CKRecord {
        let recordID = CKRecord.ID(recordName: entry.id.uuidString, zoneID: zoneID)
        let record = CKRecord(recordType: entryRecordType, recordID: recordID)
        record["goalId"] = entry.goalId.uuidString as CKRecordValue
        record["amount"] = entry.amount as CKRecordValue
        record["timestamp"] = entry.timestamp as CKRecordValue
        record["isDeleted"] = entry.isDeleted as CKRecordValue
        return record
    }

    private func ckRecordToEntry(_ record: CKRecord) -> Entry {
        let id = UUID(uuidString: record.recordID.recordName) ?? UUID()
        let goalIdStr = record["goalId"] as? String ?? ""
        let goalId = UUID(uuidString: goalIdStr) ?? UUID()
        let amount = record["amount"] as? Double ?? 0
        let timestamp = record["timestamp"] as? Date ?? Date()
        let isDeleted = record["isDeleted"] as? Bool ?? false
        return Entry(id: id, goalId: goalId, amount: amount, timestamp: timestamp, isDeleted: isDeleted)
    }

    // MARK: - Change Token Persistence

    private func loadChangeToken() -> CKServerChangeToken? {
        guard let data = UserDefaults.standard.data(forKey: changeTokenKey) else { return nil }
        return try? NSKeyedUnarchiver.unarchivedObject(ofClass: CKServerChangeToken.self, from: data)
    }

    private func saveChangeToken(_ token: CKServerChangeToken) {
        let data = try? NSKeyedArchiver.archivedData(withRootObject: token, requiringSecureCoding: true)
        UserDefaults.standard.set(data, forKey: changeTokenKey)
    }

    private func clearChangeToken() {
        UserDefaults.standard.removeObject(forKey: changeTokenKey)
    }
}
