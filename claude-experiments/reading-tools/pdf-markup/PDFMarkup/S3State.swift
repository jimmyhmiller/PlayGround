import Foundation

struct S3State: Codable {
    let uploaded: [String: UploadedFile]

    struct UploadedFile: Codable {
        let originalPath: String
        let hash: String
        let uploadedAt: String
    }
}

class S3StateManager {
    static let shared = S3StateManager()
    private var state: S3State?

    private init() {}

    func loadState() {
        guard AWSCredentials.isConfigured() else {
            print("AWS credentials not configured")
            return
        }

        Task {
            do {
                if let data = try await DrawingSyncManager.shared.signedDownloadData(key: "pdf-sync-state.json") {
                    state = try JSONDecoder().decode(S3State.self, from: data)
                }
            } catch {
                print("Failed to load S3 state: \(error)")
            }
        }
    }

    func s3Key(for hash: String) -> String? {
        if state == nil {
            loadState()
        }

        guard let state = state else {
            return nil
        }

        // Find the S3 key that contains this hash
        return state.uploaded.keys.first { key in
            key.contains(hash)
        }
    }

    func s3URL(for hash: String) -> URL? {
        guard AWSCredentials.isConfigured() else {
            print("AWS credentials not configured")
            return nil
        }

        guard let key = s3Key(for: hash) else {
            print("No S3 key found for hash: \(hash)")
            return nil
        }

        // S3 URL format: https://bucket.s3.region.amazonaws.com/key
        let urlString = "https://\(AWSCredentials.bucket).s3.\(AWSCredentials.region).amazonaws.com/\(key)"
        return URL(string: urlString)
    }
}
