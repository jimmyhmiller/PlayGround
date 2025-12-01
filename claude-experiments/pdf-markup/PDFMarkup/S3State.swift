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
        let statePath = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/reading-tools/pdf-sync/.pdf-sync-state.json"

        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: statePath))
            state = try JSONDecoder().decode(S3State.self, from: data)
        } catch {
            print("Failed to load S3 state: \(error)")
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

    func s3URL(for hash: String, bucket: String = "jimmyhmiller-bucket") -> URL? {
        guard let key = s3Key(for: hash) else {
            print("No S3 key found for hash: \(hash)")
            return nil
        }

        // S3 URL format: https://bucket.s3.amazonaws.com/key
        let urlString = "https://\(bucket).s3.amazonaws.com/\(key)"
        return URL(string: urlString)
    }
}
