//
//  DrawingSyncManager.swift
//  PDFMarkup
//
//  S3 sync using URLSession - No AWS SDK required!
//

import Foundation
import Combine
import CryptoKit

class DrawingSyncManager: ObservableObject {
    static let shared = DrawingSyncManager()

    @Published var syncState: SyncState = .idle
    @Published var lastSyncDate: Date?

    private let syncQueue = DispatchQueue(label: "com.pdfmarkup.sync", qos: .background)
    private var syncTimer: Timer?
    private var pendingUploads: Set<String> = []

    enum SyncState {
        case idle
        case syncing
        case synced
        case error(String)

        var description: String {
            switch self {
            case .idle: return "Ready"
            case .syncing: return "Syncing..."
            case .synced: return "Synced"
            case .error(let msg): return "Error: \(msg)"
            }
        }
    }

    private init() {
        // Check if credentials are configured
        if !AWSCredentials.isConfigured() {
            print("⚠️ AWS credentials not configured!")
            print("📝 Edit AWSCredentials.swift and paste your AWS access keys")
        }

        // Start periodic sync every 30 seconds
        startPeriodicSync()
    }

    // MARK: - Public API

    func uploadDrawings(pdfHash: String) async throws {
        guard AWSCredentials.isConfigured() else {
            throw SyncError.credentialsNotConfigured
        }

        guard let drawings = DrawingManager.shared.loadDrawings(pdfHash: pdfHash) else {
            print("No drawings found for PDF: \(pdfHash)")
            return
        }

        let key = "drawings/\(pdfHash).drawings.json"

        do {
            // Encode drawings to JSON
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(drawings)

            // Create signed PUT request
            let url = URL(string: "https://\(AWSCredentials.bucket).s3.\(AWSCredentials.region).amazonaws.com/\(key)")!
            var request = URLRequest(url: url)
            request.httpMethod = "PUT"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            // Note: Don't set httpBody when using upload(for:from:) - it's provided as the second parameter

            // Sign the request
            try signRequest(&request, body: data)

            // Upload to S3
            print("📤 Uploading to: \(url)")
            let (responseData, response) = try await URLSession.shared.upload(for: request, from: data)

            guard let httpResponse = response as? HTTPURLResponse else {
                print("❌ Invalid response type")
                throw SyncError.uploadFailed
            }

            print("📥 Response status: \(httpResponse.statusCode)")

            if !(200...299).contains(httpResponse.statusCode) {
                if let responseString = String(data: responseData, encoding: .utf8) {
                    print("❌ S3 Error Response: \(responseString)")
                }
                throw SyncError.uploadFailed
            }

            print("✅ Successfully uploaded drawings for \(pdfHash) to S3")
            await MainActor.run {
                self.lastSyncDate = Date()
                self.syncState = .synced
            }
        } catch {
            print("❌ Failed to upload drawings: \(error)")
            await MainActor.run {
                self.syncState = .error(error.localizedDescription)
            }
            throw error
        }
    }

    func downloadDrawings(pdfHash: String) async throws -> PDFDrawings? {
        guard AWSCredentials.isConfigured() else {
            throw SyncError.credentialsNotConfigured
        }

        let key = "drawings/\(pdfHash).drawings.json"
        let url = URL(string: "https://\(AWSCredentials.bucket).s3.\(AWSCredentials.region).amazonaws.com/\(key)")!

        var request = URLRequest(url: url)
        request.httpMethod = "GET"

        // Sign the request
        try signRequest(&request, body: nil)

        do {
            let (data, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw SyncError.downloadFailed
            }

            // 404 is okay - means no drawings exist yet
            if httpResponse.statusCode == 404 {
                print("ℹ️ No drawings found on S3 for \(pdfHash) (new PDF)")
                return nil
            }

            guard (200...299).contains(httpResponse.statusCode) else {
                throw SyncError.downloadFailed
            }

            let decoder = JSONDecoder()
            let drawings = try decoder.decode(PDFDrawings.self, from: data)

            print("✅ Successfully downloaded drawings for \(pdfHash) from S3")
            return drawings
        } catch let error as DecodingError {
            print("⚠️  Failed to decode drawings: \(error)")
            return nil
        } catch {
            if (error as NSError).domain == NSURLErrorDomain && (error as NSError).code == NSURLErrorBadServerResponse {
                // Likely a 404 or similar
                return nil
            }
            throw error
        }
    }

    func sync(pdfHash: String) async throws {
        await MainActor.run { self.syncState = .syncing }

        do {
            // Download remote version
            let remoteDrawings = try await downloadDrawings(pdfHash: pdfHash)
            let localDrawings = DrawingManager.shared.loadDrawings(pdfHash: pdfHash)

            // Resolve conflicts
            let mergedDrawings = resolveConflict(
                local: localDrawings,
                remote: remoteDrawings,
                pdfHash: pdfHash
            )

            // Save merged version locally
            if let merged = mergedDrawings {
                saveMergedDrawings(merged)
            }

            // Upload to S3
            if localDrawings != nil || mergedDrawings != nil {
                try await uploadDrawings(pdfHash: pdfHash)
            }

            await MainActor.run {
                self.syncState = .synced
                self.lastSyncDate = Date()
            }
        } catch {
            await MainActor.run {
                self.syncState = .error(error.localizedDescription)
            }
            throw error
        }
    }

    func syncAll() async {
        await MainActor.run { self.syncState = .syncing }

        let drawingFiles = DrawingManager.shared.getAllDrawingFiles()
        print("🔄 Syncing \(drawingFiles.count) PDFs with drawings")

        for file in drawingFiles {
            let hash = file.replacingOccurrences(of: ".drawings.json", with: "")

            do {
                try await sync(pdfHash: hash)
            } catch {
                print("❌ Failed to sync \(hash): \(error)")
            }
        }

        await MainActor.run {
            self.syncState = .synced
            self.lastSyncDate = Date()
        }
    }

    func markForSync(pdfHash: String) {
        pendingUploads.insert(pdfHash)
    }

    // MARK: - AWS Signature V4 Signing

    private func signRequest(_ request: inout URLRequest, body: Data?) throws {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        dateFormatter.timeZone = TimeZone(identifier: "UTC")
        let timestamp = dateFormatter.string(from: Date())

        let dateStamp = String(timestamp.prefix(8))

        // Calculate payload hash
        let payloadHash: String
        if let body = body {
            payloadHash = SHA256.hash(data: body).hexString
        } else {
            payloadHash = SHA256.hash(data: Data()).hexString
        }

        // Add required headers
        request.setValue(timestamp, forHTTPHeaderField: "X-Amz-Date")
        request.setValue(request.url!.host!, forHTTPHeaderField: "Host")
        request.setValue(payloadHash, forHTTPHeaderField: "X-Amz-Content-Sha256")

        // Create canonical request
        let httpMethod = request.httpMethod!
        let canonicalURI = request.url!.path.isEmpty ? "/" : request.url!.path
        let canonicalQueryString = ""
        let canonicalHeaders = "host:\(request.url!.host!)\nx-amz-content-sha256:\(payloadHash)\nx-amz-date:\(timestamp)\n"
        let signedHeaders = "host;x-amz-content-sha256;x-amz-date"

        let canonicalRequest = """
        \(httpMethod)
        \(canonicalURI)
        \(canonicalQueryString)
        \(canonicalHeaders)
        \(signedHeaders)
        \(payloadHash)
        """

        let canonicalRequestHash = SHA256.hash(data: Data(canonicalRequest.utf8)).hexString

        // Create string to sign
        let credentialScope = "\(dateStamp)/\(AWSCredentials.region)/s3/aws4_request"
        let stringToSign = """
        AWS4-HMAC-SHA256
        \(timestamp)
        \(credentialScope)
        \(canonicalRequestHash)
        """

        // Calculate signing key
        let kDate = hmac(key: Data("AWS4\(AWSCredentials.secretAccessKey)".utf8), data: Data(dateStamp.utf8))
        let kRegion = hmac(key: kDate, data: Data(AWSCredentials.region.utf8))
        let kService = hmac(key: kRegion, data: Data("s3".utf8))
        let kSigning = hmac(key: kService, data: Data("aws4_request".utf8))

        // Calculate signature
        let signature = hmac(key: kSigning, data: Data(stringToSign.utf8)).hexString

        // Create authorization header
        let authorization = "AWS4-HMAC-SHA256 Credential=\(AWSCredentials.accessKeyId)/\(credentialScope), SignedHeaders=\(signedHeaders), Signature=\(signature)"

        request.setValue(authorization, forHTTPHeaderField: "Authorization")
    }

    private func hmac(key: Data, data: Data) -> Data {
        var hmac = HMAC<SHA256>(key: SymmetricKey(data: key))
        hmac.update(data: data)
        return Data(hmac.finalize())
    }

    // MARK: - Conflict Resolution

    private func resolveConflict(
        local: PDFDrawings?,
        remote: PDFDrawings?,
        pdfHash: String
    ) -> PDFDrawings? {
        guard let local = local, let remote = remote else {
            return local ?? remote
        }

        // Last-write-wins
        if local.lastModified > remote.lastModified {
            print("✓ Using local version (newer)")
            return local
        } else if remote.lastModified > local.lastModified {
            print("✓ Using remote version (newer)")
            return remote
        } else {
            print("⚠️ Same timestamp - merging pages")
            var merged = local

            for (pageIndex, pageData) in remote.pages {
                if merged.pages[pageIndex] == nil {
                    merged.pages[pageIndex] = pageData
                }
            }

            merged.lastModified = Date()
            return merged
        }
    }

    private func saveMergedDrawings(_ drawings: PDFDrawings) {
        for (pageIndex, _) in drawings.pages {
            if let drawing = drawings.getDrawing(forPage: pageIndex) {
                DrawingManager.shared.saveDrawing(drawing, pdfHash: drawings.pdfHash, page: pageIndex)
            }
        }
    }

    // MARK: - Background Sync

    private func startPeriodicSync() {
        syncTimer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }

            Task {
                await self.syncPending()
            }
        }
    }

    private func syncPending() async {
        guard !pendingUploads.isEmpty else { return }
        guard AWSCredentials.isConfigured() else { return }

        let toSync = Array(pendingUploads)
        pendingUploads.removeAll()

        for pdfHash in toSync {
            do {
                try await uploadDrawings(pdfHash: pdfHash)
            } catch {
                print("❌ Failed to sync \(pdfHash): \(error)")
                pendingUploads.insert(pdfHash)
            }
        }
    }

    deinit {
        syncTimer?.invalidate()
    }
}

// MARK: - Errors

enum SyncError: LocalizedError {
    case credentialsNotConfigured
    case uploadFailed
    case downloadFailed

    var errorDescription: String? {
        switch self {
        case .credentialsNotConfigured:
            return "AWS credentials not configured. Edit AWSCredentials.swift"
        case .uploadFailed:
            return "Failed to upload to S3"
        case .downloadFailed:
            return "Failed to download from S3"
        }
    }
}

// MARK: - Helpers

extension Digest {
    var hexString: String {
        map { String(format: "%02x", $0) }.joined()
    }
}

extension Data {
    var hexString: String {
        map { String(format: "%02x", $0) }.joined()
    }
}
