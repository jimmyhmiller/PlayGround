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
            request.setValue("public-read", forHTTPHeaderField: "x-amz-acl")
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
        print("📥 downloadDrawings called for \(pdfHash)")
        guard AWSCredentials.isConfigured() else {
            print("❌ AWS credentials not configured")
            throw SyncError.credentialsNotConfigured
        }

        let key = "drawings/\(pdfHash).drawings.json"
        let url = URL(string: "https://\(AWSCredentials.bucket).s3.\(AWSCredentials.region).amazonaws.com/\(key)")!
        print("📥 Fetching from: \(url)")

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
            decoder.dateDecodingStrategy = .deferredToDate
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

    /// Downloads data from S3 using signed requests
    func signedDownloadData(key: String) async throws -> Data? {
        guard AWSCredentials.isConfigured() else {
            throw SyncError.credentialsNotConfigured
        }

        let url = URL(string: "https://\(AWSCredentials.bucket).s3.\(AWSCredentials.region).amazonaws.com/\(key)")!

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        try signRequest(&request, body: nil)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw SyncError.downloadFailed
        }

        if httpResponse.statusCode == 404 {
            return nil
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            throw SyncError.downloadFailed
        }

        return data
    }

    /// Downloads a file from S3 using signed requests
    func signedDownloadFile(key: String, to destination: URL) async throws -> Bool {
        guard AWSCredentials.isConfigured() else {
            throw SyncError.credentialsNotConfigured
        }

        let url = URL(string: "https://\(AWSCredentials.bucket).s3.\(AWSCredentials.region).amazonaws.com/\(key)")!

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        try signRequest(&request, body: nil)

        let (tempURL, response) = try await URLSession.shared.download(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw SyncError.downloadFailed
        }

        if httpResponse.statusCode == 404 {
            return false
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            throw SyncError.downloadFailed
        }

        // Move to destination
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)

        return true
    }

    func sync(pdfHash: String) async throws {
        print("🔄 Starting sync for \(pdfHash)")
        await MainActor.run { self.syncState = .syncing }

        do {
            // Download remote version
            print("📥 Downloading remote drawings...")
            let remoteDrawings = try await downloadDrawings(pdfHash: pdfHash)
            print("📥 Remote drawings: \(remoteDrawings != nil ? "found \(remoteDrawings!.pages.count) pages" : "none")")

            let localDrawings = DrawingManager.shared.loadDrawings(pdfHash: pdfHash)
            print("💾 Local drawings: \(localDrawings != nil ? "found \(localDrawings!.pages.count) pages" : "none")")

            // Resolve conflicts
            let mergedDrawings = resolveConflict(
                local: localDrawings,
                remote: remoteDrawings,
                pdfHash: pdfHash
            )
            print("🔀 Merged drawings: \(mergedDrawings != nil ? "found \(mergedDrawings!.pages.count) pages" : "none")")

            // Save merged version locally
            if let merged = mergedDrawings {
                print("💾 Saving merged drawings locally...")
                saveMergedDrawings(merged)
                print("✅ Saved merged drawings")
            }

            // Only upload if we have local changes AND we successfully checked remote
            // This prevents overwriting remote with empty local when download fails
            let shouldUpload = localDrawings != nil && (remoteDrawings == nil || localDrawings!.lastModified > remoteDrawings!.lastModified)
            if shouldUpload {
                print("📤 Uploading local changes to S3...")
                try await uploadDrawings(pdfHash: pdfHash)
            } else {
                print("⏭️ Skipping upload (no local changes or remote is newer)")
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

    // MARK: - PDF Upload

    /// Uploads a shared PDF to S3 and updates index files
    func uploadSharedPDF(data: Data, hash: String, fileName: String, pageCount: Int) async throws {
        guard AWSCredentials.isConfigured() else {
            throw SyncError.credentialsNotConfigured
        }

        await MainActor.run { self.syncState = .syncing }

        do {
            // 1. Upload the PDF file
            let pdfKey = "pdfs/shared/\(hash)_original.pdf"
            try await uploadData(data, toKey: pdfKey, contentType: "application/pdf")
            print("✅ Uploaded PDF to \(pdfKey)")

            // 2. Update pdf-sync-state.json
            try await updateSyncState(hash: hash, s3Key: pdfKey, originalPath: "shared/\(fileName)")

            // 3. Update pdf-index.json
            try await updatePDFIndex(hash: hash, fileName: fileName, pageCount: pageCount)

            // 4. Reload state so the PDF appears in library
            S3StateManager.shared.loadState()

            await MainActor.run {
                self.syncState = .synced
                self.lastSyncDate = Date()
            }

            print("✅ Successfully uploaded and indexed shared PDF: \(fileName)")
        } catch {
            await MainActor.run {
                self.syncState = .error(error.localizedDescription)
            }
            throw error
        }
    }

    private func uploadData(_ data: Data, toKey key: String, contentType: String, makePublic: Bool = true) async throws {
        let url = URL(string: "https://\(AWSCredentials.bucket).s3.\(AWSCredentials.region).amazonaws.com/\(key)")!
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        if makePublic {
            request.setValue("public-read", forHTTPHeaderField: "x-amz-acl")
        }

        try signRequest(&request, body: data)

        let (responseData, response) = try await URLSession.shared.upload(for: request, from: data)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            if let responseString = String(data: responseData, encoding: .utf8) {
                print("❌ S3 Error: \(responseString)")
            }
            throw SyncError.uploadFailed
        }
    }

    private func updateSyncState(hash: String, s3Key: String, originalPath: String) async throws {
        // Download current state
        let stateURL = URL(string: "https://\(AWSCredentials.bucket).s3.\(AWSCredentials.region).amazonaws.com/pdf-sync-state.json")!

        var currentState: [String: Any] = [:]
        var uploaded: [String: Any] = [:]

        // Try to fetch existing state
        if let (data, response) = try? await URLSession.shared.data(from: stateURL),
           let httpResponse = response as? HTTPURLResponse,
           httpResponse.statusCode == 200,
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            currentState = json
            uploaded = (json["uploaded"] as? [String: Any]) ?? [:]
        }

        // Add new entry
        let dateFormatter = ISO8601DateFormatter()
        uploaded[s3Key] = [
            "originalPath": originalPath,
            "hash": hash,
            "uploadedAt": dateFormatter.string(from: Date())
        ]

        currentState["uploaded"] = uploaded

        // Upload updated state
        let updatedData = try JSONSerialization.data(withJSONObject: currentState, options: .prettyPrinted)
        try await uploadData(updatedData, toKey: "pdf-sync-state.json", contentType: "application/json")
        print("✅ Updated pdf-sync-state.json")
    }

    private func updatePDFIndex(hash: String, fileName: String, pageCount: Int) async throws {
        // Download current index
        let indexURL = URL(string: "https://\(AWSCredentials.bucket).s3.\(AWSCredentials.region).amazonaws.com/pdf-index.json")!

        var pdfs: [[String: Any]] = []

        // Try to fetch existing index
        if let (data, response) = try? await URLSession.shared.data(from: indexURL),
           let httpResponse = response as? HTTPURLResponse,
           httpResponse.statusCode == 200,
           let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            pdfs = json
        }

        // Check if this hash already exists
        if pdfs.contains(where: { ($0["hash"] as? String) == hash }) {
            print("ℹ️ PDF already in index, skipping")
            return
        }

        // Add new entry
        let dateFormatter = ISO8601DateFormatter()
        let newEntry: [String: Any] = [
            "hash": hash,
            "path": "shared/\(fileName)",
            "fileName": fileName,
            "title": NSNull(),
            "author": NSNull(),
            "metadataFound": false,
            "totalPages": pageCount,
            "creator": NSNull(),
            "producer": NSNull(),
            "creationDate": NSNull(),
            "error": NSNull(),
            "processedAt": dateFormatter.string(from: Date()),
            "ocr_title": NSNull(),
            "ocr_author": NSNull()
        ]

        pdfs.append(newEntry)

        // Upload updated index
        let updatedData = try JSONSerialization.data(withJSONObject: pdfs, options: .prettyPrinted)
        try await uploadData(updatedData, toKey: "pdf-index.json", contentType: "application/json")
        print("✅ Updated pdf-index.json")
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

        // Check if x-amz-acl header is set (for public uploads)
        let aclHeader = request.value(forHTTPHeaderField: "x-amz-acl")
        let canonicalHeaders: String
        let signedHeaders: String

        if let acl = aclHeader {
            canonicalHeaders = "host:\(request.url!.host!)\nx-amz-acl:\(acl)\nx-amz-content-sha256:\(payloadHash)\nx-amz-date:\(timestamp)\n"
            signedHeaders = "host;x-amz-acl;x-amz-content-sha256;x-amz-date"
        } else {
            canonicalHeaders = "host:\(request.url!.host!)\nx-amz-content-sha256:\(payloadHash)\nx-amz-date:\(timestamp)\n"
            signedHeaders = "host;x-amz-content-sha256;x-amz-date"
        }

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
