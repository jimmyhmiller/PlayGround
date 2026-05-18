#if !MAS

import Foundation
import Security
import IOKit
import Combine

/// Activates and validates a Lemon Squeezy license key.
///
/// The Lemon Squeezy v1 Licenses API has three endpoints we use:
///   POST /v1/licenses/activate   — first-time activation (binds key → instance)
///   POST /v1/licenses/validate   — periodic check that the activation is alive
///   POST /v1/licenses/deactivate — release the instance (e.g. "sign out")
///
/// Docs: https://docs.lemonsqueezy.com/api/license-api
///
/// The MAS build does not need this — App Store purchases are handled by
/// Apple. We gate the whole file behind `#if !MAS` to keep the App Store
/// binary free of any third-party license enforcement.
@MainActor
final class LicenseManager: ObservableObject {
    static let shared = LicenseManager()

    enum State: Equatable {
        case unactivated
        case activated(LicenseInfo)
        case invalid(reason: String)
        case checking
    }

    struct LicenseInfo: Codable, Equatable {
        let licenseKey: String
        let instanceID: String
        let instanceName: String
        let activatedAt: Date
        var lastValidatedAt: Date
    }

    @Published private(set) var state: State

    private let baseURL = URL(string: "https://api.lemonsqueezy.com/v1")!
    private let keychainAccount = "com.jimmyhmiller.Ease.license"

    private init() {
        if let info = Self.loadFromKeychain() {
            self.state = .activated(info)
        } else {
            self.state = .unactivated
        }
    }

    // MARK: - Public API

    func activate(licenseKey: String) async {
        state = .checking

        let instanceName = Self.machineInstanceName()
        var request = URLRequest(url: baseURL.appendingPathComponent("licenses/activate"))
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let body = "license_key=\(licenseKey)&instance_name=\(instanceName)"
        request.httpBody = body.data(using: .utf8)

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                state = .invalid(reason: "Bad response from server")
                return
            }
            let decoded = try JSONDecoder().decode(ActivateResponse.self, from: data)
            guard http.statusCode == 200, decoded.activated, let instance = decoded.instance else {
                state = .invalid(reason: decoded.error ?? "Activation failed (HTTP \(http.statusCode))")
                return
            }

            let info = LicenseInfo(
                licenseKey: licenseKey,
                instanceID: instance.id,
                instanceName: instance.name,
                activatedAt: Date(),
                lastValidatedAt: Date()
            )
            try Self.saveToKeychain(info)
            state = .activated(info)
        } catch {
            state = .invalid(reason: error.localizedDescription)
        }
    }

    func validate() async {
        guard case let .activated(info) = state else { return }

        var request = URLRequest(url: baseURL.appendingPathComponent("licenses/validate"))
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let body = "license_key=\(info.licenseKey)&instance_id=\(info.instanceID)"
        request.httpBody = body.data(using: .utf8)

        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            let decoded = try JSONDecoder().decode(ValidateResponse.self, from: data)
            if decoded.valid {
                var updated = info
                updated.lastValidatedAt = Date()
                try? Self.saveToKeychain(updated)
                state = .activated(updated)
            } else {
                Self.clearKeychain()
                state = .invalid(reason: decoded.error ?? "License no longer valid")
            }
        } catch {
            // Network failure on validate: trust the existing activation
            // (avoid locking users out when offline). We still updated nothing.
            return
        }
    }

    func deactivate() async {
        guard case let .activated(info) = state else { return }

        var request = URLRequest(url: baseURL.appendingPathComponent("licenses/deactivate"))
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        let body = "license_key=\(info.licenseKey)&instance_id=\(info.instanceID)"
        request.httpBody = body.data(using: .utf8)
        _ = try? await URLSession.shared.data(for: request)

        Self.clearKeychain()
        state = .unactivated
    }

    // MARK: - Keychain persistence

    private static func saveToKeychain(_ info: LicenseInfo) throws {
        let data = try JSONEncoder().encode(info)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: "com.jimmyhmiller.Ease.license",
            kSecAttrService as String: "Ease"
        ]
        SecItemDelete(query as CFDictionary)
        var addQuery = query
        addQuery[kSecValueData as String] = data
        let status = SecItemAdd(addQuery as CFDictionary, nil)
        if status != errSecSuccess {
            throw NSError(domain: "Ease.LicenseManager", code: Int(status))
        }
    }

    private static func loadFromKeychain() -> LicenseInfo? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: "com.jimmyhmiller.Ease.license",
            kSecAttrService as String: "Ease",
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data,
              let info = try? JSONDecoder().decode(LicenseInfo.self, from: data) else {
            return nil
        }
        return info
    }

    private static func clearKeychain() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: "com.jimmyhmiller.Ease.license",
            kSecAttrService as String: "Ease"
        ]
        SecItemDelete(query as CFDictionary)
    }

    /// Stable per-machine identifier for the Lemon Squeezy `instance_name`.
    /// Uses the IOPlatformUUID, which persists across reboots and reinstalls
    /// but changes on hardware swap.
    private static func machineInstanceName() -> String {
        let platformExpert = IOServiceGetMatchingService(
            kIOMainPortDefault,
            IOServiceMatching("IOPlatformExpertDevice")
        )
        defer { IOObjectRelease(platformExpert) }
        guard platformExpert != 0,
              let uuid = IORegistryEntryCreateCFProperty(
                  platformExpert,
                  "IOPlatformUUID" as CFString,
                  kCFAllocatorDefault,
                  0
              )?.takeRetainedValue() as? String else {
            return Host.current().localizedName ?? "unknown-mac"
        }
        return uuid
    }
}

// MARK: - Wire types matching the Lemon Squeezy API

private struct ActivateResponse: Decodable {
    let activated: Bool
    let error: String?
    let instance: Instance?

    struct Instance: Decodable {
        let id: String
        let name: String
    }
}

private struct ValidateResponse: Decodable {
    let valid: Bool
    let error: String?
}

#endif
