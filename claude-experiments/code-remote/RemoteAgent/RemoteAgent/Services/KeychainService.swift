import Foundation
import Security

/// Service for securely storing and retrieving passwords from Keychain
/// Integrates with Face ID/Touch ID automatically on iOS
enum KeychainService {

    private static let serviceName = "com.remoteagent.ssh"

    /// Save a password for a server
    static func savePassword(_ password: String, for serverID: UUID) throws {
        let account = serverID.uuidString
        let passwordData = password.data(using: .utf8)!

        // Delete any existing password first
        try? deletePassword(for: serverID)

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            kSecValueData as String: passwordData,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]

        let status = SecItemAdd(query as CFDictionary, nil)

        guard status == errSecSuccess else {
            throw KeychainError.saveFailed(status)
        }
    }

    /// Retrieve a password for a server
    static func getPassword(for serverID: UUID) throws -> String? {
        let account = serverID.uuidString

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        switch status {
        case errSecSuccess:
            guard let data = result as? Data,
                  let password = String(data: data, encoding: .utf8) else {
                return nil
            }
            return password

        case errSecItemNotFound:
            return nil

        default:
            throw KeychainError.retrieveFailed(status)
        }
    }

    /// Delete a password for a server
    static func deletePassword(for serverID: UUID) throws {
        let account = serverID.uuidString

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account
        ]

        let status = SecItemDelete(query as CFDictionary)

        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.deleteFailed(status)
        }
    }

    /// Check if a password exists for a server
    static func hasPassword(for serverID: UUID) -> Bool {
        let account = serverID.uuidString

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            kSecReturnData as String: false
        ]

        let status = SecItemCopyMatching(query as CFDictionary, nil)
        return status == errSecSuccess
    }
}

// MARK: - Keychain Errors

enum KeychainError: Error, LocalizedError {
    case saveFailed(OSStatus)
    case retrieveFailed(OSStatus)
    case deleteFailed(OSStatus)

    var errorDescription: String? {
        switch self {
        case .saveFailed(let status):
            return "Failed to save password to Keychain (status: \(status))"
        case .retrieveFailed(let status):
            return "Failed to retrieve password from Keychain (status: \(status))"
        case .deleteFailed(let status):
            return "Failed to delete password from Keychain (status: \(status))"
        }
    }
}
