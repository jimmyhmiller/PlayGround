//
//  AWSCredentials.swift
//  PDFMarkup
//
//  Reads AWS credentials from bundled config file (add aws-config.plist to project, gitignored)
//

import Foundation

struct AWSCredentials {
    private static var cachedConfig: [String: String]?

    static var accessKeyId: String {
        return configValue(for: "AWS_ACCESS_KEY_ID") ?? ""
    }

    static var secretAccessKey: String {
        return configValue(for: "AWS_SECRET_ACCESS_KEY") ?? ""
    }

    static var bucket: String {
        return configValue(for: "AWS_S3_BUCKET") ?? ""
    }

    static var region: String {
        return configValue(for: "AWS_REGION") ?? "us-east-1"
    }

    static func isConfigured() -> Bool {
        return !accessKeyId.isEmpty && !secretAccessKey.isEmpty && !bucket.isEmpty
    }

    private static func configValue(for key: String) -> String? {
        // Use cached config if available
        if let cached = cachedConfig {
            return cached[key]
        }

        // Load from app bundle
        guard let bundlePath = Bundle.main.path(forResource: "aws-config", ofType: "plist"),
              let dict = NSDictionary(contentsOfFile: bundlePath) as? [String: String] else {
            return nil
        }

        cachedConfig = dict
        return dict[key]
    }
}
