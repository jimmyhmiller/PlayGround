// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "acp-test",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "acp-test",
            swiftSettings: [.unsafeFlags(["-parse-as-library"])]
        )
    ]
)
