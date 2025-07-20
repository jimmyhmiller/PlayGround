// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ClaudeCodeManager",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "ClaudeCodeManager",
            targets: ["ClaudeCodeManager"]
        ),
    ],
    targets: [
        .executableTarget(
            name: "ClaudeCodeManager",
            dependencies: []
        ),
    ]
)