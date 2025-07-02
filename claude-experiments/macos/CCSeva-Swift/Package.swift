// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "CCSeva",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "CCSeva",
            targets: ["CCSeva"]
        ),
        .executable(
            name: "test-menubar",
            targets: ["TestMenuBar"]
        ),
        .executable(
            name: "claude-usage",
            targets: ["ClaudeUsageCLI"]
        ),
        .library(
            name: "ClaudeUsageCore",
            targets: ["ClaudeUsageCore"]
        ),
    ],
    dependencies: [
        // Add any external dependencies here
    ],
    targets: [
        .target(
            name: "ClaudeUsageCore",
            dependencies: []
        ),
        .executableTarget(
            name: "CCSeva",
            dependencies: ["ClaudeUsageCore"]
        ),
        .executableTarget(
            name: "TestMenuBar",
            dependencies: []
        ),
        .executableTarget(
            name: "ClaudeUsageCLI",
            dependencies: ["ClaudeUsageCore"]
        ),
    ]
)