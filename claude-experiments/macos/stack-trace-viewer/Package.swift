// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "StackTraceViewer",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "StackTraceViewer",
            dependencies: [],
            path: "Sources"
        )
    ]
)