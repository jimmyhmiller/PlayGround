// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DecibelMonitor",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "DecibelMonitor", targets: ["DecibelMonitor"])
    ],
    targets: [
        .executableTarget(
            name: "DecibelMonitor",
            path: "Sources",
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ]
        )
    ]
)
