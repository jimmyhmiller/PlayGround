// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SwiftDashboard",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "SwiftDashboard",
            targets: ["SwiftDashboard"])
    ],
    targets: [
        .executableTarget(
            name: "SwiftDashboard",
            path: "Sources")
    ]
)
