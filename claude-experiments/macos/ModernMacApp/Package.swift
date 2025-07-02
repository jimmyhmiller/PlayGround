// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "ModernMacApp",
    platforms: [
        .macOS(.v14) // Using macOS 14 as base (macOS 26 would be too new for current Xcode)
    ],
    products: [
        .executable(
            name: "ModernMacApp",
            targets: ["ModernMacApp"]
        )
    ],
    targets: [
        .executableTarget(
            name: "ModernMacApp",
            path: "Sources"
        )
    ]
)