// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "WhiteboardApp",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "WhiteboardApp",
            targets: ["WhiteboardApp"]
        )
    ],
    targets: [
        .executableTarget(
            name: "WhiteboardApp",
            path: "Sources"
        )
    ]
)