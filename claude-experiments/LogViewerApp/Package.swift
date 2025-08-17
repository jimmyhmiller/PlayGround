// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "LogViewerApp",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "LogViewerApp",
            targets: ["LogViewerApp"]
        )
    ],
    targets: [
        .executableTarget(
            name: "LogViewerApp",
            path: "Sources"
        )
    ]
)