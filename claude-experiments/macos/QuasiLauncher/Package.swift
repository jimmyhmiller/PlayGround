// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "QuasiLauncher",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "QuasiLauncher",
            targets: ["QuasiLauncher"]
        )
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "QuasiLauncher",
            dependencies: [],
            path: "Sources"
        )
    ]
)