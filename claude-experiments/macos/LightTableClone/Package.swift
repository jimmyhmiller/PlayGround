// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "LightTableClone",
    platforms: [
        .macOS(.v11)
    ],
    products: [
        .executable(
            name: "LightTableClone",
            targets: ["LightTableClone"]
        ),
    ],
    targets: [
        .executableTarget(
            name: "LightTableClone",
            dependencies: []
        ),
    ]
)