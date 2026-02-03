// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "FlappyBird",
    platforms: [
        .macOS(.v13)
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "FlappyBird",
            dependencies: [],
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ]
        )
    ]
)
