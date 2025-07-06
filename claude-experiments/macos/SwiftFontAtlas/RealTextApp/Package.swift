// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "RealTextApp",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "..")
    ],
    targets: [
        .executableTarget(
            name: "RealTextApp",
            dependencies: ["SwiftFontAtlas"],
            resources: [
                .process("Shaders.metal")
            ],
            swiftSettings: [
                .unsafeFlags(["-Xfrontend", "-disable-availability-checking"]),
                .unsafeFlags(["-strict-concurrency=minimal"])
            ]
        )
    ]
)