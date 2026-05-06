// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "PaperExperiment",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .executable(
            name: "PaperExperiment",
            targets: ["PaperExperiment"]
        ),
    ],
    targets: [
        .executableTarget(
            name: "PaperExperiment",
            resources: [
                .process("Resources"),
            ]
        ),
    ]
)
