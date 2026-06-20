// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "CumulativeTracker",
    platforms: [.macOS(.v13), .iOS(.v17)],
    products: [
        .library(name: "CalorieModel", targets: ["CalorieModel"]),
        .executable(name: "ScenarioRunner", targets: ["ScenarioRunner"]),
    ],
    targets: [
        .target(name: "CalorieModel"),
        .executableTarget(
            name: "ScenarioRunner",
            dependencies: ["CalorieModel"]
        ),
        .testTarget(
            name: "CalorieModelTests",
            dependencies: ["CalorieModel"]
        ),
    ]
)
