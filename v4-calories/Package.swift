// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "CalorieModel",
    platforms: [.macOS(.v13), .iOS(.v17)],
    products: [
        .library(name: "CalorieModel", targets: ["CalorieModel"]),
        .executable(name: "ScenarioRunner", targets: ["ScenarioRunner"]),
        .executable(name: "RealDataDiag", targets: ["RealDataDiag"]),
    ],
    targets: [
        .target(name: "CalorieModel"),
        .executableTarget(
            name: "ScenarioRunner",
            dependencies: ["CalorieModel"]
        ),
        .executableTarget(
            name: "RealDataDiag",
            dependencies: ["CalorieModel"]
        ),
        .testTarget(
            name: "CalorieModelTests",
            dependencies: ["CalorieModel"]
        ),
    ]
)
