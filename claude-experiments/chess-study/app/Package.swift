// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ChessStudyApp",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        .executable(name: "ChessStudyApp", targets: ["ChessStudyApp"])
    ],
    targets: [
        .executableTarget(
            name: "ChessStudyApp",
            path: "Sources",
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ]
        )
    ]
)
