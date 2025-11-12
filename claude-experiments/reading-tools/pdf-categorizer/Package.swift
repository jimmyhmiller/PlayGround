// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "PDFCategorizer",
    platforms: [
        .macOS(.v13)
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "PDFCategorizer",
            dependencies: [],
            path: "Sources/PDFCategorizer",
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ]
        )
    ]
)
