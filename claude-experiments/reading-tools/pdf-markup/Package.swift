// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "PDFMarkup",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .executable(name: "PDFMarkup", targets: ["PDFMarkup"])
    ],
    targets: [
        .executableTarget(
            name: "PDFMarkup",
            path: "PDFMarkup",
            exclude: ["Info.plist", "Assets.xcassets", "sample.pdf", "pdf-index.json", "pdf-sync-state.json"],
            resources: [
                .copy("aws-config.plist"),
            ],
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ],
            linkerSettings: [
                .linkedFramework("PencilKit"),
                .linkedFramework("PDFKit"),
                .unsafeFlags(["-Xlinker", "-sectcreate", "-Xlinker", "__TEXT", "-Xlinker", "__info_plist", "-Xlinker", "EmbeddedInfo.plist"]),
            ]
        )
    ]
)
