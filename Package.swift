// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DepthPlayer",
    platforms: [
        .visionOS(.v1)
    ],
    products: [
        .executable(name: "DepthPlayer", targets: ["DepthPlayer"])
    ],
    targets: [
        .executableTarget(
            name: "DepthPlayer",
            dependencies: [],
            resources: []
        )
    ]
)
