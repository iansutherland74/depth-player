import Foundation
import OSLog

@MainActor
final class PlaybackFaultLogger {
    static let shared = PlaybackFaultLogger()
    private let logURL: URL
    private let isoFormatter: ISO8601DateFormatter
    private let logger = Logger(subsystem: "com.vision.depth-player", category: "fault")
    private let runID = UUID().uuidString
    private var sequence: UInt64 = 0

    private init() {
        let baseURL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        self.logURL = baseURL.appendingPathComponent("depthplayer-fault-log.ndjson")
        self.isoFormatter = ISO8601DateFormatter()
        self.isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]

        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }

        log("logger-initialized", fields: [
            "run_id": runID,
            "path": containerRelativePath,
        ])
    }

    var containerRelativePath: String {
        "Library/Caches/depthplayer-fault-log.ndjson"
    }

    var currentRunID: String {
        runID
    }

    func log(_ event: String, fields: [String: String] = [:]) {
        sequence &+= 1

        var payload: [String: String] = [
            "ts": isoFormatter.string(from: Date()),
            "event": event,
            "run_id": runID,
            "seq": String(sequence),
            "uptime_s": String(format: "%.3f", ProcessInfo.processInfo.systemUptime),
        ]
        for (key, value) in fields {
            payload[key] = value
        }

        logger.notice("event=\(event, privacy: .public) seq=\(self.sequence, privacy: .public) fields=\(String(describing: fields), privacy: .public)")

        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: []),
              var line = String(data: data, encoding: .utf8) else {
            return
        }
        line.append("\n")

        guard let lineData = line.data(using: .utf8) else { return }
        if let handle = try? FileHandle(forWritingTo: logURL) {
            defer { try? handle.close() }
            do {
                try handle.seekToEnd()
                try handle.write(contentsOf: lineData)
            } catch {
                return
            }
        }
    }

    func reset() {
        try? Data().write(to: logURL, options: .atomic)
        sequence = 0
        log("logger-reset", fields: ["run_id": runID])
    }
}
