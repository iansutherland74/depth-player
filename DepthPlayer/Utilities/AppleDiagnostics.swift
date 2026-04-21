import Foundation
import MetricKit
import OSLog

@MainActor
final class AppleDiagnostics: NSObject, MXMetricManagerSubscriber {
    static let shared = AppleDiagnostics()

    private let logger = Logger(subsystem: "com.vision.depth-player", category: "apple-diagnostics")
    private let isoFormatter: ISO8601DateFormatter
    private let diagnosticsURL: URL
    private var started = false

    private override init() {
        self.isoFormatter = ISO8601DateFormatter()
        self.isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]

        let baseURL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        self.diagnosticsURL = baseURL.appendingPathComponent("depthplayer-metrickit-diagnostics.ndjson")

        super.init()

        if !FileManager.default.fileExists(atPath: diagnosticsURL.path) {
            FileManager.default.createFile(atPath: diagnosticsURL.path, contents: nil)
        }
    }

    func start() {
        guard !started else { return }
        started = true

        PlaybackFaultLogger.shared.log("apple-diagnostics-start", fields: [
            "run_id": PlaybackFaultLogger.shared.currentRunID,
            "metrickit_path": "Library/Caches/depthplayer-metrickit-diagnostics.ndjson",
        ])

        MXMetricManager.shared.add(self)

        logger.notice("MetricKit subscriber registered")

        for payload in MXMetricManager.shared.pastDiagnosticPayloads {
            persistDiagnosticPayload(payload, source: "past")
        }
    }

    nonisolated func didReceive(_ payloads: [MXMetricPayload]) {
        let count = payloads.count
        Task { @MainActor in
            PlaybackFaultLogger.shared.log("metrickit-metrics-received", fields: [
                "count": String(count),
            ])
        }
    }

    nonisolated func didReceive(_ payloads: [MXDiagnosticPayload]) {
        let count = payloads.count
        let encodedPayloads = payloads.map { $0.jsonRepresentation() }
        Task { @MainActor in
            PlaybackFaultLogger.shared.log("metrickit-diagnostics-received", fields: [
                "count": String(count),
            ])
            for payloadData in encodedPayloads {
                persistDiagnosticPayloadData(payloadData, source: "live")
            }
        }
    }

    private func persistDiagnosticPayload(_ payload: MXDiagnosticPayload, source: String) {
        persistDiagnosticPayloadData(payload.jsonRepresentation(), source: source)
    }

    private func persistDiagnosticPayloadData(_ payloadData: Data, source: String) {
        guard var payloadObject = (try? JSONSerialization.jsonObject(with: payloadData)) as? [String: Any] else {
            PlaybackFaultLogger.shared.log("metrickit-payload-json-invalid", fields: ["source": source])
            return
        }

        payloadObject["ts"] = isoFormatter.string(from: Date())
        payloadObject["source"] = source
        payloadObject["run_id"] = PlaybackFaultLogger.shared.currentRunID

        guard let lineData = try? JSONSerialization.data(withJSONObject: payloadObject),
              var line = String(data: lineData, encoding: .utf8) else {
            PlaybackFaultLogger.shared.log("metrickit-payload-write-serialize-failed", fields: ["source": source])
            return
        }

        line.append("\n")

        guard let finalData = line.data(using: .utf8) else {
            return
        }

        do {
            let handle = try FileHandle(forWritingTo: diagnosticsURL)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: finalData)

            PlaybackFaultLogger.shared.log("metrickit-payload-persisted", fields: [
                "source": source,
                "bytes": String(finalData.count),
            ])
        } catch {
            PlaybackFaultLogger.shared.log("metrickit-payload-write-failed", fields: [
                "source": source,
                "error": error.localizedDescription,
            ])
        }
    }
}
