import Foundation
import CoreML
import Vision
import CoreImage
import CoreVideo

enum DepthAnythingModelLoader {
    static func loadBundledModel(configuration: MLModelConfiguration) throws -> MLModel {
        guard let modelURL = Bundle.main.url(forResource: "DepthAnythingV2SmallF16", withExtension: "mlmodelc") else {
            throw DepthEstimatorError.modelNotFound
        }

        return try MLModel(contentsOf: modelURL, configuration: configuration)
    }
}

/// Lightweight wrapper around a Depth Anything V2 Core ML model for visionOS.
final class DepthAnythingEstimator {
    private let request: VNCoreMLRequest
    private let ciContext = CIContext()
    private let maxInferenceDimension = 640
    private var inferencePixelBufferPool: CVPixelBufferPool?
    private var inferencePoolWidth = 0
    private var inferencePoolHeight = 0
    private var inferencePoolPixelFormat: OSType = 0

    init(model: MLModel) throws {
        let vnModel = try VNCoreMLModel(for: model)
        request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .scaleFill
    }

    convenience init(fromGeneratedModel generatedModel: MLModel) throws {
        try self.init(model: generatedModel)
    }

    /// Runs depth inference on a CVPixelBuffer and returns the depth map as MLMultiArray.
    func predictDepth(pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        let inferencePixelBuffer = try makeInferencePixelBuffer(from: pixelBuffer)
        let handler = VNImageRequestHandler(cvPixelBuffer: inferencePixelBuffer, orientation: .up)
        try handler.perform([request])

        guard let result = request.results?.first else {
            throw DepthEstimatorError.noResult
        }

        if let observation = result as? VNCoreMLFeatureValueObservation,
           let array = observation.featureValue.multiArrayValue {
            return array
        }

        if let observation = result as? VNCoreMLFeatureValueObservation,
           let imageBuffer = observation.featureValue.imageBufferValue {
            return try makeDepthArray(from: imageBuffer)
        }

        if let observation = result as? VNPixelBufferObservation {
            return try makeDepthArray(from: observation.pixelBuffer)
        }

        throw DepthEstimatorError.unsupportedObservation(String(describing: type(of: result)))
    }

    private func makeInferencePixelBuffer(from pixelBuffer: CVPixelBuffer) throws -> CVPixelBuffer {
        let sourceWidth = CVPixelBufferGetWidth(pixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(pixelBuffer)
        let longestEdge = max(sourceWidth, sourceHeight)

        guard longestEdge > maxInferenceDimension else {
            return pixelBuffer
        }

        let scale = Double(maxInferenceDimension) / Double(longestEdge)
        var targetWidth = max(1, Int((Double(sourceWidth) * scale).rounded()))
        var targetHeight = max(1, Int((Double(sourceHeight) * scale).rounded()))

        // Keep dimensions even to avoid odd-size edge cases in downstream kernels.
        targetWidth -= targetWidth % 2
        targetHeight -= targetHeight % 2
        if targetWidth <= 0 { targetWidth = 2 }
        if targetHeight <= 0 { targetHeight = 2 }

        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        if inferencePixelBufferPool == nil ||
            inferencePoolWidth != targetWidth ||
            inferencePoolHeight != targetHeight ||
            inferencePoolPixelFormat != pixelFormat {
            let pixelAttributes: [CFString: Any] = [
                kCVPixelBufferWidthKey: targetWidth,
                kCVPixelBufferHeightKey: targetHeight,
                kCVPixelBufferPixelFormatTypeKey: pixelFormat,
                kCVPixelBufferIOSurfacePropertiesKey: [:],
                kCVPixelBufferMetalCompatibilityKey: true,
            ]

            let poolAttributes: [CFString: Any] = [
                kCVPixelBufferPoolMinimumBufferCountKey: 2,
                kCVPixelBufferPoolAllocationThresholdKey: 4,
            ]

            var pool: CVPixelBufferPool?
            let status = CVPixelBufferPoolCreate(
                kCFAllocatorDefault,
                poolAttributes as CFDictionary,
                pixelAttributes as CFDictionary,
                &pool
            )
            guard status == kCVReturnSuccess, let pool else {
                throw DepthEstimatorError.imageCreationFailed
            }

            inferencePixelBufferPool = pool
            inferencePoolWidth = targetWidth
            inferencePoolHeight = targetHeight
            inferencePoolPixelFormat = pixelFormat
        }

        guard let pool = inferencePixelBufferPool else {
            throw DepthEstimatorError.imageCreationFailed
        }

        var output: CVPixelBuffer?
        let allocStatus = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &output)
        guard allocStatus == kCVReturnSuccess, let output else {
            return pixelBuffer
        }

        let sourceImage = CIImage(cvPixelBuffer: pixelBuffer)
        let scaledImage = sourceImage.transformed(by: CGAffineTransform(
            scaleX: CGFloat(targetWidth) / CGFloat(sourceWidth),
            y: CGFloat(targetHeight) / CGFloat(sourceHeight)
        ))

        ciContext.render(
            scaledImage,
            to: output,
            bounds: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight),
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB)
        )

        return output
    }

    /// Helper to render an MLMultiArray depth map into a displayable grayscale CIImage.
    func makeDepthPreviewImage(from depth: MLMultiArray) throws -> CIImage {
        guard depth.dataType == .float32 || depth.dataType == .float16 || depth.dataType == .double else {
            throw DepthEstimatorError.unexpectedOutputType
        }

        let shape = depth.shape.map { $0.intValue }
        let hw: (h: Int, w: Int)
        if shape.count >= 2 {
            hw = (shape[shape.count - 2], shape[shape.count - 1])
        } else {
            throw DepthEstimatorError.unexpectedOutputShape
        }

        let width = hw.w
        let height = hw.h
        let count = width * height

        var buffer = [UInt8](repeating: 0, count: count)

        var minValue = Float.greatestFiniteMagnitude
        var maxValue = -Float.greatestFiniteMagnitude

        for i in 0..<count {
            let v = depth[i].floatValue
            if v < minValue { minValue = v }
            if v > maxValue { maxValue = v }
        }

        let denom = max(maxValue - minValue, 1e-6)
        for i in 0..<count {
            let v = depth[i].floatValue
            let normalized = (v - minValue) / denom
            buffer[i] = UInt8(max(0, min(255, Int(normalized * 255.0))))
        }

        let data = Data(buffer)
        guard let provider = CGDataProvider(data: data as CFData) else {
            throw DepthEstimatorError.imageCreationFailed
        }

        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            throw DepthEstimatorError.imageCreationFailed
        }

        return CIImage(cgImage: cgImage)
    }

    private func makeDepthArray(from pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let array = try MLMultiArray(shape: [NSNumber(value: height), NSNumber(value: width)], dataType: .float32)

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let planeIndex = CVPixelBufferGetPlaneCount(pixelBuffer) > 0 ? 0 : -1
        let baseAddress: UnsafeMutableRawPointer?
        let bytesPerRow: Int

        if planeIndex >= 0 {
            baseAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, planeIndex)
            bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, planeIndex)
        } else {
            baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
            bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        }

        guard let baseAddress else {
            throw DepthEstimatorError.imageCreationFailed
        }

        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)

        switch pixelFormat {
        case kCVPixelFormatType_OneComponent16Half:
            let rowStride = bytesPerRow / MemoryLayout<UInt16>.stride
            let buffer = baseAddress.assumingMemoryBound(to: UInt16.self)
            for y in 0..<height {
                let row = buffer.advanced(by: y * rowStride)
                for x in 0..<width {
                    let value = Float(Float16(bitPattern: row[x]))
                    array[y * width + x] = NSNumber(value: value)
                }
            }

        case kCVPixelFormatType_OneComponent16:
            let rowStride = bytesPerRow / MemoryLayout<UInt16>.stride
            let buffer = baseAddress.assumingMemoryBound(to: UInt16.self)
            for y in 0..<height {
                let row = buffer.advanced(by: y * rowStride)
                for x in 0..<width {
                    let value = Float(row[x]) / Float(UInt16.max)
                    array[y * width + x] = NSNumber(value: value)
                }
            }

        case kCVPixelFormatType_OneComponent8:
            let rowStride = bytesPerRow / MemoryLayout<UInt8>.stride
            let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
            for y in 0..<height {
                let row = buffer.advanced(by: y * rowStride)
                for x in 0..<width {
                    let value = Float(row[x]) / Float(UInt8.max)
                    array[y * width + x] = NSNumber(value: value)
                }
            }

        default:
            throw DepthEstimatorError.unsupportedPixelFormat(pixelFormat)
        }

        return array
    }
}

enum DepthEstimatorError: Error, LocalizedError {
    case noResult
    case unexpectedOutputType
    case unexpectedOutputShape
    case imageCreationFailed
    case modelNotFound
    case unsupportedObservation(String)
    case unsupportedPixelFormat(OSType)

    var errorDescription: String? {
        switch self {
        case .noResult:
            return "No depth estimation result"
        case .unexpectedOutputType:
            return "Unexpected output data type"
        case .unexpectedOutputShape:
            return "Unexpected output shape"
        case .imageCreationFailed:
            return "Failed to create image"
        case .modelNotFound:
            return "Bundled Depth Anything model not found"
        case let .unsupportedObservation(observationType):
            return "Unsupported depth observation type: \(observationType)"
        case let .unsupportedPixelFormat(pixelFormat):
            return "Unsupported depth pixel format: \(pixelFormat)"
        }
    }
}
