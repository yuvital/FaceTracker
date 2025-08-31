//
//  ViewController.swift
//  FaceTracker
//
//  Created by Anurag Ajwani on 08/05/2019.
//  Updated for rPPG prototyping (HR) 2025.
//

import UIKit
import AVFoundation
import Vision
import CoreImage
import Accelerate

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    // MARK: - Camera / Vision
    private let captureSession = AVCaptureSession()
    private lazy var previewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private var drawings: [CAShapeLayer] = []

    // MARK: - rPPG buffers / config
    private var rgbBuffer: [(time: CFTimeInterval, r: CGFloat, g: CGFloat, b: CGFloat)] = []
    // Longer buffer to allow 15s Welch segments
    private let bufferLengthSec: Double = 20.0

    // DSP config
    private let targetFs: Double = 30.0
    private var bpFilters: [Biquad] = []
    private var lastBpm: Double?
    private var hrHistory: [Double] = []
    private let hrHistorySize = 5

    // Projection method: .pos (default) or .chrom or .green
    private enum ProjectionMethod { case pos, chrom, green }
    private let projectionMethod: ProjectionMethod = .pos

    // MARK: - UI
    private let signalLayer = CAShapeLayer()
    private let maxPlotPoints = 300
    private let hrLabel: UILabel = {
        let label = UILabel()
        label.text = "-- bpm"
        label.font = UIFont.boldSystemFont(ofSize: 28)
        label.textColor = .systemRed
        label.textAlignment = .center
        return label
    }()

    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        self.addCameraInput()
        self.showCameraFeed()
        self.getCameraFrames()
        self.setupSignalPlot()
        self.setupHRLabel()
        self.captureSession.startRunning()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        self.previewLayer.frame = self.view.frame
        self.signalLayer.frame = CGRect(x: 0,
                                        y: self.view.frame.height - 150,
                                        width: self.view.frame.width,
                                        height: 150)
        self.hrLabel.frame = CGRect(x: 0,
                                    y: self.view.safeAreaInsets.top + 20,
                                    width: self.view.frame.width,
                                    height: 40)
    }

    // MARK: - Capture delegate
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let frame = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        self.detectFace(in: frame, buffer: sampleBuffer)
    }

    // MARK: - Setup camera
    private func addCameraInput() {
        guard let device = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .builtInDualCamera, .builtInTrueDepthCamera],
            mediaType: .video,
            position: .front
        ).devices.first else {
            fatalError("No front camera found")
        }
        let cameraInput = try! AVCaptureDeviceInput(device: device)
        self.captureSession.addInput(cameraInput)
        // Prefer 720p
        if captureSession.canSetSessionPreset(.hd1280x720) {
            captureSession.sessionPreset = .hd1280x720
        }
        // Best-effort exposure tuning to reduce drift
        do {
            try device.lockForConfiguration()
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            }
            if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                device.whiteBalanceMode = .continuousAutoWhiteBalance
            }
            device.isSubjectAreaChangeMonitoringEnabled = false
            device.unlockForConfiguration()
        } catch {
            // ignore if cannot lock
        }
    }

    private func showCameraFeed() {
        self.previewLayer.videoGravity = .resizeAspectFill
        self.view.layer.addSublayer(self.previewLayer)
        self.previewLayer.frame = self.view.frame
    }

    private func getCameraFrames() {
        self.videoDataOutput.videoSettings = [
            (kCVPixelBufferPixelFormatTypeKey as NSString): NSNumber(value: kCVPixelFormatType_32BGRA)
        ] as [String: Any]
        self.videoDataOutput.alwaysDiscardsLateVideoFrames = true
        self.videoDataOutput.setSampleBufferDelegate(self,
            queue: DispatchQueue(label: "camera_frame_processing_queue"))
        self.captureSession.addOutput(self.videoDataOutput)
        if let connection = self.videoDataOutput.connection(with: .video),
           connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }
    }

    // MARK: - Vision
    private func detectFace(in image: CVPixelBuffer, buffer: CMSampleBuffer) {
        let request = VNDetectFaceLandmarksRequest { (request, error) in
            DispatchQueue.main.async {
                if let faces = request.results as? [VNFaceObservation],
                   let face = faces.first {
                    self.handleFaceDetectionResult(face, image: image, buffer: buffer)
                } else {
                    self.clearDrawings()
                }
            }
        }
        let handler = VNImageRequestHandler(cvPixelBuffer: image,
                                            orientation: .leftMirrored,
                                            options: [:])
        try? handler.perform([request])
    }

    private func handleFaceDetectionResult(_ face: VNFaceObservation,
                                           image: CVPixelBuffer,
                                           buffer: CMSampleBuffer) {
        self.clearDrawings()

        // Draw face box
        let faceRectOnScreen = self.previewLayer.layerRectConverted(fromMetadataOutputRect: face.boundingBox)
        let shape = CAShapeLayer()
        shape.path = CGPath(rect: faceRectOnScreen, transform: nil)
        shape.fillColor = UIColor.clear.cgColor
        shape.strokeColor = UIColor.green.cgColor
        self.view.layer.addSublayer(shape)
        self.drawings = [shape]

        // ---- Two-cheek ROIs (left & right halves of lower face) ----
        let lowerFace = CGRect(
            x: face.boundingBox.origin.x,
            y: face.boundingBox.origin.y,
            width: face.boundingBox.size.width,
            height: face.boundingBox.size.height * 0.5
        )
        let leftCheek = CGRect(x: lowerFace.minX,
                               y: lowerFace.minY,
                               width: lowerFace.width * 0.5,
                               height: lowerFace.height)
        let rightCheek = CGRect(x: lowerFace.minX + lowerFace.width * 0.5,
                                y: lowerFace.minY,
                                width: lowerFace.width * 0.5,
                                height: lowerFace.height)

        // Convert to pixel coords
        let ciImage = CIImage(cvPixelBuffer: image)
        let imageSize = ciImage.extent.size
        func toPixelRect(_ norm: CGRect) -> CGRect {
            return CGRect(
                x: norm.origin.x * imageSize.width,
                y: (1 - norm.origin.y - norm.height) * imageSize.height,
                width: norm.width * imageSize.width,
                height: norm.height * imageSize.height
            )
        }
        let leftRect = toPixelRect(leftCheek)
        let rightRect = toPixelRect(rightCheek)

        // Mean RGB for both cheeks -> average
        if let L = averageRGB(in: ciImage, roi: leftRect),
           let R = averageRGB(in: ciImage, roi: rightRect) {
            let r = (L.r + R.r) * 0.5
            let g = (L.g + R.g) * 0.5
            let b = (L.b + R.b) * 0.5
            let timestamp = CMSampleBufferGetPresentationTimeStamp(buffer).seconds
            self.appendRGBSample(time: timestamp, r: r, g: g, b: b)
        }
    }

    // MARK: - RGB extraction
    private func averageRGB(in image: CIImage, roi: CGRect) -> (r: CGFloat, g: CGFloat, b: CGFloat)? {
        guard roi.width > 1, roi.height > 1 else { return nil }
        guard let filter = CIFilter(name: "CIAreaAverage",
                                    parameters: [kCIInputImageKey: image.cropped(to: roi),
                                                 kCIInputExtentKey: CIVector(cgRect: roi)]) else { return nil }
        guard let outputImage = filter.outputImage else { return nil }

        var bitmap = [UInt8](repeating: 0, count: 4)
        let context = CIContext()
        context.render(outputImage,
                       toBitmap: &bitmap,
                       rowBytes: 4,
                       bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
                       format: .RGBA8,
                       colorSpace: CGColorSpaceCreateDeviceRGB())

        return (r: CGFloat(bitmap[0]) / 255.0,
                g: CGFloat(bitmap[1]) / 255.0,
                b: CGFloat(bitmap[2]) / 255.0)
    }

    private func appendRGBSample(time: CFTimeInterval, r: CGFloat, g: CGFloat, b: CGFloat) {
        rgbBuffer.append((time, r, g, b))
        let cutoff = time - bufferLengthSec
        rgbBuffer = rgbBuffer.filter { $0.time >= cutoff }
        self.updateSignalPlot()
        self.updateHREstimate()
    }

    // MARK: - Plot (still showing green channel for quick visual)
    private func setupSignalPlot() {
        signalLayer.strokeColor = UIColor.systemGreen.cgColor
        signalLayer.fillColor = UIColor.clear.cgColor
        signalLayer.lineWidth = 2.0
        self.view.layer.addSublayer(signalLayer)
    }

    private func updateSignalPlot() {
        guard !rgbBuffer.isEmpty else { return }
        let values = rgbBuffer.map { $0.g }
        let maxVal = values.max() ?? 1.0
        let minVal = values.min() ?? 0.0
        let range = max(0.0001, maxVal - minVal)

        let path = UIBezierPath()
        let plotWidth = self.signalLayer.bounds.width
        let plotHeight = self.signalLayer.bounds.height
        let stride = max(1, values.count / maxPlotPoints)
        for (i, v) in values.enumerated() where i % stride == 0 {
            let x = CGFloat(i) / CGFloat(values.count) * plotWidth
            let yNorm = (v - minVal) / range
            let y = plotHeight * (1 - yNorm)
            if i == 0 { path.move(to: CGPoint(x: x, y: y)) }
            else { path.addLine(to: CGPoint(x: x, y: y)) }
        }
        DispatchQueue.main.async { self.signalLayer.path = path.cgPath }
    }

    // MARK: - HR UI
    private func setupHRLabel() { self.view.addSubview(hrLabel) }

    // MARK: - HR Estimation (Welch PSD + harmonic-aware peak)
    private func updateHREstimate() {
        // Need enough samples
        guard rgbBuffer.count > 120 else { return } // ~4s at 30 Hz

        // 1) Gather raw (t, r, g, b)
        let rRaw = rgbBuffer.map { Float($0.r) }
        let gRaw = rgbBuffer.map { Float($0.g) }
        let bRaw = rgbBuffer.map { Float($0.b) }
        let tRaw = rgbBuffer.map { $0.time }
        guard let t0 = tRaw.first, let tN = tRaw.last, tN > t0 else { return }

        // 2) Resample each channel to uniform 30 Hz
        let rUniform = resampleUniform(times: tRaw, values: rRaw, fs: targetFs)
        let gUniform = resampleUniform(times: tRaw, values: gRaw, fs: targetFs)
        let bUniform = resampleUniform(times: tRaw, values: bRaw, fs: targetFs)
        guard gUniform.count >= 256 else { return }

        // 3) Projection: POS / CHROM / GREEN
        var projected = projectSignal(r: rUniform, g: gUniform, b: bUniform, method: projectionMethod)

        // 4) Band-pass 0.5–2.0 Hz (focus on fundamental)
        if bpFilters.isEmpty { bpFilters = designButterBandpass(fs: targetFs, f1: 0.5, f2: 2.0) }
        for f in bpFilters { f.process(&projected) }

        // 5) Welch PSD (≈15s segments, 50% overlap, Hann)
        let segmentSeconds: Double = 15.0
        let (psd, df) = welchPSD(signal: projected, fs: targetFs, segmentSeconds: segmentSeconds, overlap: 0.5)
        guard psd.count >= 8 && df > 0 else { return }

        // 6) Harmonic-aware peak picking in 0.5–2.0 Hz
        let fMin = 0.5, fMax = 2.0
        let iMin = max(1, Int(floor(fMin / df)))
        let iMax = min(psd.count - 1, Int(ceil(fMax / df)))

        var bestIdx = iMin
        var bestScore: Float = -Float.greatestFiniteMagnitude
        var bandPower: Float = 1e-6

        if iMax > iMin {
            for i in iMin...iMax { bandPower += psd[i] }
            func score(_ i: Int) -> Float {
                var s = psd[i]
                let i2 = i * 2
                if i2 < psd.count && Double(i2) * df <= fMax { s += 0.5 * psd[i2] }
                let ih = i / 2
                if ih >= iMin && ih < psd.count && Double(ih) * df >= fMin { s -= 0.5 * psd[ih] }
                return s
            }
            for i in iMin...iMax {
                let sc = score(i)
                if sc > bestScore { bestScore = sc; bestIdx = i }
            }
        }

        let fPeak = Double(bestIdx) * df
        let bpmRaw = fPeak * 60.0

        // 7) Confidence
        let peakVal = psd[bestIdx]
        let confidence = max(0.0, min(1.0, Double(peakVal / max(1e-6, bandPower))))
        let confLabel: String = confidence > 0.5 ? "high" : (confidence > 0.3 ? "med" : "low")

        // 8) Slew limit + confidence-weighted smoothing
        var candidate = bpmRaw
        if let prev = self.lastBpm {
            let maxDelta = 5.0 // bpm per update
            if abs(candidate - prev) > maxDelta {
                candidate = prev + maxDelta * (candidate > prev ? 1 : -1)
            }
        }
        self.lastBpm = candidate

        let alpha = max(0.2, min(0.8, confidence))
        let base = (self.hrHistory.last ?? candidate)
        let blended = (1.0 - alpha) * base + alpha * candidate

        self.hrHistory.append(blended)
        if self.hrHistory.count > self.hrHistorySize { self.hrHistory.removeFirst() }
        let finalHR = self.hrHistory.reduce(0, +) / Double(self.hrHistory.count)

        DispatchQueue.main.async {
            self.hrLabel.text = "\(Int(finalHR.rounded())) bpm (\(confLabel))"
        }
    }

    // MARK: - Projection (POS / CHROM / GREEN)
    private func projectSignal(r: [Float], g: [Float], b: [Float], method: ProjectionMethod) -> [Float] {
        let n = min(r.count, g.count, b.count)
        if n == 0 { return g }
        // Helper: mean-center
        func center(_ x: [Float]) -> [Float] {
            var m: Float = 0
            vDSP_meanv(x, 1, &m, vDSP_Length(x.count))
            var neg = -m
            var y = x
            vDSP_vsadd(x, 1, &neg, &y, 1, vDSP_Length(x.count))
            return y
        }
        // Helper: std
        func std(_ x: [Float]) -> Float {
            var mean: Float = 0
            vDSP_meanv(x, 1, &mean, vDSP_Length(x.count))
            var diff = [Float](repeating: 0, count: x.count)
            var negMean = -mean
            vDSP_vsadd(x, 1, &negMean, &diff, 1, vDSP_Length(x.count))
            var sq: Float = 0
            vDSP_measqv(diff, 1, &sq, vDSP_Length(x.count))
            return sqrt(max(1e-12, sq))
        }

        switch method {
        case .green:
            return Array(g.prefix(n))

        case .chrom:
            // De Haan & Jeanne (2013)
            let rc = center(Array(r.prefix(n)))
            let gc = center(Array(g.prefix(n)))
            let bc = center(Array(b.prefix(n)))
            var X = [Float](repeating: 0, count: n)
            var Y = [Float](repeating: 0, count: n)
            for i in 0..<n {
                X[i] = 3*rc[i] - 2*gc[i]
                Y[i] = 1.5*rc[i] + gc[i] - 1.5*bc[i]
            }
            let sx = std(X), sy = std(Y)
            if sy < 1e-9 { return Array(g.prefix(n)) }
            var S = [Float](repeating: 0, count: n)
            let k = sx / sy
            for i in 0..<n { S[i] = X[i] - k * Y[i] }
            return S

        case .pos:
            // Wang et al. (2017)
            let rc = center(Array(r.prefix(n)))
            let gc = center(Array(g.prefix(n)))
            let bc = center(Array(b.prefix(n)))
            var S1 = [Float](repeating: 0, count: n)
            var S2 = [Float](repeating: 0, count: n)
            for i in 0..<n {
                S1[i] = rc[i] - gc[i]
                S2[i] = rc[i] + gc[i] - 2*bc[i]
            }
            let s1 = std(S1), s2 = std(S2)
            if s2 < 1e-9 { return Array(g.prefix(n)) }
            var S = [Float](repeating: 0, count: n)
            let alpha = s1 / s2
            for i in 0..<n { S[i] = S1[i] - alpha * S2[i] }
            return S
        }
    }

    // MARK: - Welch PSD
    // Returns (PSD half-spectrum, df)
    private func welchPSD(signal: [Float], fs: Double, segmentSeconds: Double, overlap: Double) -> ([Float], Double) {
        let N = signal.count
        guard N > 32 else { return ([], 0) }
        // Choose segment length close to fs*segmentSeconds, round to power of two
        let targetLen = Int(fs * segmentSeconds)
        let segLenPow2 = max(128, 1 << Int(round(log2(Double(targetLen)))))
        let segLen = min(segLenPow2, N)
        if segLen < 128 { return ([], 0) }
        let hop = max(1, Int(Double(segLen) * (1.0 - overlap)))
        if hop <= 0 { return ([], 0) }

        var window = [Float](repeating: 0, count: segLen)
        vDSP_hann_window(&window, vDSP_Length(segLen), Int32(vDSP_HANN_NORM))

        let half = segLen/2
        var psdAcc = [Float](repeating: 0, count: half)
        var count = 0

        let log2n = vDSP_Length(log2(Float(segLen)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return ([], 0)
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        var idx = 0
        while idx + segLen <= N {
            var seg = Array(signal[idx..<(idx + segLen)])
            // Detrend (remove mean)
            var mean: Float = 0
            vDSP_meanv(seg, 1, &mean, vDSP_Length(seg.count))
            var negMean = -mean
            vDSP_vsadd(seg, 1, &negMean, &seg, 1, vDSP_Length(seg.count))
            // Window
            vDSP_vmul(seg, 1, window, 1, &seg, 1, vDSP_Length(segLen))
            // FFT
            var realp = [Float](repeating: 0, count: half)
            var imagp = [Float](repeating: 0, count: half)
            seg.withUnsafeBufferPointer { ptr in
                ptr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) { complexPtr in
                    var split = DSPSplitComplex(realp: &realp, imagp: &imagp)
                    vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(half))
                    vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                    var mag = [Float](repeating: 0, count: half)
                    vDSP_zvmags(&split, 1, &mag, 1, vDSP_Length(half))
                    // Normalize; scale by window energy and sampling
                    var scale: Float = 2.0 / Float(segLen)
                    var psd = [Float](repeating: 0, count: half)
                    vDSP_vsmul(mag, 1, &scale, &psd, 1, vDSP_Length(half))
                    vDSP_vadd(psd, 1, psdAcc, 1, &psdAcc, 1, vDSP_Length(half))
                }
            }
            count += 1
            idx += hop
        }

        if count == 0 { return ([], 0) }
        var normCount = Float(count)
        vDSP_vsdiv(psdAcc, 1, &normCount, &psdAcc, 1, vDSP_Length(psdAcc.count))
        let df = fs / Double(segLen)
        return (psdAcc, df)
    }

    // MARK: - Helpers: resampling & filters

    // Resample irregular (time,value) to uniform fs using linear interpolation
    private func resampleUniform(times: [CFTimeInterval], values: [Float], fs: Double) -> [Float] {
        guard times.count >= 2 else { return values }
        let t0 = times.first!
        let tN = times.last!
        let nOut = Int(((tN - t0) * fs).rounded(.down)) + 1
        if nOut <= 1 { return values }

        var out = [Float](repeating: values.first ?? 0, count: nOut)
        var j = 0
        for i in 0..<nOut {
            let t = t0 + Double(i) / fs
            while j + 1 < times.count && times[j + 1] < t { j += 1 }
            if j + 1 >= times.count { out[i] = values.last!; continue }
            let t0s = times[j], t1s = times[j + 1]
            let v0 = values[j], v1 = values[j + 1]
            let alpha = Float((t - t0s) / (t1s - t0s))
            out[i] = v0 + alpha * (v1 - v0)
        }
        return out
    }

    // Simple RBJ band-pass cascaded to approximate 4th-order Butterworth
    private func designButterBandpass(fs: Double, f1: Double, f2: Double) -> [Biquad] {
        func rbjBandpass(fs: Double, fc: Double, q: Double) -> Biquad {
            let omega = 2.0 * Double.pi * fc / fs
            let sinw = sin(omega), cosw = cos(omega)
            let alpha = sinw / (2.0 * q)
            let b0 =  Float(alpha)
            let b1 =  Float(0.0)
            let b2 =  Float(-alpha)
            let a0 =  Float(1.0 + alpha)
            let a1 =  Float(-2.0 * cosw)
            let a2 =  Float(1.0 - alpha)
            let biq = Biquad()
            biq.b0 = b0 / a0; biq.b1 = b1 / a0; biq.b2 = b2 / a0
            biq.a1 = a1 / a0; biq.a2 = a2 / a0
            return biq
        }
        let fc = sqrt(f1 * f2)              // geometric center
        let biq1 = rbjBandpass(fs: fs, fc: fc, q: 0.707) // wider
        let biq2 = rbjBandpass(fs: fs, fc: fc, q: 1.414) // narrower
        return [biq1, biq2]
    }

    // Biquad IIR filter (Direct Form II Transposed)
    final class Biquad {
        var b0: Float = 1, b1: Float = 0, b2: Float = 0, a1: Float = 0, a2: Float = 0
        private var z1: Float = 0, z2: Float = 0
        func process(_ x: inout [Float]) {
            for i in 0..<x.count {
                let inVal = x[i]
                let out = b0 * inVal + z1
                z1 = b1 * inVal - a1 * out + z2
                z2 = b2 * inVal - a2 * out
                x[i] = out
            }
        }
    }

    // MARK: - Cleanup
    private func clearDrawings() {
        self.drawings.forEach { $0.removeFromSuperlayer() }
    }
}
