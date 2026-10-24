import AVFoundation
import CoreImage
import Photos
import UIKit

struct ZoomOption: Identifiable, Hashable {
    let id: String     // 显示标签，如 "0.5"
    let factor: CGFloat
}

final class CameraManager: NSObject, ObservableObject {
    // 界面状态
    @Published var fx: FXParams = Presets.all[1].fx { didSet { storeFX() } }
    @Published var presetName: String = Presets.all[1].id
    @Published var ev: Float = 0 { didSet { applyEV() } }
    @Published var zoomOptions: [ZoomOption] = []
    @Published var zoomLabel: String = "1"
    @Published var saveOriginal = false
    @Published var toast: String?
    @Published var busy = false
    @Published var authorized = true

    let renderer = Renderer()
    weak var metalLayer: CAMetalLayer?

    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "cam.session")
    private let videoQueue = DispatchQueue(label: "cam.video", qos: .userInteractive)
    private let photoQueue = DispatchQueue(label: "cam.photo", qos: .userInitiated)
    private let videoOutput = AVCaptureVideoDataOutput()
    private let photoOutput = AVCapturePhotoOutput()
    private var device: AVCaptureDevice?
    private var photoDims = CMVideoDimensions(width: 0, height: 0)

    private let fxLock = NSLock()
    private var fxShared: FXParams = Presets.all[1].fx
    private let ciContext = CIContext()

    private func storeFX() {
        fxLock.lock(); fxShared = fx; fxLock.unlock()
    }

    private func currentFX() -> FXParams {
        fxLock.lock(); var p = fxShared; fxLock.unlock()
        p.seed = Float.random(in: 0...100)
        return p
    }

    func apply(_ preset: Preset) {
        presetName = preset.id
        fx = preset.fx
    }

    // MARK: - 启动

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            sessionQueue.async { self.configure() }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { ok in
                DispatchQueue.main.async { self.authorized = ok }
                if ok { self.sessionQueue.async { self.configure() } }
            }
        default:
            authorized = false
        }
    }

    private func configure() {
        if session.isRunning { return }
        session.beginConfiguration()
        session.sessionPreset = .photo
        session.automaticallyConfiguresCaptureDeviceForWideColor = false

        let dev = AVCaptureDevice.default(.builtInTripleCamera, for: .video, position: .back)
            ?? AVCaptureDevice.default(.builtInDualWideCamera, for: .video, position: .back)
            ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        guard let dev, let input = try? AVCaptureDeviceInput(device: dev), session.canAddInput(input) else {
            session.commitConfiguration(); return
        }
        session.addInput(input)
        device = dev

        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }
        if session.canAddOutput(photoOutput) { session.addOutput(photoOutput) }

        for conn in [videoOutput.connection(with: .video), photoOutput.connection(with: .video)] {
            if let c = conn, c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }
        }

        // 控制拍照尺寸在 12MP 左右，保证 GPU 内存与速度
        let dims = dev.activeFormat.supportedMaxPhotoDimensions
        let best = dims.filter { Int($0.width) * Int($0.height) <= 12_600_000 }
            .max { Int($0.width) * Int($0.height) < Int($1.width) * Int($1.height) } ?? dims.first
        if let best {
            photoOutput.maxPhotoDimensions = best
            photoDims = best
        }
        photoOutput.maxPhotoQualityPrioritization = .quality

        do {
            try dev.lockForConfiguration()
            if dev.activeFormat.supportedColorSpaces.contains(.sRGB) { dev.activeColorSpace = .sRGB }
            if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
            if dev.isExposureModeSupported(.continuousAutoExposure) { dev.exposureMode = .continuousAutoExposure }
            dev.unlockForConfiguration()
        } catch {}

        session.commitConfiguration()
        buildZoomOptions(dev)
        session.startRunning()
    }

    // MARK: - 变焦

    private func buildZoomOptions(_ dev: AVCaptureDevice) {
        let switches = dev.virtualDeviceSwitchOverVideoZoomFactors.map { CGFloat($0.doubleValue) }
        let hasUltraWide = dev.deviceType == .builtInTripleCamera || dev.deviceType == .builtInDualWideCamera
        let wide: CGFloat = hasUltraWide ? (switches.first ?? 2) : 1
        var opts: [ZoomOption] = []
        if hasUltraWide { opts.append(ZoomOption(id: "0.5", factor: 1)) }
        opts.append(ZoomOption(id: "1", factor: wide))
        opts.append(ZoomOption(id: "2", factor: wide * 2))
        if dev.deviceType == .builtInTripleCamera, let tele = switches.last, tele > wide * 2 {
            let ratio = tele / wide
            let label = ratio == ratio.rounded() ? String(Int(ratio)) : String(format: "%.1f", ratio)
            opts.append(ZoomOption(id: label, factor: tele))
        }
        let maxZ = dev.maxAvailableVideoZoomFactor
        let valid = opts.filter { $0.factor <= maxZ }
        if let one = valid.first(where: { $0.id == "1" }) { setZoom(one) }
        DispatchQueue.main.async { self.zoomOptions = valid }
    }

    func setZoom(_ opt: ZoomOption) {
        DispatchQueue.main.async { self.zoomLabel = opt.id }
        sessionQueue.async {
            guard let dev = self.device, (try? dev.lockForConfiguration()) != nil else { return }
            dev.videoZoomFactor = min(max(opt.factor, dev.minAvailableVideoZoomFactor), dev.maxAvailableVideoZoomFactor)
            dev.unlockForConfiguration()
        }
    }

    // MARK: - 曝光与对焦

    private func applyEV() {
        let v = ev
        sessionQueue.async {
            guard let dev = self.device, (try? dev.lockForConfiguration()) != nil else { return }
            dev.setExposureTargetBias(min(max(v, dev.minExposureTargetBias), dev.maxExposureTargetBias), completionHandler: nil)
            dev.unlockForConfiguration()
        }
    }

    /// u, v：竖屏画面内的归一化坐标（左上为 0,0）
    func focus(u: CGFloat, v: CGFloat) {
        let point = CGPoint(x: v, y: 1 - u) // 竖屏 → 传感器横向坐标
        sessionQueue.async {
            guard let dev = self.device, (try? dev.lockForConfiguration()) != nil else { return }
            if dev.isFocusPointOfInterestSupported {
                dev.focusPointOfInterest = point
                dev.focusMode = dev.isFocusModeSupported(.continuousAutoFocus) ? .continuousAutoFocus : .autoFocus
            }
            if dev.isExposurePointOfInterestSupported {
                dev.exposurePointOfInterest = point
                dev.exposureMode = .continuousAutoExposure
            }
            dev.unlockForConfiguration()
        }
    }

    // MARK: - 拍照

    func capture() {
        guard !busy else { return }
        busy = true
        sessionQueue.async {
            let settings: AVCapturePhotoSettings
            if self.photoOutput.availablePhotoPixelFormatTypes.contains(kCVPixelFormatType_32BGRA) {
                settings = AVCapturePhotoSettings(format: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
            } else {
                settings = AVCapturePhotoSettings()
            }
            if self.photoDims.width > 0 { settings.maxPhotoDimensions = self.photoDims }
            settings.photoQualityPrioritization = .balanced
            self.photoOutput.capturePhoto(with: settings, delegate: self)
        }
    }

    private func finish(_ message: String) {
        DispatchQueue.main.async {
            self.busy = false
            self.toast = message
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.6) { self.toast = nil }
        }
    }

    private func heif(_ image: CGImage, orientation: CGImagePropertyOrientation) -> Data? {
        let ci = CIImage(cgImage: image).oriented(orientation)
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        let opts: [CIImageRepresentationOption: Any] = [
            CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): 0.92
        ]
        return ciContext.heifRepresentation(of: ci, format: .RGBA8, colorSpace: cs, options: opts)
    }

    private func save(_ datas: [Data]) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                self.finish("没有相册写入权限"); return
            }
            PHPhotoLibrary.shared().performChanges({
                for d in datas {
                    PHAssetCreationRequest.forAsset().addResource(with: .photo, data: d, options: nil)
                }
            }) { ok, _ in
                self.finish(ok ? "已保存" : "保存失败")
            }
        }
    }
}

// MARK: - 实时帧

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer), let layer = metalLayer, let r = renderer else { return }
        r.renderPreview(pixelBuffer: pb, params: currentFX(), layer: layer)
    }
}

// MARK: - 照片回调

extension CameraManager: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if error != nil { finish("拍摄失败"); return }
        let params = currentFX()
        let rawOrient = (photo.metadata[kCGImagePropertyOrientation as String] as? NSNumber)?.uint32Value ?? 6
        let orientation = CGImagePropertyOrientation(rawValue: rawOrient) ?? .right
        let keepOriginal = saveOriginal

        // 先拿到 BGRA 位图（未压缩优先，否则解码）
        var bgra: (Data, Int, Int, Int)?
        if let pb = photo.pixelBuffer, CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_32BGRA {
            CVPixelBufferLockBaseAddress(pb, .readOnly)
            if let base = CVPixelBufferGetBaseAddress(pb) {
                let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb), bpr = CVPixelBufferGetBytesPerRow(pb)
                bgra = (Data(bytes: base, count: bpr * h), w, h, bpr)
            }
            CVPixelBufferUnlockBaseAddress(pb, .readOnly)
        } else if let cg = photo.cgImageRepresentation() {
            bgra = Self.bgraData(from: cg)
        }
        guard let bg = bgra, let r = renderer else { finish("拍摄失败"); return }
        let (data, w, h, bpr) = bg

        photoQueue.async {
            var outputs: [Data] = []
            let processed: CGImage? = data.withUnsafeBytes { buf in
                guard let base = buf.baseAddress else { return nil }
                return r.processPhoto(bgraBytes: base, width: w, height: h, bytesPerRow: bpr, params: params)
            }
            if let img = processed, let d = self.heif(img, orientation: orientation) { outputs.append(d) }
            if keepOriginal, let orig = Self.cgImage(fromBGRA: data, w, h, bpr), let d = self.heif(orig, orientation: orientation) {
                outputs.append(d)
            }
            if outputs.isEmpty { self.finish("处理失败") } else { self.save(outputs) }
        }
    }

    static func bgraData(from cg: CGImage) -> (Data, Int, Int, Int)? {
        let w = cg.width, h = cg.height, bpr = w * 4
        var data = Data(count: bpr * h)
        let ok: Bool = data.withUnsafeMutableBytes { buf in
            guard let base = buf.baseAddress, let cs = CGColorSpace(name: CGColorSpace.sRGB),
                  let ctx = CGContext(data: base, width: w, height: h, bitsPerComponent: 8, bytesPerRow: bpr, space: cs,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
            else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        return ok ? (data, w, h, bpr) : nil
    }

    static func cgImage(fromBGRA data: Data, _ w: Int, _ h: Int, _ bpr: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: data as CFData), let cs = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: bpr, space: cs,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}
