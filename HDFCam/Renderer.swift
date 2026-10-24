import Metal
import QuartzCore
import CoreVideo
import CoreGraphics

/// 一套按尺寸分配的中间纹理
final class TextureSet {
    let width: Int
    let height: Int
    let lin: MTLTexture
    let bloomDowns: [MTLTexture]
    let bloomUps: [MTLTexture]
    let halMask: MTLTexture
    let haloDowns: [MTLTexture]
    let haloUps: [MTLTexture]
    let output: MTLTexture

    static let bloomLevels = 6
    static let haloLevels = 4

    init?(device: MTLDevice, width: Int, height: Int, sharedOutput: Bool) {
        self.width = width
        self.height = height

        func make(_ w: Int, _ h: Int, _ fmt: MTLPixelFormat = .rgba16Float, shared: Bool = false) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: max(1, w), height: max(1, h), mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = shared ? .shared : .private
            return device.makeTexture(descriptor: d)
        }

        guard let lin = make(width, height) else { return nil }
        self.lin = lin

        var downs: [MTLTexture] = []
        var ups: [MTLTexture] = []
        var w = width, h = height
        for _ in 0..<TextureSet.bloomLevels {
            w = max(1, w / 2); h = max(1, h / 2)
            guard let d = make(w, h), let u = make(w, h) else { return nil }
            downs.append(d); ups.append(u)
        }
        bloomDowns = downs
        bloomUps = ups

        guard let mask = make(downs[0].width, downs[0].height) else { return nil }
        halMask = mask
        var hd: [MTLTexture] = []
        var hu: [MTLTexture] = []
        w = mask.width; h = mask.height
        for _ in 0..<TextureSet.haloLevels {
            w = max(1, w / 2); h = max(1, h / 2)
            guard let d = make(w, h), let u = make(w, h) else { return nil }
            hd.append(d); hu.append(u)
        }
        haloDowns = hd
        haloUps = hu

        guard let out = make(width, height, .rgba8Unorm, shared: sharedOutput) else { return nil }
        output = out
    }
}

final class Renderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    private let prep: MTLComputePipelineState
    private let down: MTLComputePipelineState
    private let scale: MTLComputePipelineState
    private let upAdd: MTLComputePipelineState
    private let halMask: MTLComputePipelineState
    private let composite: MTLComputePipelineState
    private let preview: MTLRenderPipelineState
    private var textureCache: CVMetalTextureCache?
    private var previewSet: TextureSet?
    private let inFlight = DispatchSemaphore(value: 1)

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let lib = device.makeDefaultLibrary() else { return nil }
        self.device = device
        self.queue = queue

        func cp(_ name: String) -> MTLComputePipelineState? {
            guard let f = lib.makeFunction(name: name) else { return nil }
            return try? device.makeComputePipelineState(function: f)
        }
        guard let a = cp("prepKernel"), let b = cp("downKernel"), let c = cp("scaleKernel"),
              let d = cp("upAddKernel"), let e = cp("halMaskKernel"), let f = cp("compositeKernel") else { return nil }
        prep = a; down = b; scale = c; upAdd = d; halMask = e; composite = f

        let rp = MTLRenderPipelineDescriptor()
        rp.vertexFunction = lib.makeFunction(name: "previewVertex")
        rp.fragmentFunction = lib.makeFunction(name: "previewFragment")
        rp.colorAttachments[0].pixelFormat = .bgra8Unorm
        guard let pr = try? device.makeRenderPipelineState(descriptor: rp) else { return nil }
        preview = pr

        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
    }

    // MARK: - 通用编码

    private func dispatch(_ enc: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState, _ tex: MTLTexture) {
        enc.setComputePipelineState(pso)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let groups = MTLSize(width: (tex.width + 15) / 16, height: (tex.height + 15) / 16, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
    }

    /// 金字塔：base → downs → 逐层带权上采样，结果在 ups[0]
    private func pyramid(_ enc: MTLComputeCommandEncoder, base: MTLTexture, downs: [MTLTexture], ups: [MTLTexture], weights: [Float]) {
        var src = base
        for d in downs {
            enc.setTexture(src, index: 0)
            enc.setTexture(d, index: 1)
            dispatch(enc, down, d)
            src = d
        }
        let n = downs.count
        var w = weights[n - 1]
        enc.setTexture(downs[n - 1], index: 0)
        enc.setTexture(ups[n - 1], index: 1)
        enc.setBytes(&w, length: MemoryLayout<Float>.size, index: 0)
        dispatch(enc, scale, ups[n - 1])
        for i in stride(from: n - 2, through: 0, by: -1) {
            var wi = weights[i]
            enc.setTexture(ups[i + 1], index: 0)
            enc.setTexture(downs[i], index: 1)
            enc.setTexture(ups[i], index: 2)
            enc.setBytes(&wi, length: MemoryLayout<Float>.size, index: 0)
            dispatch(enc, upAdd, ups[i])
        }
    }

    private func encodeEffect(_ cmd: MTLCommandBuffer, input: MTLTexture, set: TextureSet, params: FXParams) {
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        var p = params
        let plen = MemoryLayout<FXParams>.stride

        enc.setTexture(input, index: 0)
        enc.setTexture(set.lin, index: 1)
        enc.setBytes(&p, length: plen, index: 0)
        dispatch(enc, prep, set.lin)

        pyramid(enc, base: set.lin, downs: set.bloomDowns, ups: set.bloomUps, weights: params.bloomLevelWeights)

        enc.setTexture(set.bloomDowns[0], index: 0)
        enc.setTexture(set.halMask, index: 1)
        enc.setBytes(&p, length: plen, index: 0)
        dispatch(enc, halMask, set.halMask)
        pyramid(enc, base: set.halMask, downs: set.haloDowns, ups: set.haloUps, weights: [0.15, 0.35, 0.35, 0.15])

        enc.setTexture(set.lin, index: 0)
        enc.setTexture(set.bloomUps[0], index: 1)
        enc.setTexture(set.bloomDowns[1], index: 2)
        enc.setTexture(set.haloUps[0], index: 3)
        enc.setTexture(set.output, index: 4)
        enc.setBytes(&p, length: plen, index: 0)
        dispatch(enc, composite, set.output)
        enc.endEncoding()
    }

    // MARK: - 实时预览（相机线程调用）

    func renderPreview(pixelBuffer: CVPixelBuffer, params: FXParams, layer: CAMetalLayer) {
        guard let cache = textureCache else { return }
        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        var cvTex: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, pixelBuffer, nil, .bgra8Unorm, w, h, 0, &cvTex)
        guard let cvTexture = cvTex, let input = CVMetalTextureGetTexture(cvTexture) else { return }

        inFlight.wait()
        if previewSet == nil || previewSet!.width != w || previewSet!.height != h {
            previewSet = TextureSet(device: device, width: w, height: h, sharedOutput: false)
        }
        guard let set = previewSet,
              let cmd = queue.makeCommandBuffer(),
              let drawable = layer.nextDrawable() else { inFlight.signal(); return }

        encodeEffect(cmd, input: input, set: set, params: params)

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = drawable.texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        if let r = cmd.makeRenderCommandEncoder(descriptor: pass) {
            let ds = layer.drawableSize
            let imageAspect = Float(w) / Float(h)
            let viewAspect = Float(ds.width) / Float(max(ds.height, 1))
            var s = SIMD2<Float>(1, 1)
            if imageAspect > viewAspect { s.y = viewAspect / imageAspect } else { s.x = imageAspect / viewAspect }
            r.setRenderPipelineState(preview)
            r.setVertexBytes(&s, length: MemoryLayout<SIMD2<Float>>.size, index: 0)
            r.setFragmentTexture(set.output, index: 0)
            r.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            r.endEncoding()
        }
        cmd.present(drawable)
        let sem = inFlight
        cmd.addCompletedHandler { _ in
            _ = cvTexture
            sem.signal()
        }
        cmd.commit()
    }

    // MARK: - 拍照全分辨率渲染

    /// 输入为 BGRA 8 位位图；输出 sRGB 的 CGImage
    func processPhoto(bgraBytes: UnsafeRawPointer, width: Int, height: Int, bytesPerRow: Int, params: FXParams) -> CGImage? {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let input = device.makeTexture(descriptor: d) else { return nil }
        input.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: bgraBytes, bytesPerRow: bytesPerRow)

        guard let set = TextureSet(device: device, width: width, height: height, sharedOutput: true),
              let cmd = queue.makeCommandBuffer() else { return nil }
        encodeEffect(cmd, input: input, set: set, params: params)
        cmd.commit()
        cmd.waitUntilCompleted()

        let rowBytes = width * 4
        var data = Data(count: rowBytes * height)
        data.withUnsafeMutableBytes { buf in
            if let base = buf.baseAddress {
                set.output.getBytes(base, bytesPerRow: rowBytes, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
            }
        }
        guard let provider = CGDataProvider(data: data as CFData),
              let cs = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: rowBytes,
                       space: cs, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}
