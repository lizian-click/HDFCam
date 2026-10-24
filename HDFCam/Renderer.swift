import Metal
import QuartzCore
import CoreVideo
import CoreGraphics

/// 某个光瓣使用的 mip 级别与纹理
final class BlurSlot {
    let level: Int
    let out: MTLTexture
    let tmp: MTLTexture
    init(level: Int, out: MTLTexture, tmp: MTLTexture) { self.level = level; self.out = out; self.tmp = tmp }
}

/// 一套按尺寸分配的中间纹理
final class TextureSet {
    let width: Int
    let height: Int
    let device: MTLDevice
    let linRaw: MTLTexture
    let aux: MTLTexture
    let linHat: MTLTexture
    let auxMips: [MTLTexture]   // 第 k 级在下标 k-1
    let mips: [MTLTexture]
    let output: MTLTexture
    var auxSlot: BlurSlot?
    var slots: [BlurSlot?] = Array(repeating: nil, count: 7)

    var shortSide: Float { Float(min(width, height)) }

    static func makeTex(_ device: MTLDevice, _ w: Int, _ h: Int, _ fmt: MTLPixelFormat = .rgba16Float, shared: Bool = false) -> MTLTexture? {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: max(1, w), height: max(1, h), mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = shared ? .shared : .private
        return device.makeTexture(descriptor: d)
    }

    init?(device: MTLDevice, width: Int, height: Int, sharedOutput: Bool) {
        self.device = device
        self.width = width
        self.height = height
        guard let a = TextureSet.makeTex(device, width, height),
              let b = TextureSet.makeTex(device, width, height),
              let c = TextureSet.makeTex(device, width, height),
              let o = TextureSet.makeTex(device, width, height, .rgba8Unorm, shared: sharedOutput) else { return nil }
        linRaw = a; aux = b; linHat = c; output = o

        var am: [MTLTexture] = []
        var lm: [MTLTexture] = []
        var w = width, h = height
        while max(w, h) > 4 && lm.count < 12 {
            w = max(1, w / 2); h = max(1, h / 2)
            guard let t1 = TextureSet.makeTex(device, w, h), let t2 = TextureSet.makeTex(device, w, h) else { return nil }
            am.append(t1); lm.append(t2)
        }
        auxMips = am
        mips = lm
    }

    var levels: Int { mips.count }

    func slot(_ existing: BlurSlot?, level: Int) -> BlurSlot? {
        if let s = existing, s.level == level { return s }
        let src = mips[level - 1]
        guard let o = TextureSet.makeTex(device, src.width, src.height),
              let t = TextureSet.makeTex(device, src.width, src.height) else { return nil }
        return BlurSlot(level: level, out: o, tmp: t)
    }
}

struct BlurParam {
    var sigma: Float
    var radius: Float
    var dx: Float
    var dy: Float
}

final class Renderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    private let prepA: MTLComputePipelineState
    private let prepB: MTLComputePipelineState
    private let down: MTLComputePipelineState
    private let blur: MTLComputePipelineState
    private let composite: MTLComputePipelineState
    private let preview: MTLRenderPipelineState
    private let grain: MTLTexture
    private let dummy: MTLTexture
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
        guard let a = cp("prepAKernel"), let b = cp("prepBKernel"), let c = cp("downBoxKernel"),
              let d = cp("blurKernel"), let e = cp("compositeKernel") else { return nil }
        prepA = a; prepB = b; down = c; blur = d; composite = e

        let rp = MTLRenderPipelineDescriptor()
        rp.vertexFunction = lib.makeFunction(name: "previewVertex")
        rp.fragmentFunction = lib.makeFunction(name: "previewFragment")
        rp.colorAttachments[0].pixelFormat = .bgra8Unorm
        guard let pr = try? device.makeRenderPipelineState(descriptor: rp) else { return nil }
        preview = pr

        guard let g = GrainTexture.make(device: device) else { return nil }
        grain = g

        let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        dd.usage = [.shaderRead]
        dd.storageMode = .shared
        guard let dt = device.makeTexture(descriptor: dd) else { return nil }
        let zero = [Float16](repeating: 0, count: 4)
        zero.withUnsafeBytes { buf in
            if let base = buf.baseAddress { dt.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: base, bytesPerRow: 8) }
        }
        dummy = dt

        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
    }

    // MARK: - 编码工具

    private func dispatch(_ enc: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState, _ tex: MTLTexture) {
        enc.setComputePipelineState(pso)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let groups = MTLSize(width: (tex.width + 15) / 16, height: (tex.height + 15) / 16, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
    }

    private func buildMips(_ enc: MTLComputeCommandEncoder, base: MTLTexture, mips: [MTLTexture]) {
        var src = base
        for m in mips {
            enc.setTexture(src, index: 0)
            enc.setTexture(m, index: 1)
            dispatch(enc, down, m)
            src = m
        }
    }

    /// 选一个 mip 级别，使该级上的 σ 约 1.5–3 像素，再做可分离高斯
    private func levelFor(sigmaPx: Float, maxLevel: Int) -> Int {
        let n = Int(ceil(log2(max(sigmaPx, 1) / 3.0)))
        return min(max(n, 1), maxLevel)
    }

    private func gaussian(_ enc: MTLComputeCommandEncoder, src: MTLTexture, slot: BlurSlot, sigmaPx: Float) {
        let scale = Float(1 << slot.level)
        let sl = sigmaPx / scale
        let s = sqrt(max(sl * sl - 0.3, 0.16))     // 扣掉盒式降采样自带的模糊
        let r = min(ceil(3 * s), 24)
        var bh = BlurParam(sigma: s, radius: r, dx: 1, dy: 0)
        enc.setTexture(src, index: 0)
        enc.setTexture(slot.tmp, index: 1)
        enc.setBytes(&bh, length: MemoryLayout<BlurParam>.stride, index: 0)
        dispatch(enc, blur, slot.tmp)
        var bv = BlurParam(sigma: s, radius: r, dx: 0, dy: 1)
        enc.setTexture(slot.tmp, index: 0)
        enc.setTexture(slot.out, index: 1)
        enc.setBytes(&bv, length: MemoryLayout<BlurParam>.stride, index: 0)
        dispatch(enc, blur, slot.out)
    }

    private func encodeEffect(_ cmd: MTLCommandBuffer, input: MTLTexture, set: TextureSet, built: Look.Built) {
        guard let enc = cmd.makeComputeCommandEncoder(), set.levels >= 2 else { return }
        var p = built.params
        let plen = MemoryLayout<GPUParams>.stride
        let H = set.shortSide

        // 1. 线性化
        enc.setTexture(input, index: 0)
        enc.setTexture(set.linRaw, index: 1)
        enc.setTexture(set.aux, index: 2)
        enc.setBytes(&p, length: plen, index: 0)
        dispatch(enc, prepA, set.linRaw)

        // 2. 辅助图的 mip 与 1.5%H 模糊（点光源判定 + 色相恢复）
        buildMips(enc, base: set.aux, mips: set.auxMips)
        let auxSigma = 0.015 * H
        let auxLevel = levelFor(sigmaPx: auxSigma, maxLevel: set.levels)
        if let s = set.slot(set.auxSlot, level: auxLevel) {
            set.auxSlot = s
            gaussian(enc, src: set.auxMips[auxLevel - 1], slot: s, sigmaPx: auxSigma)
        }
        guard let auxBlur = set.auxSlot?.out else { enc.endEncoding(); return }

        // 3. 高光重建
        enc.setTexture(set.linRaw, index: 0)
        enc.setTexture(auxBlur, index: 1)
        enc.setTexture(set.auxMips[set.levels - 1], index: 2)
        enc.setTexture(set.linHat, index: 3)
        enc.setBytes(&p, length: plen, index: 0)
        dispatch(enc, prepB, set.linHat)

        // 4. 重建后的 mip 与各光瓣
        buildMips(enc, base: set.linHat, mips: set.mips)

        var lobeParams = [LobeParam](repeating: LobeParam(), count: 7)
        var lobeTex = [MTLTexture](repeating: dummy, count: 7)
        var all: [(Int, Lobe)] = built.lobes.prefix(6).enumerated().map { ($0.offset, $0.element) }
        if let h = built.halation { all.append((6, h)) }
        for (i, lobe) in all {
            let lv = levelFor(sigmaPx: lobe.sigmaPx, maxLevel: set.levels)
            guard let s = set.slot(set.slots[i], level: lv) else { continue }
            set.slots[i] = s
            gaussian(enc, src: set.mips[lv - 1], slot: s, sigmaPx: lobe.sigmaPx)
            lobeTex[i] = s.out
            lobeParams[i] = LobeParam(r: lobe.rgb.x, g: lobe.rgb.y, b: lobe.rgb.z,
                                      desat: lobe.desat, grit: lobe.grit, mode: lobe.mode, active: 1, pad: 0)
        }

        // 5. 合成
        enc.setTexture(set.linHat, index: 0)
        enc.setTexture(set.mips[0], index: 1)
        enc.setTexture(set.mips[1], index: 2)
        enc.setTexture(set.mips[set.levels - 1], index: 3)
        enc.setTexture(grain, index: 4)
        enc.setTextures(lobeTex, range: 5..<12)
        enc.setTexture(set.output, index: 12)
        enc.setBytes(&p, length: plen, index: 0)
        enc.setBytes(&lobeParams, length: MemoryLayout<LobeParam>.stride * 7, index: 1)
        dispatch(enc, composite, set.output)
        _ = H
        enc.endEncoding()
    }

    // MARK: - 实时预览（相机线程调用）

    func renderPreview(pixelBuffer: CVPixelBuffer, look: Look, layer: CAMetalLayer) {
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

        let built = look.build(shortSide: set.shortSide, clipLevel: 1, linearInput: false,
                               seed: Float.random(in: 0...1), freezeGrain: false)
        encodeEffect(cmd, input: input, set: set, built: built)

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

    func makeInputBGRA(bytes: UnsafeRawPointer, width: Int, height: Int, bytesPerRow: Int) -> MTLTexture? {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d) else { return nil }
        t.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: bytes, bytesPerRow: bytesPerRow)
        return t
    }

    func makeInputHalf(bytes: UnsafeRawPointer, width: Int, height: Int, bytesPerRow: Int) -> MTLTexture? {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d) else { return nil }
        t.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: bytes, bytesPerRow: bytesPerRow)
        return t
    }

    /// 输出 sRGB 的 CGImage
    func processPhoto(input: MTLTexture, look: Look, linearInput: Bool, clipLevel: Float) -> CGImage? {
        let width = input.width, height = input.height
        guard let set = TextureSet(device: device, width: width, height: height, sharedOutput: true),
              let cmd = queue.makeCommandBuffer() else { return nil }
        let built = look.build(shortSide: set.shortSide, clipLevel: clipLevel, linearInput: linearInput,
                               seed: 0, freezeGrain: true)
        encodeEffect(cmd, input: input, set: set, built: built)
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
