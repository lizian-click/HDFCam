import Foundation
import Metal

/// 预烘焙胶片颗粒纹理（512×512，可平铺）
/// 方法：非均匀布尔模型（Newson et al. 2017 的简化版）——
/// 在 2× 超采样网格上撒对数正态半径的随机圆盘（卤化银颗粒），降采样后做高通并归一化。
/// R 通道 = 亮度颗粒，G/B = 两路独立的色度颗粒。
enum GrainTexture {
    static let size = 512

    static func make(device: MTLDevice) -> MTLTexture? {
        let n = size
        let r = field(seed: 0x9E3779B9)
        let g = field(seed: 0x85EBCA6B)
        let b = field(seed: 0xC2B2AE35)

        var pixels = [Float16](repeating: 0, count: n * n * 4)
        for i in 0..<(n * n) {
            pixels[i * 4 + 0] = Float16(r[i])
            pixels[i * 4 + 1] = Float16(g[i])
            pixels[i * 4 + 2] = Float16(b[i])
            pixels[i * 4 + 3] = 1
        }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: n, height: n, mipmapped: false)
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: d) else { return nil }
        pixels.withUnsafeBytes { buf in
            if let base = buf.baseAddress {
                tex.replace(region: MTLRegionMake2D(0, 0, n, n), mipmapLevel: 0, withBytes: base, bytesPerRow: n * 4 * MemoryLayout<Float16>.size)
            }
        }
        return tex
    }

    /// 简单可复现的随机数（xorshift）
    private struct RNG {
        var s: UInt64
        mutating func next() -> UInt64 { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s }
        mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }
        mutating func gauss() -> Float {
            let u1 = max(unit(), 1e-7), u2 = unit()
            return sqrt(-2 * log(u1)) * cos(2 * Float.pi * u2)
        }
    }

    private static func field(seed: UInt64) -> [Float] {
        let n = size
        let S = n * 2                         // 超采样网格
        var cover = [UInt8](repeating: 0, count: S * S)
        var rng = RNG(s: seed | 1)

        let meanR: Float = 1.6                // 超采样网格上的平均半径 ≈ 输出 0.8 纹素
        let sigmaLn: Float = 0.25
        let meanArea = Float.pi * meanR * meanR * exp(2 * sigmaLn * sigmaLn)
        let coverage: Float = 0.45
        let count = Int(-log(1 - coverage) / meanArea * Float(S * S))

        for _ in 0..<count {
            let cx = rng.unit() * Float(S)
            let cy = rng.unit() * Float(S)
            let rad = meanR * exp(sigmaLn * rng.gauss() - 0.5 * sigmaLn * sigmaLn)
            let r2 = rad * rad
            let x0 = Int(floor(cx - rad)), x1 = Int(ceil(cx + rad))
            let y0 = Int(floor(cy - rad)), y1 = Int(ceil(cy + rad))
            for y in y0...y1 {
                let dy = Float(y) + 0.5 - cy
                let yy = ((y % S) + S) % S
                for x in x0...x1 {
                    let dx = Float(x) + 0.5 - cx
                    if dx * dx + dy * dy <= r2 {
                        cover[yy * S + ((x % S) + S) % S] = 1
                    }
                }
            }
        }

        // 2× 降采样
        var v = [Float](repeating: 0, count: n * n)
        for y in 0..<n {
            for x in 0..<n {
                let a = Int(cover[(2 * y) * S + 2 * x]) + Int(cover[(2 * y) * S + 2 * x + 1])
                    + Int(cover[(2 * y + 1) * S + 2 * x]) + Int(cover[(2 * y + 1) * S + 2 * x + 1])
                v[y * n + x] = Float(a) * 0.25
            }
        }

        // 高通：减去半径 4 的平铺盒式模糊（去掉低频「云斑」）
        let R = 4
        var tmp = [Float](repeating: 0, count: n * n)
        for y in 0..<n {
            for x in 0..<n {
                var s: Float = 0
                for k in -R...R { s += v[y * n + ((x + k + n) % n)] }
                tmp[y * n + x] = s / Float(2 * R + 1)
            }
        }
        var blur = [Float](repeating: 0, count: n * n)
        for y in 0..<n {
            for x in 0..<n {
                var s: Float = 0
                for k in -R...R { s += tmp[((y + k + n) % n) * n + x] }
                blur[y * n + x] = s / Float(2 * R + 1)
            }
        }
        var mean: Float = 0
        for i in 0..<(n * n) { v[i] -= blur[i]; mean += v[i] }
        mean /= Float(n * n)
        var varSum: Float = 0
        for i in 0..<(n * n) { v[i] -= mean; varSum += v[i] * v[i] }
        let std = sqrt(varSum / Float(n * n))
        if std > 1e-6 { for i in 0..<(n * n) { v[i] /= std } }
        return v
    }
}
