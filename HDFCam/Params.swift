import Foundation
import simd

// MARK: - GPU 端结构（与 Shaders.metal 一一对应，全部 Float，顺序不能改）

struct GPUParams {
    var exposure: Float = 0
    var clipLevel: Float = 1
    var linearInput: Float = 0
    var eArea: Float = 0.5
    var ePtDay: Float = 3
    var ePtNight: Float = 5.5
    var fTotal: Float = 0
    var veil: Float = 0
    var veilR: Float = 1, veilG: Float = 1, veilB: Float = 1
    var soften: Float = 0
    var softT: Float = 0
    var gritScale: Float = 1
    var curveAmt: Float = 0
    var wbWarm: Float = 0
    var splitAmt: Float = 0
    var hslAmt: Float = 0
    var grainAmt: Float = 0
    var grainTexPerPx: Float = 1
    var grainChroma: Float = 0
    var grainOffX: Float = 0, grainOffY: Float = 0, grainRot: Float = 0
    var halThresh: Float = 1
    var halCore: Float = 0.8
    var shoulderKnee: Float = 0.8
    var bw: Float = 0
    var shortSide: Float = 1000
    var pad0: Float = 0, pad1: Float = 0, pad2: Float = 0
}

struct LobeParam {
    var r: Float = 0, g: Float = 0, b: Float = 0
    var desat: Float = 0
    var grit: Float = 0
    var mode: Float = 0
    var active: Float = 0
    var pad: Float = 0
}

/// 一个高斯光瓣：σ（像素）+ 已乘散射能量的色调
struct Lobe {
    var sigmaPx: Float
    var rgb: SIMD3<Float>
    var desat: Float
    var grit: Float
    var mode: Float = 0
}

// MARK: - 胶片片种（颗粒参数：峰值 σ(EV)、颗粒尺寸(%短边)、彩色颗粒比例）

enum FilmStock: String, CaseIterable, Identifiable {
    case none, ektar100, portra160, pro400h, portra400, portra800, gold200, cinestill800t, hp5
    var id: String { rawValue }

    var name: String {
        switch self {
        case .none: return "无"
        case .ektar100: return "Ektar 100"
        case .portra160: return "Portra 160"
        case .pro400h: return "Pro 400H"
        case .portra400: return "Portra 400"
        case .portra800: return "Portra 800"
        case .gold200: return "Gold 200"
        case .cinestill800t: return "CineStill 800T"
        case .hp5: return "HP5 黑白"
        }
    }

    /// (峰值 σ EV, 尺寸 %H, 色度比例)
    var grain: (Float, Float, Float) {
        switch self {
        case .none: return (0, 0.035, 0)
        case .ektar100: return (0.04, 0.025, 0.30)
        case .portra160: return (0.05, 0.030, 0.25)
        case .pro400h: return (0.065, 0.033, 0.30)
        case .portra400: return (0.07, 0.035, 0.30)
        case .portra800: return (0.09, 0.040, 0.40)
        case .gold200: return (0.08, 0.040, 0.45)
        case .cinestill800t: return (0.10, 0.045, 0.50)
        case .hp5: return (0.12, 0.050, 0)
        }
    }

    var halation: Bool { self == .cinestill800t }
    var isBW: Bool { self == .hp5 }
}

// MARK: - 用户可见的「风格」

struct Look: Equatable {
    var hdf: Float = 0          // HDF 强度（1.0 ≈ GR f/2.8，0.4 ≈ f/8）
    var satin: Float = 0        // 黑丝绒档位 0 关 / 1=1/8 / 2=1/4 / 3=1/2 / 4=1
    var pearl: Float = 0        // 珠光档位
    var glimmer: Float = 0      // 微光档位
    var bpm: Float = 0          // 黑柔档位（对照）
    var stock: FilmStock = .none
    var grain: Float = 1        // 颗粒强度倍数
    var tone: Float = 0         // 胶片曲线 + 分离色调
    var hsl: Float = 0          // 人像色彩
    var warm: Float = 0         // 白平衡偏暖（-1 冷 … 1 暖）
    var pointBoost: Float = 1   // 点光源能量倍数
    var exposure: Float = 0     // 数字曝光（档）
    var knee: Float = 0.8       // 高光肩部起点（越低高光越柔）

    /// 档位插值：g ∈ [0,4]，values = [1/8, 1/4, 1/2, 1]
    private static func grade(_ g: Float, _ v: [Float]) -> Float {
        let t = [0] + v
        let x = min(max(g, 0), 4)
        let i = min(Int(x), 3)
        let f = x - Float(i)
        return t[i] + (t[i + 1] - t[i]) * f
    }

    struct Built {
        var params: GPUParams
        var lobes: [Lobe]       // 最多 6 个光学光瓣
        var halation: Lobe?     // 胶片红晕（单独一个槽）
    }

    func build(shortSide H: Float, clipLevel: Float, linearInput: Bool, seed: Float, freezeGrain: Bool) -> Built {
        var p = GPUParams()
        let pct: (Float) -> Float = { $0 / 100 * H }   // %H → 像素

        var raw: [Lobe] = []
        var fSum: Float = 0
        var veil = SIMD3<Float>(0, 0, 0)
        var soften: Float = 0
        var softSigmaPct: Float = 0

        func add(f: Float, _ shape: [(Float, Float)], tint: SIMD3<Float>, desat: Float = 0, grit: Float = 0, sigmaScale: Float = 1) {
            guard f > 0.0005 else { return }
            fSum += f
            for (w, s) in shape {
                raw.append(Lobe(sigmaPx: pct(s * sigmaScale), rgb: tint * (w * f), desat: desat, grit: grit))
            }
        }

        // HDF：锐利核心 + 紧贴的短光瓣 + 极弱拖尾，光晕保留光源原色
        add(f: 0.12 * hdf, [(0.55, 0.35), (0.35, 1.1), (0.10, 3.5)], tint: SIMD3(1, 1, 1))

        // 黑丝绒 Black Satin：收敛的暖调光晕 + 高光颗粒感 + 皮肤微柔化
        let fs = Look.grade(satin, [0.035, 0.05, 0.075, 0.11])
        let satinTint = SIMD3<Float>(1, 0.95, 0.88)
        add(f: fs, [(0.6, 0.6), (0.4, 2.0)], tint: satinTint, grit: 1)
        veil += satinTint * (0.1 * fs)
        if satin > 0.01 { soften = max(soften, Look.grade(satin, [0.25, 0.35, 0.5, 0.6])); softSigmaPct = max(softSigmaPct, 0.06) }

        // 珠光 Pearlescent：更宽的奶油色光晕，散射光去饱和，粉彩感
        let fp = Look.grade(pearl, [0.05, 0.09, 0.15, 0.25])
        let pearlTint = SIMD3<Float>(1, 0.97, 0.98)
        add(f: fp, [(0.4, 1.0), (0.4, 4.0), (0.2, 10)], tint: pearlTint, desat: 0.35)
        veil += pearlTint * (0.25 * fp)
        if pearl > 0.01 { soften = max(soften, Look.grade(pearl, [0.3, 0.45, 0.6, 0.7])); softSigmaPct = max(softSigmaPct, 0.08) }

        // 微光 Glimmerglass：光源周围干净的小光晕，几乎不降对比
        let fg = Look.grade(glimmer, [0.035, 0.05, 0.08, 0.12])
        add(f: fg, [(0.65, 0.5), (0.35, 1.8)], tint: SIMD3(1, 1, 1), desat: 0.15)
        veil += SIMD3<Float>(1, 1, 1) * (0.3 * fg)

        // 黑柔 Black Pro-Mist（对照）：宽光瓣 + 长拖尾，逆光会发灰
        let fb = Look.grade(bpm, [0.06, 0.10, 0.16, 0.24])
        let bpmTint = SIMD3<Float>(1, 0.97, 0.92)
        add(f: fb, [(0.35, 0.8), (0.35, 3), (0.30, 12)], tint: bpmTint, sigmaScale: pow(1.15, max(bpm - 1, 0)))
        veil += bpmTint * (0.15 * fb)

        // 总散射能量上限 0.25，保证能量守恒、不发灰
        let scale: Float = fSum > 0.25 ? 0.25 / fSum : 1
        for i in raw.indices { raw[i].rgb *= scale }
        p.fTotal = fSum * scale
        veil *= scale
        p.veil = 1
        p.veilR = veil.x; p.veilG = veil.y; p.veilB = veil.z

        // 胶片
        let (gSigma, gSize, gChroma) = stock.grain
        if stock != .none { soften = min(soften + 0.25, 0.75); softSigmaPct = max(softSigmaPct, gSize * 0.5) }
        p.soften = soften
        let softPx = pct(softSigmaPct)
        p.softT = min(max((softPx - 0.9) / 0.9, 0), 1)
        p.gritScale = max(pct(0.05), 0.5)

        p.curveAmt = tone
        p.splitAmt = tone
        p.hslAmt = hsl
        p.wbWarm = warm
        p.bw = stock.isBW ? 1 : 0
        p.shoulderKnee = knee

        p.grainAmt = gSigma * grain
        p.grainTexPerPx = 0.8 / max(pct(gSize), 0.05)
        p.grainChroma = gChroma
        if freezeGrain {
            p.grainOffX = 0; p.grainOffY = 0; p.grainRot = 0
        } else {
            p.grainOffX = Float.random(in: 0..<512)
            p.grainOffY = Float.random(in: 0..<512)
            p.grainRot = Bool.random() ? 1 : 0
        }

        // 高光重建：RAW 本身已多 2 档左右余量，扩展相应减少
        p.clipLevel = clipLevel
        p.linearInput = linearInput ? 1 : 0
        p.eArea = linearInput ? 0.25 : 0.5
        p.ePtDay = (linearInput ? 2.0 : 3.0) * pointBoost
        p.ePtNight = (linearInput ? 4.0 : 5.5) * pointBoost
        p.exposure = exposure
        p.shortSide = H

        var halo: Lobe? = nil
        if stock.halation {
            halo = Lobe(sigmaPx: pct(0.6), rgb: SIMD3(1, 0.10, 0.033) * 0.6, desat: 0, grit: 0, mode: 2)
            p.halThresh = 1.0
            p.halCore = 0.8
        }

        return Built(params: p, lobes: Look.merge(raw, maxCount: 6), halation: halo)
    }

    /// 把 σ 相近的光瓣合并，控制 GPU 开销
    static func merge(_ input: [Lobe], maxCount: Int) -> [Lobe] {
        var l = input.sorted { $0.sigmaPx < $1.sigmaPx }
        func weight(_ x: Lobe) -> Float { max(x.rgb.x + x.rgb.y + x.rgb.z, 1e-6) }
        func combine(_ a: Lobe, _ b: Lobe) -> Lobe {
            let wa = weight(a), wb = weight(b), w = wa + wb
            let s = exp((log(a.sigmaPx) * wa + log(b.sigmaPx) * wb) / w)
            return Lobe(sigmaPx: s, rgb: a.rgb + b.rgb,
                        desat: (a.desat * wa + b.desat * wb) / w,
                        grit: (a.grit * wa + b.grit * wb) / w)
        }
        var i = 0
        while i + 1 < l.count {
            if l[i + 1].sigmaPx / l[i].sigmaPx < 1.3 {
                l[i] = combine(l[i], l[i + 1]); l.remove(at: i + 1)
            } else { i += 1 }
        }
        while l.count > maxCount {
            var best = 0
            var bestRatio = Float.greatestFiniteMagnitude
            for j in 0..<(l.count - 1) {
                let r = l[j + 1].sigmaPx / l[j].sigmaPx
                if r < bestRatio { bestRatio = r; best = j }
            }
            l[best] = combine(l[best], l[best + 1]); l.remove(at: best + 1)
        }
        return l
    }
}

// MARK: - 预设

struct Preset: Identifiable {
    let id: String
    let look: Look
}

enum Presets {
    static let all: [Preset] = [
        Preset(id: "原片", look: Look()),
        Preset(id: "GR HDF", look: Look(hdf: 1.0, stock: .portra400, grain: 0.7, tone: 0.4)),
        Preset(id: "汉服逆光", look: Look(hdf: 0.5, satin: 2, stock: .portra160, grain: 1, tone: 1, hsl: 1, warm: 1)),
        Preset(id: "汉服珠光", look: Look(pearl: 1, stock: .pro400h, grain: 1, tone: 1, hsl: 1, warm: 0.5, knee: 0.7)),
        Preset(id: "夜景摆摊", look: Look(hdf: 1.0, glimmer: 1, stock: .portra800, grain: 1, tone: 0.6, warm: 1)),
        Preset(id: "白天外景", look: Look(hdf: 0.4, stock: .gold200, grain: 1, tone: 0.8, hsl: 0.5)),
        Preset(id: "CineStill", look: Look(hdf: 0.3, stock: .cinestill800t, grain: 1, tone: 0.7, warm: -1)),
        Preset(id: "HP5 黑白", look: Look(hdf: 0.6, stock: .hp5, grain: 1, tone: 0.8)),
        Preset(id: "黑柔对照", look: Look(bpm: 2)),
    ]
}
