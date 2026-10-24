import Foundation

/// 与 Shaders.metal 中的 FXParams 一一对应，全部为 Float，顺序不能改
struct FXParams: Equatable {
    var exposure: Float = 0
    var hlThreshold: Float = 0.85
    var hlStops: Float = 3
    var bloom: Float = 0.07
    var wCore: Float = 0.45
    var wMid: Float = 0.35
    var wTail: Float = 0.20
    var warmEdge: Float = 0.3
    var halation: Float = 0
    var halThreshold: Float = 0.75
    var halCore: Float = 0.8
    var softSkin: Float = 0
    var shoulder: Float = 0.4
    var warmth: Float = 0
    var saturation: Float = 1
    var grain: Float = 0.15
    var grainSize: Float = 1.3
    var grainShadow: Float = 0.3
    var seed: Float = 0
    var pad: Float = 0

    /// HDF 金字塔 6 层权重（归一化）
    var bloomLevelWeights: [Float] {
        let w: [Float] = [wCore * 0.55, wCore * 0.45, wMid * 0.5, wMid * 0.5, wTail * 0.45, wTail * 0.55]
        let s = w.reduce(0, +)
        return s > 0.0001 ? w.map { $0 / s } : Array(repeating: 1.0 / 6.0, count: 6)
    }
}

struct Preset: Identifiable {
    let id: String
    let fx: FXParams
}

enum Presets {
    static let all: [Preset] = [
        Preset(id: "原片", fx: {
            var p = FXParams()
            p.bloom = 0; p.halation = 0; p.softSkin = 0; p.grain = 0; p.shoulder = 0.3; p.warmEdge = 0
            return p
        }()),
        Preset(id: "HDF", fx: FXParams()),
        Preset(id: "黑柔1/8", fx: {
            var p = FXParams()
            p.bloom = 0.05; p.wCore = 0.2; p.wMid = 0.35; p.wTail = 0.45
            p.hlStops = 2; p.warmEdge = 0; p.shoulder = 0.5; p.grain = 0.1
            return p
        }()),
        Preset(id: "汉服柔光", fx: {
            var p = FXParams()
            p.bloom = 0.06; p.wCore = 0.3; p.wMid = 0.45; p.wTail = 0.25
            p.hlStops = 2.5; p.warmEdge = 0.2; p.softSkin = 0.5
            p.warmth = 0.1; p.saturation = 0.95; p.grain = 0.1; p.shoulder = 0.5
            return p
        }()),
        Preset(id: "18胶片", fx: {
            var p = FXParams()
            p.bloom = 0.02; p.halation = 0.08; p.shoulder = 0.7
            p.grain = 0.4; p.grainShadow = 0.8; p.saturation = 0.9; p.warmEdge = 0
            return p
        }()),
        Preset(id: "CineStill夜", fx: {
            var p = FXParams()
            p.bloom = 0.03; p.halation = 0.12; p.halThreshold = 0.7; p.halCore = 0.8
            p.hlStops = 3.5; p.warmth = -0.4; p.grain = 0.35; p.shoulder = 0.6; p.warmEdge = 0
            return p
        }()),
    ]
}
