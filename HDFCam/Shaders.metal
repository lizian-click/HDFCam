//  Shaders.metal — 柔光机 着色器管线
//  线性光 → 高光能量重建 → 多尺度 HDF 扩散 → 胶片光晕 → 柔肤 → 色彩保持色调映射 → 颗粒
#include <metal_stdlib>
using namespace metal;

// 必须与 Swift 端 FXParams 字段顺序完全一致（全部 float）
struct FXParams {
    float exposure;      // 数字曝光（档）
    float hlThreshold;   // 高光重建起点（编码值 0-1）
    float hlStops;       // 高光扩展（档）
    float bloom;         // HDF 散射能量占比 a
    float wCore;
    float wMid;
    float wTail;
    float warmEdge;      // 光晕外缘偏暖
    float halation;      // 胶片红色光晕强度
    float halThreshold;
    float halCore;       // 核心抑制
    float softSkin;
    float shoulder;      // 高光压缩
    float warmth;        // 色温
    float saturation;
    float grain;
    float grainSize;
    float grainShadow;
    float seed;
    float pad;
};

constexpr sampler linSampler(coord::normalized, address::clamp_to_edge, filter::linear);

static inline float3 srgbToLinear(float3 c) {
    float3 lo = c / 12.92;
    float3 hi = pow((c + 0.055) / 1.055, 2.4);
    return select(hi, lo, c <= 0.04045);
}

static inline float3 linearToSrgb(float3 c) {
    c = clamp(c, 0.0, 1.0);
    float3 lo = c * 12.92;
    float3 hi = 1.055 * pow(c, 1.0 / 2.4) - 0.055;
    return select(hi, lo, c <= 0.0031308);
}

static inline float luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// ---------- 1. 线性化 + 高光能量重建 ----------
kernel void prepKernel(texture2d<float, access::read>  src [[texture(0)]],
                       texture2d<float, access::write> dst [[texture(1)]],
                       constant FXParams& p [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float3 enc = src.read(gid).rgb;
    float3 lin = srgbToLinear(enc);
    // 被裁剪的高光：编码值越接近 1，越可能是被截断的强光，按比例恢复能量
    float m = max(enc.r, max(enc.g, enc.b));
    float s = smoothstep(p.hlThreshold, 1.0, m);
    lin *= 1.0 + (exp2(p.hlStops) - 1.0) * s * s;
    lin *= exp2(p.exposure);
    dst.write(float4(lin, 1.0), gid);
}

// ---------- 2. Dual Kawase 降采样 ----------
kernel void downKernel(texture2d<float, access::sample> src [[texture(0)]],
                       texture2d<float, access::write>  dst [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]])
{
    uint w = dst.get_width(), h = dst.get_height();
    if (gid.x >= w || gid.y >= h) return;
    float2 uv = (float2(gid) + 0.5) / float2(w, h);
    float2 t = 1.0 / float2(src.get_width(), src.get_height());
    float4 sum = src.sample(linSampler, uv, level(0)) * 4.0;
    sum += src.sample(linSampler, uv - t, level(0));
    sum += src.sample(linSampler, uv + t, level(0));
    sum += src.sample(linSampler, uv + float2(t.x, -t.y), level(0));
    sum += src.sample(linSampler, uv - float2(t.x, -t.y), level(0));
    dst.write(sum / 8.0, gid);
}

// 最深一层：U = w * D
kernel void scaleKernel(texture2d<float, access::read>  src [[texture(0)]],
                        texture2d<float, access::write> dst [[texture(1)]],
                        constant float& weight [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    dst.write(src.read(gid) * weight, gid);
}

// ---------- 3. 帐篷上采样并按层权重累加：U_k = tent(U_k+1) + w_k * D_k ----------
kernel void upAddKernel(texture2d<float, access::sample> low  [[texture(0)]],
                        texture2d<float, access::read>   same [[texture(1)]],
                        texture2d<float, access::write>  dst  [[texture(2)]],
                        constant float& weight [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]])
{
    uint w = dst.get_width(), h = dst.get_height();
    if (gid.x >= w || gid.y >= h) return;
    float2 uv = (float2(gid) + 0.5) / float2(w, h);
    float2 hp = 0.5 / float2(low.get_width(), low.get_height());
    float4 sum = 0.0;
    sum += low.sample(linSampler, uv + float2(-2.0 * hp.x, 0.0), level(0));
    sum += low.sample(linSampler, uv + float2( 2.0 * hp.x, 0.0), level(0));
    sum += low.sample(linSampler, uv + float2(0.0,  2.0 * hp.y), level(0));
    sum += low.sample(linSampler, uv + float2(0.0, -2.0 * hp.y), level(0));
    sum += low.sample(linSampler, uv + float2(-hp.x,  hp.y), level(0)) * 2.0;
    sum += low.sample(linSampler, uv + float2( hp.x,  hp.y), level(0)) * 2.0;
    sum += low.sample(linSampler, uv + float2( hp.x, -hp.y), level(0)) * 2.0;
    sum += low.sample(linSampler, uv + float2(-hp.x, -hp.y), level(0)) * 2.0;
    dst.write(sum / 12.0 + same.read(gid) * weight, gid);
}

// ---------- 4. 胶片光晕的高光掩码（半分辨率） ----------
kernel void halMaskKernel(texture2d<float, access::read>  src [[texture(0)]],
                          texture2d<float, access::write> dst [[texture(1)]],
                          constant FXParams& p [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float Y = luma(src.read(gid).rgb);
    float m = smoothstep(p.halThreshold, p.halThreshold + 0.25, Y) * Y;
    dst.write(float4(m, m, m, 1.0), gid);
}

// ---------- 颗粒用的值噪声 ----------
static inline float hash21(float2 q) {
    q = fract(q * float2(123.34, 456.21));
    q += dot(q, q + 45.32);
    return fract(q.x * q.y);
}

static inline float valueNoise(float2 q) {
    float2 i = floor(q);
    float2 f = fract(q);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + float2(1.0, 0.0));
    float c = hash21(i + float2(0.0, 1.0));
    float d = hash21(i + float2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// ---------- 5. 合成 ----------
kernel void compositeKernel(texture2d<float, access::read>   lin   [[texture(0)]],
                            texture2d<float, access::sample> bloom [[texture(1)]],
                            texture2d<float, access::sample> d2    [[texture(2)]],
                            texture2d<float, access::sample> halo  [[texture(3)]],
                            texture2d<float, access::write>  outT  [[texture(4)]],
                            constant FXParams& p [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]])
{
    uint W = outT.get_width(), H = outT.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float2 uv = (float2(gid) + 0.5) / float2(W, H);

    float3 L = lin.read(gid).rgb;

    // [A] HDF：能量守恒的多尺度扩散 O = (1-a)L + a·Σw·B
    float3 B = bloom.sample(linSampler, uv, level(0)).rgb;
    float3 edgeTint = float3(1.0 + 0.25 * p.warmEdge, 1.0, 1.0 - 0.2 * p.warmEdge);
    float3 O = (1.0 - p.bloom) * L + p.bloom * B * edgeTint;

    // [C] 柔肤：四分之一分辨率亮度 + 边缘保护 + 肤色掩码，只动亮度
    if (p.softSkin > 0.001) {
        float3 S = d2.sample(linSampler, uv, level(0)).rgb;
        float3 se = linearToSrgb(S);
        float cb = -0.168736 * se.r - 0.331264 * se.g + 0.5 * se.b + 0.5;
        float cr =  0.5 * se.r - 0.418688 * se.g - 0.081312 * se.b + 0.5;
        float ys = luma(se);
        float skin = smoothstep(0.50, 0.54, cr) * (1.0 - smoothstep(0.66, 0.70, cr))
                   * smoothstep(0.28, 0.32, cb) * (1.0 - smoothstep(0.48, 0.52, cb))
                   * smoothstep(0.08, 0.20, ys);
        float Y  = luma(O);
        float Ys = luma(S) * (1.0 - p.bloom) + luma(B) * p.bloom;
        float d = (Y - Ys) / (Ys + 0.05);
        float edge = exp(-d * d / 0.08);
        float Yn = mix(Y, Ys, clamp(p.softSkin * skin * edge * 0.85, 0.0, 1.0));
        O *= Yn / max(Y, 1e-4);
    }

    // [B] 胶片光晕：红橙色，带核心抑制
    if (p.halation > 0.001) {
        float Hm = halo.sample(linSampler, uv, level(0)).r;
        float core = smoothstep(p.halThreshold, p.halThreshold + 0.25, luma(L));
        O += p.halation * float3(1.0, 0.10, 0.033) * Hm * (1.0 - p.halCore * core);
    }

    // 色温与饱和度（线性光下）
    O *= float3(1.0 + 0.08 * p.warmth, 1.0, 1.0 - 0.08 * p.warmth);
    float Yc = luma(O);
    O = max(mix(float3(Yc), O, p.saturation), 0.0);

    // 色彩保持的高光肩部
    float knee = mix(0.85, 0.5, p.shoulder);
    float m = max(O.r, max(O.g, O.b));
    if (m > knee) {
        float r = 1.0 - knee;
        float mt = knee + r * (1.0 - exp(-(m - knee) / r));
        O *= mt / m;
    }

    float3 c = linearToSrgb(O);

    // [D] 颗粒：中间调最强，可向暗部偏置；尺寸与输出分辨率挂钩
    if (p.grain > 0.001) {
        float scale = max(p.grainSize, 0.3) * float(W) / 1500.0;
        float2 q = float2(gid) / scale + p.seed * float2(17.3, 9.1);
        float n = (valueNoise(q) - 0.5) * 2.0 * 0.7
                + (valueNoise(q * 2.03 + 11.1) - 0.5) * 2.0 * 0.3;
        float Y = luma(c);
        float sig = sqrt(clamp(Y * (1.0 - Y), 0.0, 0.25)) * 2.0 * (1.0 + p.grainShadow * (1.0 - Y));
        c += p.grain * 0.10 * sig * n;
    }

    outT.write(float4(clamp(c, 0.0, 1.0), 1.0), gid);
}

// ---------- 6. 预览：等比缩放显示 ----------
struct VOut { float4 pos [[position]]; float2 uv; };

vertex VOut previewVertex(uint vid [[vertex_id]], constant float2& scale [[buffer(0)]]) {
    float2 pos[4] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1) };
    float2 uvs[4] = { float2(0, 1),   float2(1, 1),  float2(0, 0),  float2(1, 0) };
    VOut o;
    o.pos = float4(pos[vid] * scale, 0.0, 1.0);
    o.uv = uvs[vid];
    return o;
}

fragment float4 previewFragment(VOut in [[stage_in]], texture2d<float> tex [[texture(0)]]) {
    return float4(tex.sample(linSampler, in.uv).rgb, 1.0);
}
