//  Shaders.metal — 柔光机 v2
//  管线：线性化 → 点光源感知的高光重建 → 光学 PSF 引擎（HDF / 黑丝绒 / 珠光 / 微光 / 黑柔）
//        → 微柔化 → 白平衡 → 色彩保持肩部 → 胶片曲线 / 分离色调 / HSL → 对数域布尔模型颗粒
//  所有尺寸都按「画幅短边 H 的百分比」定义，预览和成片一致。
#include <metal_stdlib>
using namespace metal;

// 必须与 Swift 端 GPUParams 字段顺序完全一致（全部 float）
struct GPUParams {
    float exposure;       // 线性增益（档）
    float clipLevel;      // 输入的裁切白位（SDR=1，RAW 为实测最大值）
    float linearInput;    // 1 = 输入已是线性光（RAW）
    float eArea;          // 大面积高光扩展（档）
    float ePtDay;         // 白天点光源扩展（档）
    float ePtNight;       // 夜景点光源扩展（档）
    float fTotal;         // 总散射能量
    float veil;           // 遮幕眩光强度
    float veilR, veilG, veilB;
    float soften;         // 微柔化混合量
    float softT;          // 微柔化尺度（0 = 1 级 mip，1 = 2 级 mip）
    float gritScale;      // 黑丝绒颗粒感噪声尺寸（像素）
    float curveAmt;       // 胶片曲线
    float wbWarm;         // 白平衡偏暖（1 ≈ +200K）
    float splitAmt;       // 分离色调
    float hslAmt;         // 人像 HSL
    float grainAmt;       // 颗粒峰值强度（EV）
    float grainTexPerPx;  // 颗粒纹理每像素纹素数
    float grainChroma;    // 彩色颗粒比例
    float grainOffX, grainOffY, grainRot;
    float halThresh;      // 红晕阈值（线性，1 = 白）
    float halCore;        // 红晕核心抑制
    float shoulderKnee;   // 肩部起点
    float bw;             // 1 = 黑白
    float shortSide;      // 当前分辨率的短边像素
    float pad0, pad1, pad2;
};

struct LobeParam {
    float r, g, b;   // 已乘上散射能量与色调
    float desat;     // 散射光去饱和
    float grit;      // 黑丝绒颗粒感
    float mode;      // 0 = 光学散射，2 = 胶片红晕
    float active;
    float pad;
};

struct BlurParam { float sigma; float radius; float dx; float dy; };

constexpr sampler linSampler(coord::normalized, address::clamp_to_edge, filter::linear);
constexpr sampler repSampler(coord::normalized, address::repeat, filter::linear);

static inline float3 srgbToLinear(float3 c) {
    c = clamp(c, 0.0, 1.0);
    return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
}

static inline float3 linearToSrgb(float3 c) {
    c = clamp(c, 0.0, 1.0);
    return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
}

static inline float luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }
static inline float max3(float3 c) { return max(c.r, max(c.g, c.b)); }

// ---------------- 1. 线性化：写出线性图与「色相/裁切」辅助图 ----------------
kernel void prepAKernel(texture2d<float, access::read>  src    [[texture(0)]],
                        texture2d<float, access::write> linRaw [[texture(1)]],
                        texture2d<float, access::write> aux    [[texture(2)]],
                        constant GPUParams& p [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= linRaw.get_width() || gid.y >= linRaw.get_height()) return;
    float3 v = src.read(gid).rgb;
    float3 L = p.linearInput > 0.5 ? max(v, 0.0) : srgbToLinear(v);
    float mN = max3(L) / p.clipLevel;
    float c = smoothstep(0.80, 0.98, mN);
    float M0 = step(0.98, mN);
    linRaw.write(float4(L, M0), gid);
    aux.write(float4(L * (1.0 - c), M0), gid);
}

// ---------------- 2. 2× 盒式降采样（一次双线性采样 = 4 像素平均） ----------------
kernel void downBoxKernel(texture2d<float, access::sample> src [[texture(0)]],
                          texture2d<float, access::write>  dst [[texture(1)]],
                          uint2 gid [[thread_position_in_grid]])
{
    uint w = dst.get_width(), h = dst.get_height();
    if (gid.x >= w || gid.y >= h) return;
    float2 uv = (float2(gid) + 0.5) / float2(w, h);
    dst.write(src.sample(linSampler, uv, level(0)), gid);
}

// ---------------- 3. 可分离高斯（在某一级 mip 上做） ----------------
kernel void blurKernel(texture2d<float, access::read>  src [[texture(0)]],
                       texture2d<float, access::write> dst [[texture(1)]],
                       constant BlurParam& b [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]])
{
    int w = int(dst.get_width()), h = int(dst.get_height());
    if (int(gid.x) >= w || int(gid.y) >= h) return;
    int R = int(b.radius);
    float inv = 1.0 / (2.0 * b.sigma * b.sigma);
    float4 sum = 0.0;
    float wsum = 0.0;
    int2 dir = int2(int(b.dx), int(b.dy));
    for (int i = -R; i <= R; i++) {
        int2 q = clamp(int2(gid) + dir * i, int2(0), int2(w - 1, h - 1));
        float wt = exp(-float(i * i) * inv);
        sum += src.read(uint2(q)) * wt;
        wsum += wt;
    }
    dst.write(sum / wsum, gid);
}

// ---------------- 4. 高光重建：区分点光源与大面积白 ----------------
kernel void prepBKernel(texture2d<float, access::read>   linRaw  [[texture(0)]],
                        texture2d<float, access::sample> auxBlur [[texture(1)]],
                        texture2d<float, access::sample> auxDeep [[texture(2)]],
                        texture2d<float, access::write>  linHat  [[texture(3)]],
                        constant GPUParams& p [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]])
{
    uint W = linHat.get_width(), H = linHat.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float2 uv = (float2(gid) + 0.5) / float2(W, H);

    float4 raw = linRaw.read(gid);
    float3 L = raw.rgb;
    float M0 = raw.a;
    float m = max3(L);
    float c = smoothstep(0.80, 0.98, m / p.clipLevel);

    float4 ab = auxBlur.sample(linSampler, uv, level(0));
    float Mb = ab.a;
    float P = saturate((M0 - Mb) / (M0 + 1e-3));                 // 孤立小亮点 ≈ 1，大面积白 ≈ 0
    float meanY = luma(auxDeep.sample(linSampler, float2(0.5), level(0)).rgb) / p.clipLevel;
    float N = 1.0 - smoothstep(0.03, 0.12, meanY);              // 夜景度
    float Ept = mix(p.ePtDay, p.ePtNight, N);
    float gain = exp2(c * mix(p.eArea, Ept, P));

    float3 hue = ab.rgb;
    float hm = max3(hue);
    float3 hueN = hm > 1e-4 ? hue / hm : float3(1.0, 0.93, 0.82);
    hueN = mix(float3(1.0, 0.93, 0.82), hueN, saturate(hm / (0.05 * p.clipLevel)));

    float3 Lh = mix(L, m * gain * hueN, c);
    Lh = Lh / p.clipLevel * exp2(p.exposure);
    linHat.write(float4(Lh, 1.0), gid);
}

// ---------------- 工具：噪声、HSV ----------------
static inline float hash21(float2 q) {
    q = fract(q * float2(123.34, 456.21));
    q += dot(q, q + 45.32);
    return fract(q.x * q.y);
}

static inline float valueNoise(float2 q) {
    float2 i = floor(q), f = fract(q);
    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + float2(1, 0)), u.x),
               mix(hash21(i + float2(0, 1)), hash21(i + float2(1, 1)), u.x), u.y);
}

static inline float3 rgb2hsv(float3 c) {
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = mix(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    float4 q = mix(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

static inline float3 hsv2rgb(float3 c) {
    float3 p = abs(fract(c.xxx + float3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(float3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

static inline float hueDist(float a, float b) {   // 度
    float d = fmod(abs(a - b), 360.0);
    return min(d, 360.0 - d);
}

// Portra / Frontier 风格的显示域曲线：抬趾部、柔肩、白点不顶满
static inline float filmCurve(float x) {
    const float xs[7] = { 0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0 };
    const float ys[7] = { 0.035, 0.115, 0.255, 0.52, 0.76, 0.875, 0.955 };
    x = clamp(x, 0.0, 1.0);
    for (int i = 0; i < 6; i++) {
        if (x <= xs[i + 1]) {
            float t = (x - xs[i]) / (xs[i + 1] - xs[i]);
            return mix(ys[i], ys[i + 1], t);
        }
    }
    return ys[6];
}

// 负片扫描的颗粒强度：暗部到中低调最强，高光最弱
static inline float grainCurve(float y) {
    const float xs[8] = { 0.0, 0.02, 0.08, 0.18, 0.35, 0.60, 0.85, 1.0 };
    const float ys[8] = { 0.50, 0.70, 1.00, 0.95, 0.75, 0.50, 0.30, 0.15 };
    y = clamp(y, 0.0, 1.0);
    for (int i = 0; i < 7; i++) {
        if (y <= xs[i + 1]) {
            float t = (y - xs[i]) / (xs[i + 1] - xs[i]);
            return mix(ys[i], ys[i + 1], t);
        }
    }
    return ys[7];
}

// ---------------- 5. 合成 ----------------
kernel void compositeKernel(texture2d<float, access::read>   linHat [[texture(0)]],
                            texture2d<float, access::sample> mip1   [[texture(1)]],
                            texture2d<float, access::sample> mip2   [[texture(2)]],
                            texture2d<float, access::sample> deep   [[texture(3)]],
                            texture2d<float, access::sample> grainT [[texture(4)]],
                            array<texture2d<float, access::sample>, 7> lobes [[texture(5)]],
                            texture2d<float, access::write>  outT   [[texture(12)]],
                            constant GPUParams& p [[buffer(0)]],
                            constant LobeParam* lp [[buffer(1)]],
                            uint2 gid [[thread_position_in_grid]])
{
    uint W = outT.get_width(), H = outT.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float2 uv = (float2(gid) + 0.5) / float2(W, H);

    float3 L = linHat.read(gid).rgb;

    // 微柔化（黑丝绒/珠光的整体变软 + 胶片降锐），只动亮度
    if (p.soften > 0.001) {
        float3 Ls = mix(mip1.sample(linSampler, uv, level(0)).rgb, mip2.sample(linSampler, uv, level(0)).rgb, p.softT);
        float Y = luma(L);
        float Yn = mix(Y, luma(Ls), p.soften);
        L *= Yn / max(Y, 1e-5);
    }

    // 光学散射：O = (1-f)·L + Σ 光瓣
    float3 O = (1.0 - p.fTotal) * L;
    float grit = (valueNoise(float2(gid) / max(p.gritScale, 0.5)) - 0.5) * 2.0;
    for (int i = 0; i < 7; i++) {
        if (lp[i].active < 0.5) continue;
        float3 s = lobes[i].sample(linSampler, uv, level(0)).rgb;
        if (lp[i].mode < 1.0) {
            s = mix(s, float3(luma(s)), lp[i].desat);
            O += s * float3(lp[i].r, lp[i].g, lp[i].b) * (1.0 + lp[i].grit * 0.35 * grit);
        } else {
            float hY = max(luma(s) - p.halThresh, 0.0);
            float core = smoothstep(p.halThresh, p.halThresh * 4.0, luma(L));
            O += float3(lp[i].r, lp[i].g, lp[i].b) * hY * (1.0 - p.halCore * core);
        }
    }

    // 遮幕眩光（让黑位略微上浮，珠光/微光的「奶油感」来源之一）
    float meanY = luma(deep.sample(linSampler, float2(0.5), level(0)).rgb);
    O += p.veil * meanY * float3(p.veilR, p.veilG, p.veilB);

    // 白平衡偏暖
    O *= float3(1.0 + 0.03 * p.wbWarm, 1.0, 1.0 - 0.04 * p.wbWarm);
    O = max(O, 0.0);

    // 色彩保持的高光肩部（光晕区不被削成纯白）
    float knee = p.shoulderKnee;
    float m = max3(O);
    if (m > knee) {
        float r = 1.0 - knee;
        O *= (knee + r * (1.0 - exp(-(m - knee) / r))) / m;
    }

    float3 c = linearToSrgb(O);

    // 胶片曲线
    if (p.curveAmt > 0.001) {
        float3 cf = float3(filmCurve(c.r), filmCurve(c.g), filmCurve(c.b));
        c = mix(c, cf, p.curveAmt);
    }

    // 分离色调：阴影青绿 185°，高光暖 40°
    if (p.splitAmt > 0.001) {
        float Y = luma(c);
        float ws = 1.0 - smoothstep(0.0, 0.45, Y);
        float wh = smoothstep(0.55, 1.0, Y);
        c += p.splitAmt * (ws * float3(-0.0437, 0.0113, 0.0163) + wh * float3(0.0155, -0.0012, -0.0345));
        c = clamp(c, 0.0, 1.0);
    }

    // 人像 HSL：橙提亮降饱和、红保护、黄绿偏橄榄降饱和、蓝偏青
    if (p.hslAmt > 0.001) {
        float3 hsv = rgb2hsv(c);
        float hd = hsv.x * 360.0;
        // 中心色相, 半宽, 色相偏移(度), 饱和度, 明度
        const float4 bands[5] = {
            float4(0.0,   25.0,  1.0, -0.03),
            float4(30.0,  25.0,  0.0, -0.05),
            float4(60.0,  25.0, -2.4, -0.10),
            float4(120.0, 45.0, -4.5, -0.20),
            float4(220.0, 40.0, -2.4, -0.10)
        };
        const float lum[5] = { 0.0, 0.04, 0.0, -0.02, 0.0 };
        float dh = 0.0, ds = 0.0, dv = 0.0;
        for (int i = 0; i < 5; i++) {
            float w = max(0.0, 1.0 - hueDist(hd, bands[i].x) / bands[i].y);
            dh += w * bands[i].z; ds += w * bands[i].w; dv += w * lum[i];
        }
        hsv.x = fract((hd + dh * p.hslAmt) / 360.0 + 1.0);
        hsv.y = saturate(hsv.y * (1.0 + ds * p.hslAmt));
        hsv.z = saturate(hsv.z * (1.0 + dv * p.hslAmt * hsv.y));
        c = hsv2rgb(hsv);
    }

    if (p.bw > 0.5) c = float3(luma(c));

    // 颗粒：布尔模型纹理 × 3 个尺度，在对数域（乘性）施加
    if (p.grainAmt > 0.0001) {
        float2 q = float2(gid) * p.grainTexPerPx + float2(p.grainOffX, p.grainOffY);
        if (p.grainRot > 0.5) q = float2(q.y, -q.x);
        float3 n = 0.60 * grainT.sample(repSampler, q / 512.0, level(0)).rgb
                 + 0.30 * grainT.sample(repSampler, (q * 0.5 + 137.0) / 512.0, level(0)).rgb
                 + 0.10 * grainT.sample(repSampler, (q * 0.25 + 291.0) / 512.0, level(0)).rgb;
        n /= 0.678;
        float3 chroma = float3(n.y, -0.5 * n.y + 0.866 * n.z, -0.5 * n.y - 0.866 * n.z);
        float3 nrgb = n.x + p.grainChroma * (1.0 - p.bw) * chroma;
        float sigma = grainCurve(luma(c)) * p.grainAmt;
        float3 lin = srgbToLinear(c) * exp2(sigma * nrgb);
        c = linearToSrgb(lin);
    }

    outT.write(float4(clamp(c, 0.0, 1.0), 1.0), gid);
}

// ---------------- 6. 预览：等比缩放显示 ----------------
struct VOut { float4 pos [[position]]; float2 uv; };

vertex VOut previewVertex(uint vid [[vertex_id]], constant float2& scale [[buffer(0)]]) {
    const float2 pos[4] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1) };
    const float2 uvs[4] = { float2(0, 1),   float2(1, 1),  float2(0, 0),  float2(1, 0) };
    VOut o;
    o.pos = float4(pos[vid] * scale, 0.0, 1.0);
    o.uv = uvs[vid];
    return o;
}

fragment float4 previewFragment(VOut in [[stage_in]], texture2d<float> tex [[texture(0)]]) {
    return float4(tex.sample(linSampler, in.uv).rgb, 1.0);
}
