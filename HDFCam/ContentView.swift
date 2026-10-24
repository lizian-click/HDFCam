import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @State private var showPanel = false
    @State private var flash = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 0) {
                presetBar
                ZStack {
                    CameraPreview(camera: camera) { _ in
                        UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    }
                    if flash { Color.black.opacity(0.6).allowsHitTesting(false) }
                    if let t = camera.toast {
                        Text(t)
                            .font(.subheadline.weight(.medium))
                            .padding(.horizontal, 14).padding(.vertical, 8)
                            .background(.ultraThinMaterial, in: Capsule())
                            .transition(.opacity)
                    }
                    if !camera.authorized {
                        Text("请在 设置 → 柔光机 中允许使用相机")
                            .foregroundStyle(.white).padding()
                    }
                }
                zoomBar
                controlBar
            }

            if showPanel {
                VStack {
                    Spacer()
                    ParamPanel(camera: camera, close: { withAnimation(.easeOut(duration: 0.2)) { showPanel = false } })
                        .transition(.move(edge: .bottom))
                }
                .ignoresSafeArea(edges: .bottom)
            }
        }
        .preferredColorScheme(.dark)
        .statusBarHidden(true)
        .onAppear { camera.start() }
    }

    private var presetBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(Presets.all) { p in
                    Button { camera.apply(p) } label: {
                        Text(p.id)
                            .font(.footnote.weight(camera.presetName == p.id ? .semibold : .regular))
                            .padding(.horizontal, 12).padding(.vertical, 7)
                            .foregroundStyle(camera.presetName == p.id ? Color.black : Color.white)
                            .background(camera.presetName == p.id ? Color(red: 1, green: 0.85, blue: 0.6) : Color.white.opacity(0.12),
                                        in: Capsule())
                    }
                }
            }
            .padding(.horizontal, 16).padding(.vertical, 10)
        }
    }

    private var zoomBar: some View {
        HStack(spacing: 10) {
            ForEach(camera.zoomOptions) { z in
                Button { camera.setZoom(z) } label: {
                    Text(camera.zoomLabel == z.id ? "\(z.id)×" : z.id)
                        .font(.caption.weight(.semibold))
                        .frame(width: 38, height: 38)
                        .foregroundStyle(camera.zoomLabel == z.id ? Color.yellow : Color.white)
                        .background(Color.white.opacity(0.12), in: Circle())
                }
            }
        }
        .padding(.vertical, 10)
    }

    private var controlBar: some View {
        HStack {
            Button { camera.saveOriginal.toggle() } label: {
                VStack(spacing: 4) {
                    Image(systemName: camera.saveOriginal ? "square.on.square.fill" : "square.on.square")
                        .font(.title2)
                    Text(camera.saveOriginal ? "存原图" : "仅效果").font(.caption2)
                }
                .foregroundStyle(.white)
                .frame(width: 70)
            }
            Spacer()
            Button {
                UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                flash = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) { flash = false }
                camera.capture()
            } label: {
                ZStack {
                    Circle().stroke(Color.white, lineWidth: 4).frame(width: 76, height: 76)
                    Circle().fill(camera.busy ? Color.gray : Color.white).frame(width: 62, height: 62)
                }
            }
            .disabled(camera.busy)
            Spacer()
            Button { withAnimation(.easeOut(duration: 0.2)) { showPanel = true } } label: {
                VStack(spacing: 4) {
                    Image(systemName: "slider.horizontal.3").font(.title2)
                    Text("调参").font(.caption2)
                }
                .foregroundStyle(.white)
                .frame(width: 70)
            }
        }
        .padding(.horizontal, 24)
        .padding(.bottom, 24)
        .padding(.top, 6)
    }
}

struct ParamPanel: View {
    @ObservedObject var camera: CameraManager
    var close: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("参数").font(.headline)
                Spacer()
                Button("重置预设") {
                    if let p = Presets.all.first(where: { $0.id == camera.presetName }) { camera.apply(p) }
                }
                .font(.subheadline)
                Button("完成", action: close).font(.subheadline.weight(.semibold)).padding(.leading, 16)
            }
            .padding(16)

            ScrollView {
                VStack(alignment: .leading, spacing: 6) {
                    section("曝光")
                    row("曝光补偿 EV", $camera.ev, -2...2, "%.1f")

                    section("HDF 柔光")
                    row("强度", $camera.fx.bloom, 0...0.2, "%.3f")
                    row("紧贴高光", $camera.fx.wCore, 0...1, "%.2f")
                    row("中晕", $camera.fx.wMid, 0...1, "%.2f")
                    row("长拖尾", $camera.fx.wTail, 0...1, "%.2f")
                    row("高光扩展（档）", $camera.fx.hlStops, 0...5, "%.1f")
                    row("高光起点", $camera.fx.hlThreshold, 0.6...0.98, "%.2f")
                    row("光晕暖边", $camera.fx.warmEdge, 0...1, "%.2f")

                    section("胶片红晕")
                    row("强度", $camera.fx.halation, 0...0.3, "%.3f")
                    row("阈值", $camera.fx.halThreshold, 0.4...1.2, "%.2f")
                    row("核心抑制", $camera.fx.halCore, 0...1, "%.2f")

                    section("人像")
                    row("柔肤", $camera.fx.softSkin, 0...1, "%.2f")

                    section("影调")
                    row("高光压缩", $camera.fx.shoulder, 0...1, "%.2f")
                    row("色温", $camera.fx.warmth, -1...1, "%.2f")
                    row("饱和度", $camera.fx.saturation, 0...1.5, "%.2f")

                    section("颗粒")
                    row("强度", $camera.fx.grain, 0...1, "%.2f")
                    row("尺寸", $camera.fx.grainSize, 0.6...3, "%.1f")
                    row("暗部偏重", $camera.fx.grainShadow, 0...1, "%.2f")
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 40)
            }
        }
        .frame(maxHeight: UIScreen.main.bounds.height * 0.5)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
    }

    private func section(_ t: String) -> some View {
        Text(t).font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.top, 12)
    }

    private func row(_ title: String, _ value: Binding<Float>, _ range: ClosedRange<Float>, _ fmt: String) -> some View {
        VStack(spacing: 2) {
            HStack {
                Text(title).font(.footnote)
                Spacer()
                Text(String(format: fmt, value.wrappedValue)).font(.footnote.monospacedDigit()).foregroundStyle(.secondary)
            }
            Slider(value: value, in: range)
        }
    }
}
