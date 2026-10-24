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
                    if camera.useRAW {
                        VStack {
                            HStack {
                                Spacer()
                                Text("RAW").font(.caption2.weight(.bold))
                                    .padding(.horizontal, 6).padding(.vertical, 3)
                                    .background(Color.orange, in: RoundedRectangle(cornerRadius: 4))
                                    .foregroundStyle(.black)
                                    .padding(10)
                            }
                            Spacer()
                        }
                        .allowsHitTesting(false)
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
                    row("曝光补偿 EV", $camera.ev, -2...2) { String(format: "%.1f", $0) }
                    row("数字曝光", $camera.look.exposure, -1.5...1.5) { String(format: "%.1f 档", $0) }
                    row("点光源能量", $camera.look.pointBoost, 0...1.5) { String(format: "×%.2f", $0) }
                    row("高光肩部", $camera.look.knee, 0.55...0.9) { String(format: "%.2f", $0) }
                    if camera.rawSupported {
                        Toggle("RAW 真高光（拍照用 ProRAW，较慢）", isOn: $camera.useRAW).font(.footnote).padding(.top, 4)
                    }

                    section("光学滤镜")
                    row("理光 HDF", $camera.look.hdf, 0...1.3) { hdfLabel($0) }
                    row("天芬 黑丝绒", $camera.look.satin, 0...4) { gradeLabel($0) }
                    row("天芬 珠光", $camera.look.pearl, 0...4) { gradeLabel($0) }
                    row("天芬 微光", $camera.look.glimmer, 0...4) { gradeLabel($0) }
                    row("黑柔（对照）", $camera.look.bpm, 0...4) { gradeLabel($0) }

                    section("胶片")
                    HStack {
                        Text("片种").font(.footnote)
                        Spacer()
                        Picker("片种", selection: $camera.look.stock) {
                            ForEach(FilmStock.allCases) { s in Text(s.name).tag(s) }
                        }
                        .pickerStyle(.menu)
                    }
                    row("颗粒强度", $camera.look.grain, 0...2) { String(format: "×%.2f", $0) }
                    row("胶片影调", $camera.look.tone, 0...1) { String(format: "%.2f", $0) }
                    row("人像色彩", $camera.look.hsl, 0...1) { String(format: "%.2f", $0) }
                    row("冷暖", $camera.look.warm, -1...1) { String(format: "%+.2f", $0) }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 40)
            }
        }
        .frame(maxHeight: UIScreen.main.bounds.height * 0.55)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
    }

    private func hdfLabel(_ v: Float) -> String {
        if v < 0.02 { return "关" }
        let stops: [(Float, String)] = [(1.0, "≈f/2.8"), (0.75, "≈f/4"), (0.55, "≈f/5.6"), (0.4, "≈f/8")]
        let nearest = stops.min { abs($0.0 - v) < abs($1.0 - v) }!
        return String(format: "%.2f %@", v, nearest.1)
    }

    private func gradeLabel(_ g: Float) -> String {
        if g < 0.05 { return "关" }
        let names = ["1/8", "1/4", "1/2", "1"]
        let i = min(max(Int(g.rounded()) - 1, 0), 3)
        return String(format: "%.1f · %@", g, names[i])
    }

    private func section(_ t: String) -> some View {
        Text(t).font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.top, 12)
    }

    private func row(_ title: String, _ value: Binding<Float>, _ range: ClosedRange<Float>, _ fmt: @escaping (Float) -> String) -> some View {
        VStack(spacing: 2) {
            HStack {
                Text(title).font(.footnote)
                Spacer()
                Text(fmt(value.wrappedValue)).font(.footnote.monospacedDigit()).foregroundStyle(.secondary)
            }
            Slider(value: value, in: range)
        }
    }
}
