import SwiftUI
import UIKit
import QuartzCore

/// 用 CAMetalLayer 直接显示，相机线程可直接往里画
final class MetalPreviewUIView: UIView {
    override class var layerClass: AnyClass { CAMetalLayer.self }
    var metalLayer: CAMetalLayer { layer as! CAMetalLayer }
    var onTap: ((CGFloat, CGFloat) -> Void)?
    /// 画面宽高比（宽/高），竖屏 4:3 照片为 0.75
    var imageAspect: CGFloat = 3.0 / 4.0

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .black
        metalLayer.pixelFormat = .bgra8Unorm
        metalLayer.framebufferOnly = true
        addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(tapped(_:))))
    }

    required init?(coder: NSCoder) { fatalError() }

    override func layoutSubviews() {
        super.layoutSubviews()
        let scale = window?.screen.scale ?? UIScreen.main.scale
        metalLayer.contentsScale = scale
        metalLayer.drawableSize = CGSize(width: bounds.width * scale, height: bounds.height * scale)
    }

    @objc private func tapped(_ g: UITapGestureRecognizer) {
        let p = g.location(in: self)
        let viewAspect = bounds.width / max(bounds.height, 1)
        var rect = bounds
        if imageAspect > viewAspect {
            let h = bounds.width / imageAspect
            rect = CGRect(x: 0, y: (bounds.height - h) / 2, width: bounds.width, height: h)
        } else {
            let w = bounds.height * imageAspect
            rect = CGRect(x: (bounds.width - w) / 2, y: 0, width: w, height: bounds.height)
        }
        guard rect.contains(p) else { return }
        onTap?((p.x - rect.minX) / rect.width, (p.y - rect.minY) / rect.height)
    }
}

struct CameraPreview: UIViewRepresentable {
    let camera: CameraManager
    var onTap: (CGPoint) -> Void

    func makeUIView(context: Context) -> MetalPreviewUIView {
        let v = MetalPreviewUIView()
        v.metalLayer.device = camera.renderer?.device
        camera.metalLayer = v.metalLayer
        v.onTap = { u, w in
            camera.focus(u: u, v: w)
            onTap(CGPoint(x: u, y: w))
        }
        return v
    }

    func updateUIView(_ uiView: MetalPreviewUIView, context: Context) {}
}
