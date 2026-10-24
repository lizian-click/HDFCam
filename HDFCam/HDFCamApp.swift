import SwiftUI

@main
struct HDFCamApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .onAppear { UIApplication.shared.isIdleTimerDisabled = true } // 摆摊时不自动锁屏
        }
    }
}
