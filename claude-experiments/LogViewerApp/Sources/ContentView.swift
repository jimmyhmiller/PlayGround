import SwiftUI

struct ContentView: View {
    @EnvironmentObject var logStore: LogStore
    
    var body: some View {
        HStack(spacing: 0) {
            TimelineSidebar()
                .frame(width: 80)
                .background(Color(red: 0.12, green: 0.12, blue: 0.13))
            
            Divider()
                .background(Color(red: 0.2, green: 0.2, blue: 0.2))
            
            LogContentView()
                .background(Color(red: 0.15, green: 0.15, blue: 0.16))
        }
        .frame(minWidth: 800, minHeight: 600)
        .preferredColorScheme(.dark)
    }
}