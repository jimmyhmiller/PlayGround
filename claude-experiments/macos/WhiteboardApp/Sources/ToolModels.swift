import SwiftUI

enum Tool: CaseIterable {
    case select
    case rectangle
    case text
    
    var name: String {
        switch self {
        case .select: return "Select"
        case .rectangle: return "Rectangle"
        case .text: return "Text"
        }
    }
    
    var icon: String {
        switch self {
        case .select: return "cursor.arrow"
        case .rectangle: return "rectangle"
        case .text: return "textformat"
        }
    }
}