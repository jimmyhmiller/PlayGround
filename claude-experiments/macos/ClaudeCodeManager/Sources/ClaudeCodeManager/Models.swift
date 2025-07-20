import Foundation
import AppKit

struct WorkspaceSession: Identifiable, Codable {
    var id = UUID()
    var name: String
    var path: String
    var status: SessionStatus
    var lastUsed: Date
    var processID: Int32?
    var todos: [TodoItem]
    
    init(name: String, path: String) {
        self.id = UUID()
        self.name = name
        self.path = path
        self.status = .inactive
        self.lastUsed = Date()
        self.todos = []
    }
}

enum SessionStatus: String, Codable, CaseIterable {
    case active = "Active"
    case inactive = "Inactive"
    case error = "Error"
    
    var color: NSColor {
        switch self {
        case .active: return .systemGreen
        case .inactive: return .systemGray
        case .error: return .systemRed
        }
    }
}

struct TodoItem: Identifiable, Codable {
    var id = UUID()
    var content: String
    var status: TodoStatus
    var priority: TodoPriority
    var createdAt: Date
    
    init(content: String, status: TodoStatus = .pending, priority: TodoPriority = .medium) {
        self.id = UUID()
        self.content = content
        self.status = status
        self.priority = priority
        self.createdAt = Date()
    }
}

enum TodoStatus: String, Codable, CaseIterable {
    case pending = "Pending"
    case inProgress = "In Progress"
    case completed = "Completed"
    
    var color: NSColor {
        switch self {
        case .pending: return .systemOrange
        case .inProgress: return .systemBlue
        case .completed: return .systemGreen
        }
    }
}

enum TodoPriority: String, Codable, CaseIterable {
    case low = "Low"
    case medium = "Medium"
    case high = "High"
    
    var color: NSColor {
        switch self {
        case .low: return .systemGray
        case .medium: return .systemYellow
        case .high: return .systemRed
        }
    }
}