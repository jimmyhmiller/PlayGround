import Foundation
import Combine

class SessionManager: ObservableObject {
    @Published var sessions: [WorkspaceSession] = []
    @Published var selectedSession: WorkspaceSession?
    
    private let userDefaults = UserDefaults.standard
    private let sessionsKey = "ClaudeCodeSessions"
    
    init() {
        loadSessions()
    }
    
    func addSession(name: String, path: String) {
        let session = WorkspaceSession(name: name, path: path)
        sessions.append(session)
        saveSessions()
    }
    
    func removeSession(_ session: WorkspaceSession) {
        if let index = sessions.firstIndex(where: { $0.id == session.id }) {
            // Stop the session if it's active
            if sessions[index].status == .active {
                stopSession(sessions[index])
            }
            sessions.remove(at: index)
            saveSessions()
        }
    }
    
    func updateSession(_ session: WorkspaceSession) {
        if let index = sessions.firstIndex(where: { $0.id == session.id }) {
            sessions[index] = session
            saveSessions()
            // Note: Disabled old todo saving since we now use raw markdown editor
            // saveTodosToFile(for: sessions[index])
        }
    }
    
    func startSession(_ session: WorkspaceSession) {
        guard let index = sessions.firstIndex(where: { $0.id == session.id }) else { return }
        
        let claudeCodePath = "/usr/local/bin/claude-code"
        let process = Process()
        process.executableURL = URL(fileURLWithPath: claudeCodePath)
        process.arguments = ["--resume", session.path]
        
        do {
            try process.run()
            sessions[index].status = .active
            sessions[index].processID = process.processIdentifier
            sessions[index].lastUsed = Date()
            saveSessions()
        } catch {
            sessions[index].status = .error
            saveSessions()
            print("Failed to start Claude Code session: \(error)")
        }
    }
    
    func stopSession(_ session: WorkspaceSession) {
        guard let index = sessions.firstIndex(where: { $0.id == session.id }),
              let processID = sessions[index].processID else { return }
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/kill")
        process.arguments = [String(processID)]
        
        do {
            try process.run()
            sessions[index].status = .inactive
            sessions[index].processID = nil
            saveSessions()
        } catch {
            print("Failed to stop Claude Code session: \(error)")
        }
    }
    
    func selectSession(_ session: WorkspaceSession) {
        selectedSession = session
        // Note: Disabled old todo parsing since we now use raw markdown editor
        // loadTodosFromFile(for: session)
    }
    
    private func saveSessions() {
        if let encoded = try? JSONEncoder().encode(sessions) {
            userDefaults.set(encoded, forKey: sessionsKey)
        }
    }
    
    private func loadSessions() {
        guard let data = userDefaults.data(forKey: sessionsKey),
              let decoded = try? JSONDecoder().decode([WorkspaceSession].self, from: data) else {
            return
        }
        sessions = decoded
    }
    
    private func loadTodosFromFile(for session: WorkspaceSession) {
        guard let index = sessions.firstIndex(where: { $0.id == session.id }) else { 
            return 
        }
        
        let basePath = URL(fileURLWithPath: session.path)
        let possiblePaths = [
            basePath.appendingPathComponent("TODO.md"),
            basePath.appendingPathComponent("todo.md"),
            basePath.appendingPathComponent("Todo.md")
        ]
        
        var todoPath: URL?
        var content: String?
        
        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path.path) {
                if let fileContent = try? String(contentsOf: path) {
                    todoPath = path
                    content = fileContent
                    break
                }
            }
        }
        
        guard let content = content else {
            return
        }
        
        let todos = parseMarkdownTodos(content)
        sessions[index].todos = todos
        
        if selectedSession?.id == session.id {
            selectedSession = sessions[index]
        }
    }
    
    private func saveTodosToFile(for session: WorkspaceSession) {
        let basePath = URL(fileURLWithPath: session.path)
        let possiblePaths = [
            basePath.appendingPathComponent("TODO.md"),
            basePath.appendingPathComponent("todo.md"),
            basePath.appendingPathComponent("Todo.md")
        ]
        
        // Find existing file or default to TODO.md
        var todoPath = basePath.appendingPathComponent("TODO.md")
        for path in possiblePaths {
            if FileManager.default.fileExists(atPath: path.path) {
                todoPath = path
                break
            }
        }
        
        let content = generateMarkdownTodos(session.todos)
        try? content.write(to: todoPath, atomically: true, encoding: .utf8)
    }
    
    private func parseMarkdownTodos(_ content: String) -> [TodoItem] {
        let lines = content.components(separatedBy: .newlines)
        var todos: [TodoItem] = []
        
        for (index, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            
            // Parse standard checkbox markdown todo items: - [ ] content or - [x] content
            if trimmed.hasPrefix("- [") && trimmed.count > 5 {
                let isCompleted = trimmed.hasPrefix("- [x]") || trimmed.hasPrefix("- [X]")
                let isInProgress = trimmed.hasPrefix("- [~]") || trimmed.hasPrefix("- [-]")
                
                let contentStart = trimmed.index(trimmed.startIndex, offsetBy: 5)
                let content = String(trimmed[contentStart...]).trimmingCharacters(in: .whitespaces)
                
                if !content.isEmpty {
                    let status: TodoStatus = isCompleted ? .completed : (isInProgress ? .inProgress : .pending)
                    let todo = TodoItem(content: content, status: status, priority: .medium)
                    todos.append(todo)
                }
            }
            // Parse heading-style todos: ### 4. ✅ Task name or ### 4. Task name
            else if trimmed.hasPrefix("### ") && trimmed.count > 4 {
                let headerContent = String(trimmed.dropFirst(4)).trimmingCharacters(in: .whitespaces)
                
                let isCompleted = headerContent.contains("✅")
                var cleanContent = headerContent
                
                // Remove number prefix like "4. " 
                if let dotIndex = cleanContent.firstIndex(of: "."),
                   cleanContent.distance(from: cleanContent.startIndex, to: dotIndex) < 5 {
                    let afterDot = cleanContent.index(after: dotIndex)
                    if afterDot < cleanContent.endIndex {
                        cleanContent = String(cleanContent[afterDot...]).trimmingCharacters(in: .whitespaces)
                    }
                }
                
                // Remove checkmark emoji
                cleanContent = cleanContent.replacingOccurrences(of: "✅", with: "").trimmingCharacters(in: .whitespaces)
                
                if !cleanContent.isEmpty {
                    let status: TodoStatus = isCompleted ? .completed : .pending
                    
                    // Determine priority based on section
                    var priority: TodoPriority = .medium
                    if content.contains("High Priority") {
                        priority = .high
                    } else if content.contains("Low Priority") {
                        priority = .low
                    }
                    
                    let todo = TodoItem(content: cleanContent, status: status, priority: priority)
                    todos.append(todo)
                }
            }
        }
        
        return todos
    }
    
    private func generateMarkdownTodos(_ todos: [TodoItem]) -> String {
        var lines: [String] = []
        
        for todo in todos {
            let checkbox: String
            switch todo.status {
            case .pending:
                checkbox = "- [ ]"
            case .inProgress:
                checkbox = "- [~]"
            case .completed:
                checkbox = "- [x]"
            }
            
            let priorityMarker: String
            switch todo.priority {
            case .high:
                priorityMarker = " !!!"
            case .medium:
                priorityMarker = " !!"
            case .low:
                priorityMarker = " !"
            }
            
            lines.append("\(checkbox) \(todo.content)\(priorityMarker)")
        }
        
        return lines.joined(separator: "\n")
    }
}