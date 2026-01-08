import Foundation

/// SSH connection pool for better performance and resource management
class SSHConnectionPool {
    private static var sharedServices: [String: SSHService] = [:]
    private static let lock = NSLock()
    
    /// Get or create an SSH service for a specific server
    /// - Parameter server: The server to connect to
    /// - Returns: An SSH service instance
    static func getService(for server: Server) -> SSHService {
        lock.lock()
        defer { lock.unlock() }
        
        let serviceKey = "\(server.host):\(server.port):\(server.username)"
        
        if let existingService = sharedServices[serviceKey] {
            return existingService
        }
        
        // Create new service
        let newService = SSHService()
        sharedServices[serviceKey] = newService
        return newService
    }
    
    /// Remove a service from the pool (call when disconnecting)
    /// - Parameter server: The server whose service should be removed
    static func removeService(for server: Server) {
        lock.lock()
        defer { lock.unlock() }
        
        let serviceKey = "\(server.host):\(server.port):\(server.username)"
        sharedServices.removeValue(forKey: serviceKey)
    }
    
    /// Clear all services from the pool
    static func clearAll() {
        lock.lock()
        defer { lock.unlock() }
        
        sharedServices.removeAll()
    }
    
    /// Get the number of active connections
    static func activeConnectionCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        
        return sharedServices.count
    }
}