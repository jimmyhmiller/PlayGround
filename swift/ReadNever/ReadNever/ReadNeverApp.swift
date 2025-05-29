//
//  ReadNeverApp.swift
//  ReadNever
//
//  Created by Jimmy Miller on 5/25/25.
//

import SwiftUI
import SwiftData

@main
struct ReadNeverApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .navigationTitle("LinkReader")
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified)
        .windowResizability(.contentSize)
        .defaultSize(width: 1024, height: 768)
        .modelContainer(
            for: [LinkItem.self, Category.self],
            inMemory: false,
            isAutosaveEnabled: true,
            isUndoEnabled: true,
            onSetup: { container in
                print("Model container setup completed successfully")
            }
        )
    }
    
    // Helper to delete database in case of critical errors
    func deleteDatabase() {
        let containerURL = URL.applicationSupportDirectory.appendingPathComponent("default.store")
        
        do {
            try FileManager.default.removeItem(at: containerURL)
            print("Database removed due to critical error. The app will create a fresh database.")
        } catch {
            print("Failed to remove database: \(error)")
        }
    }
}
