use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};
use std::thread;

use structured_lang::database::Database;
use structured_lang::gui::App;
use structured_lang::wire::dispatch;

fn main() {
    let db = Arc::new(Mutex::new(Database::new()));

    // Seed with a default branch so the UI has something to show
    {
        let mut db = db.lock().unwrap();
        db.create_branch("main");
    }

    // Spawn stdin reader thread — agent commands come in here
    let db_stdin = db.clone();
    thread::spawn(move || {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let mut db = db_stdin.lock().unwrap();
            let response = dispatch(&mut db, trimmed);
            drop(db); // release lock before writing
            let json = serde_json::to_string(&response).unwrap_or_else(|e| {
                format!(r#"{{"ok":false,"error":"{}"}}"#, e)
            });
            writeln!(stdout, "{}", json).ok();
            stdout.flush().ok();
        }
    });

    // Run the GUI on the main thread
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 700.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Structured Lang — Version Control for Structure Editing",
        native_options,
        Box::new(move |_cc| Ok(Box::new(App::new(db.clone())))),
    ).unwrap();
}
