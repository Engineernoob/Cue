// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Command, Stdio};
use tauri::Manager;

fn main() {
    // 🚀 Launch Python backend automatically in dev builds
    #[cfg(debug_assertions)]
    {
        let _ = Command::new("python3") // macOS: use python3 explicitly
            .arg("backend/main.py")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to start Python backend");
    }

    tauri::Builder::default()
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                window.set_always_on_top(true).unwrap();
                window.set_decorations(false).unwrap();
                window.set_always_on_top(true).unwrap();
                window.set_size(tauri::Size::Logical(tauri::LogicalSize {
                    width: 800.0,
                    height: 100.0,
                })).unwrap();
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Cue");
}
