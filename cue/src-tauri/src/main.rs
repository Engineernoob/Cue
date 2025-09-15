// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Command, Stdio};
use tauri::{Manager, WindowEvent};

fn main() {
    // 🚀 Launch Python backend automatically in dev builds
    #[cfg(debug_assertions)]
    {
        let _ = Command::new("python3")
            .args(["backend/main.py"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to start Python backend");
    }

    tauri::Builder::default()
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                window.set_always_on_top(true).ok();
                window.set_decorations(false).ok();
                window.set_size(tauri::Size::Logical(tauri::LogicalSize {
                    width: 800.0,
                    height: 100.0,
                })).ok();
                window.set_resizable(true).ok();
                window.center().ok();

                // ✅ Listen for window events at the window level
                let _id = window.listen("tauri://close-requested", move |_event| {
                    println!("⚠️ Main window close requested.");
                    // You can gracefully shut down backend here if needed
                });
            } else {
                println!("⚠️ No 'main' window found. Check tauri.conf.json!");
            }
            Ok(())
        })
        .on_window_event(|event| {
            if let WindowEvent::CloseRequested { api, .. } = event.event() {
                println!("⚠️ Close requested, preventing quit.");
                api.prevent_close(); // keeps the app running in tray if you add one later
            }
        })
        .run(tauri::generate_context!())
        .expect("❌ error while running Cue");
}


