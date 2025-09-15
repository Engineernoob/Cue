// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Command, Stdio, Child};
use std::sync::{Arc, Mutex};
use tauri::Manager;

fn main() {
    // Shared reference to backend process
    let backend_process: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));

    {
        let backend_process = Arc::clone(&backend_process);

        // 🚀 Launch Python backend automatically in dev builds
        #[cfg(debug_assertions)]
        {
            let child = Command::new("python3")
                .args(["backend/main.py"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("❌ Failed to start Python backend");

            *backend_process.lock().unwrap() = Some(child);
        }
    }

    tauri::Builder::default()
        .setup(move |app| {
            let backend_process = Arc::clone(&backend_process);

            // Kill backend when all windows are closed
            let app_handle = app.handle();
            app_handle.on_window_event(move |event| {
                if let tauri::WindowEvent::Destroyed = event.event() {
                    // when last window closes, shut down backend
                    if app_handle.webview_windows().is_empty() {
                        if let Some(mut child) = backend_process.lock().unwrap().take() {
                            let _ = child.kill();
                        }
                    }
                }
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("❌ error while running Cue");
}

