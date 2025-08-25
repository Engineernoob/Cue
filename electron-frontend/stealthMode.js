// Stealth Mode Manager - Make Cue completely undetectable
const { ipcMain } = require('electron');
const { getConfig, setConfig } = require('./config');

class StealthModeManager {
  constructor(mainWindow) {
    this.mainWindow = mainWindow;
    this.stealthActive = getConfig('ui.stealthMode');
    this.originalBounds = null;
    this.isHidden = false;
    this.guidanceQueue = [];
    this.hotkeysOnly = false;
  }

  enableStealthMode() {
    console.log('🥷 Enabling stealth mode - Cue going invisible');
    
    this.stealthActive = true;
    setConfig('ui.stealthMode', true);
    
    // Store original window bounds
    if (this.mainWindow && !this.mainWindow.isDestroyed()) {
      this.originalBounds = this.mainWindow.getBounds();
      
      // Make window completely transparent and move off-screen
      this.hideWindow();
      
      // Enable keyboard-only mode
      this.hotkeysOnly = true;
      
      this.mainWindow.webContents.send('stealth-mode-changed', { 
        enabled: true,
        message: 'Cue is now invisible. Use Ctrl/Cmd + Shift + C for guidance.'
      });
    }
  }

  disableStealthMode() {
    console.log('👁️ Disabling stealth mode - Cue becoming visible');
    
    this.stealthActive = false;
    setConfig('ui.stealthMode', false);
    
    // Restore window visibility
    this.showWindow();
    this.hotkeysOnly = false;
    
    if (this.mainWindow && !this.mainWindow.isDestroyed()) {
      this.mainWindow.webContents.send('stealth-mode-changed', { 
        enabled: false,
        message: 'Cue is now visible'
      });
    }
  }

  hideWindow() {
    if (!this.mainWindow || this.mainWindow.isDestroyed()) return;
    
    // Multiple layers of invisibility
    this.mainWindow.setOpacity(0); // Completely transparent
    this.mainWindow.hide(); // Hidden from taskbar/dock
    this.mainWindow.setSkipTaskbar(true); // Don't show in task switcher
    this.mainWindow.setVisibleOnAllWorkspaces(false); // Not on all desktops
    
    // Move window far off-screen as extra precaution
    this.mainWindow.setBounds({
      x: -10000,
      y: -10000,
      width: 1,
      height: 1
    });
    
    this.isHidden = true;
  }

  showWindow() {
    if (!this.mainWindow || this.mainWindow.isDestroyed()) return;
    
    // Restore window properties
    if (this.originalBounds) {
      this.mainWindow.setBounds(this.originalBounds);
    }
    
    this.mainWindow.setOpacity(getConfig('ui.opacity'));
    this.mainWindow.setSkipTaskbar(false);
    this.mainWindow.show();
    this.isHidden = false;
  }

  // Show guidance in stealth mode (temporary popup)
  showStealthGuidance(guidance) {
    if (!this.stealthActive) return;
    
    // Create a temporary, minimal guidance window
    this.createStealthGuidanceWindow(guidance);
  }

  createStealthGuidanceWindow(guidance) {
    const { BrowserWindow, screen } = require('electron');
    const { width: screenWidth, height: screenHeight } = screen.getPrimaryDisplay().workAreaSize;
    
    // Create minimal guidance window
    const guidanceWindow = new BrowserWindow({
      width: 400,
      height: 300,
      x: screenWidth - 420, // Right edge
      y: 100, // Top area
      frame: false,
      transparent: true,
      alwaysOnTop: true,
      skipTaskbar: true,
      resizable: false,
      movable: false,
      focusable: false, // Won't steal focus
      show: false,
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true
      }
    });

    // Load minimal guidance HTML
    const guidanceHTML = this.generateGuidanceHTML(guidance);
    guidanceWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(guidanceHTML)}`);
    
    // Show temporarily
    guidanceWindow.show();
    guidanceWindow.setOpacity(0.95);
    
    // Auto-hide after 8 seconds
    setTimeout(() => {
      if (!guidanceWindow.isDestroyed()) {
        guidanceWindow.setOpacity(0);
        setTimeout(() => {
          if (!guidanceWindow.isDestroyed()) {
            guidanceWindow.close();
          }
        }, 300);
      }
    }, 8000);
    
    // Click to dismiss
    guidanceWindow.on('blur', () => {
      if (!guidanceWindow.isDestroyed()) {
        guidanceWindow.close();
      }
    });
  }

  generateGuidanceHTML(guidance) {
    const isDarkMode = true; // Always dark in stealth mode
    const bgColor = isDarkMode ? '#1a1a1a' : '#ffffff';
    const textColor = isDarkMode ? '#ffffff' : '#000000';
    const accentColor = '#4CAF50';
    
    let content = '';
    
    if (guidance.type === 'problem_guidance') {
      content = `
        <div style="margin-bottom: 12px;">
          <h3 style="color: ${accentColor}; margin: 0 0 8px 0; font-size: 14px;">
            🎯 ${guidance.platform?.toUpperCase() || 'CODING'} Problem Detected
          </h3>
          <p style="margin: 0; font-size: 12px; line-height: 1.4;">${guidance.message}</p>
        </div>
        ${guidance.analysis?.patterns?.length > 0 ? `
          <div style="background: rgba(76, 175, 80, 0.1); padding: 8px; border-radius: 4px; margin: 8px 0;">
            <strong style="color: ${accentColor}; font-size: 11px;">Patterns:</strong> 
            <span style="font-size: 11px;">${guidance.analysis.patterns.join(', ').replace(/_/g, ' ')}</span>
          </div>
        ` : ''}
      `;
    } else if (guidance.type === 'progressive_hint') {
      content = `
        <div style="margin-bottom: 12px;">
          <h3 style="color: #FF9800; margin: 0 0 8px 0; font-size: 14px;">💡 Hint #${guidance.hint_level}</h3>
          <p style="margin: 0; font-size: 12px; line-height: 1.4;">${guidance.message}</p>
        </div>
        ${guidance.code_template ? `
          <pre style="background: #0f0f0f; padding: 8px; border-radius: 4px; font-size: 10px; margin: 8px 0; overflow-x: auto;"><code>${guidance.code_template}</code></pre>
        ` : ''}
      `;
    }
    
    return `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <style>
            body {
              margin: 0;
              padding: 12px;
              background: ${bgColor}E6;
              color: ${textColor};
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
              font-size: 12px;
              border-radius: 8px;
              backdrop-filter: blur(10px);
              box-shadow: 0 4px 20px rgba(0,0,0,0.3);
              cursor: pointer;
            }
            body:hover {
              opacity: 0.7;
            }
          </style>
        </head>
        <body onclick="window.close()">
          ${content}
          <div style="text-align: right; margin-top: 8px; opacity: 0.6;">
            <small>Press Ctrl+Shift+C for more help • Click to dismiss</small>
          </div>
        </body>
      </html>
    `;
  }

  // Detect screen sharing (WebRTC, Zoom, Teams, etc.)
  detectScreenSharing() {
    // This would need platform-specific implementation
    // For now, we'll check for common screen sharing apps
    const { exec } = require('child_process');
    
    if (process.platform === 'darwin') {
      // macOS - check for screen sharing processes
      exec('ps aux | grep -i "screen.*share\\|zoom\\|teams\\|meet\\|webex"', (error, stdout) => {
        if (stdout && stdout.length > 0) {
          if (!this.stealthActive && getConfig('ui.hideDuringScreenShare')) {
            console.log('🔒 Screen sharing detected - hiding Cue');
            this.hideWindow();
          }
        }
      });
    }
  }

  // Global hotkey handlers for stealth mode
  registerStealthHotkeys() {
    // These will be registered in main.js
    return {
      'stealth-toggle': 'CommandOrControl+Shift+S', // Toggle stealth mode
      'stealth-guidance': 'CommandOrControl+Shift+C', // Show guidance in stealth
      'stealth-hint': 'CommandOrControl+Shift+H', // Force a hint
    };
  }

  handleStealthHotkey(action) {
    switch (action) {
      case 'stealth-toggle':
        if (this.stealthActive) {
          this.disableStealthMode();
        } else {
          this.enableStealthMode();
        }
        break;
        
      case 'stealth-guidance':
        if (this.stealthActive) {
          // Request guidance from screen monitor
          this.mainWindow?.webContents.send('request-stealth-guidance');
        }
        break;
        
      case 'stealth-hint':
        if (this.stealthActive) {
          // Force a progressive hint
          this.mainWindow?.webContents.send('request-stealth-hint');
        }
        break;
    }
  }

  // Check if we should be in stealth mode based on context
  autoStealthDetection() {
    // Auto-enable stealth when coding assessment sites are detected
    const assessmentSites = [
      'leetcode.com', 'hackerrank.com', 'codility.com', 'codesignal.com',
      'interviewing.io', 'pramp.com', 'hackerearth.com'
    ];
    
    // This would integrate with screen monitoring to detect URLs
    // For now, it's a placeholder for future enhancement
  }

  isStealthActive() {
    return this.stealthActive;
  }

  canShowUI() {
    if (this.stealthActive) return false;
    if (getConfig('ui.showOnlyWhenNeeded')) {
      return this.guidanceQueue.length > 0;
    }
    return true;
  }
}

function createStealthManager(mainWindow) {
  return new StealthModeManager(mainWindow);
}

function registerStealthHandlers(stealthManager) {
  ipcMain.handle('stealth:enable', () => {
    stealthManager.enableStealthMode();
    return true;
  });

  ipcMain.handle('stealth:disable', () => {
    stealthManager.disableStealthMode();
    return true;
  });

  ipcMain.handle('stealth:toggle', () => {
    if (stealthManager.isStealthActive()) {
      stealthManager.disableStealthMode();
    } else {
      stealthManager.enableStealthMode();
    }
    return stealthManager.isStealthActive();
  });

  ipcMain.on('stealth:show-guidance', (_event, guidance) => {
    stealthManager.showStealthGuidance(guidance);
  });

  return stealthManager.registerStealthHotkeys();
}

module.exports = {
  StealthModeManager,
  createStealthManager,
  registerStealthHandlers
};