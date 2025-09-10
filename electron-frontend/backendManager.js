// Backend Manager - Seamlessly integrate Python backend with Electron
const { spawn, exec } = require('child_process');
const path = require('path');
const fs = require('fs');
const { app } = require('electron');

class BackendManager {
  constructor() {
    this.backendProcess = null;
    this.isStarting = false;
    this.isRunning = false;
    this.startAttempts = 0;
    this.maxStartAttempts = 3;
    this.backendPath = path.join(__dirname, '..', 'backend');
    this.pythonCmd = this.detectPython();
  }

  // Detect available Python command
  detectPython() {
    const pythonCommands = ['python3', 'python', 'py'];
    
    for (const cmd of pythonCommands) {
      try {
        const { execSync } = require('child_process');
        execSync(`${cmd} --version`, { stdio: 'ignore' });
        console.log(`✅ Found Python: ${cmd}`);
        return cmd;
      } catch (error) {
        continue;
      }
    }
    
    console.error('❌ No Python found. Please install Python 3.8+');
    return null;
  }

  // Check if backend directory exists
  checkBackendExists() {
    if (!fs.existsSync(this.backendPath)) {
      console.error(`❌ Backend directory not found: ${this.backendPath}`);
      return false;
    }
    
    const mainPyPath = path.join(this.backendPath, 'main.py');
    if (!fs.existsSync(mainPyPath)) {
      console.error(`❌ Backend main.py not found: ${mainPyPath}`);
      return false;
    }
    
    return true;
  }

  // Install Python dependencies
  async installDependencies() {
    if (!this.pythonCmd) {
      throw new Error('Python not found');
    }

    console.log('🔧 Installing Python dependencies...');
    
    return new Promise((resolve, reject) => {
      const requirementsPath = path.join(this.backendPath, 'requirements.txt');
      
      if (!fs.existsSync(requirementsPath)) {
        console.log('⚠️ No requirements.txt found, skipping dependency installation');
        resolve();
        return;
      }

      const installProcess = spawn(this.pythonCmd, ['-m', 'pip', 'install', '-r', 'requirements.txt'], {
        cwd: this.backendPath,
        stdio: ['ignore', 'pipe', 'pipe']
      });

      let output = '';
      let errorOutput = '';

      installProcess.stdout?.on('data', (data) => {
        output += data.toString();
      });

      installProcess.stderr?.on('data', (data) => {
        errorOutput += data.toString();
      });

      installProcess.on('close', (code) => {
        if (code === 0) {
          console.log('✅ Dependencies installed successfully');
          resolve();
        } else {
          console.error('❌ Failed to install dependencies:', errorOutput);
          reject(new Error(`Dependency installation failed: ${errorOutput}`));
        }
      });

      installProcess.on('error', (error) => {
        console.error('❌ Failed to start dependency installation:', error);
        reject(error);
      });
    });
  }

  // Start the Python backend
  async startBackend() {
    if (this.isStarting || this.isRunning) {
      console.log('Backend is already starting or running');
      return;
    }

    if (!this.pythonCmd) {
      throw new Error('Python not found. Please install Python 3.8+');
    }

    if (!this.checkBackendExists()) {
      throw new Error('Backend files not found');
    }

    this.isStarting = true;
    this.startAttempts++;

    try {
      // Try to install dependencies first
      await this.installDependencies();
      
      console.log('🚀 Starting Python backend...');
      
      // Start the backend process
      this.backendProcess = spawn(this.pythonCmd, ['main.py'], {
        cwd: this.backendPath,
        stdio: ['ignore', 'pipe', 'pipe'],
        env: {
          ...process.env,
          // Ensure fast, low-latency STT by default; user can override via env
          WHISPER_MODEL_SIZE: process.env.WHISPER_MODEL_SIZE || 'tiny',
          WHISPER_DEVICE: process.env.WHISPER_DEVICE || 'cpu',
          WHISPER_COMPUTE_TYPE: process.env.WHISPER_COMPUTE_TYPE || 'int8'
        }
      });

      // Handle backend output
      this.backendProcess.stdout?.on('data', (data) => {
        const output = data.toString();
        console.log('🐍 Backend:', output.trim());
        
        // Check if backend is ready
        if (output.includes('Uvicorn running') || output.includes('Application startup complete')) {
          this.isRunning = true;
          this.isStarting = false;
          console.log('✅ Backend is ready!');
        }
      });

      this.backendProcess.stderr?.on('data', (data) => {
        const error = data.toString();
        console.log('🐍 Backend:', error.trim());
        
        // Don't treat warnings as errors
        if (error.includes('WARNING') || error.includes('DeprecationWarning')) {
          return;
        }
        
        // Critical errors
        if (error.includes('ERROR') || error.includes('CRITICAL')) {
          console.error('❌ Backend error:', error);
        }
      });

      this.backendProcess.on('close', (code) => {
        this.isRunning = false;
        this.isStarting = false;
        
        console.log(`🐍 Backend process exited with code ${code}`);
        
        if (code !== 0 && this.startAttempts < this.maxStartAttempts) {
          console.log(`🔄 Restarting backend (attempt ${this.startAttempts + 1}/${this.maxStartAttempts})`);
          setTimeout(() => this.startBackend(), 2000);
        } else if (code !== 0) {
          console.error('❌ Backend failed to start after maximum attempts');
        }
      });

      this.backendProcess.on('error', (error) => {
        this.isRunning = false;
        this.isStarting = false;
        console.error('❌ Failed to start backend process:', error);
        
        if (this.startAttempts < this.maxStartAttempts) {
          console.log(`🔄 Retrying backend start (attempt ${this.startAttempts + 1}/${this.maxStartAttempts})`);
          setTimeout(() => this.startBackend(), 2000);
        }
      });

      // Set a timeout to mark as failed if it doesn't start in reasonable time
      setTimeout(() => {
        if (this.isStarting) {
          console.log('⏰ Backend is taking longer than expected to start...');
        }
      }, 10000);

    } catch (error) {
      this.isStarting = false;
      console.error('❌ Failed to start backend:', error);
      throw error;
    }
  }

  // Stop the backend
  stopBackend() {
    if (this.backendProcess && !this.backendProcess.killed) {
      console.log('🛑 Stopping backend...');
      
      // Try graceful shutdown first
      this.backendProcess.kill('SIGTERM');
      
      // Force kill after 5 seconds if still running
      setTimeout(() => {
        if (this.backendProcess && !this.backendProcess.killed) {
          console.log('🔪 Force stopping backend...');
          this.backendProcess.kill('SIGKILL');
        }
      }, 5000);
    }
    
    this.backendProcess = null;
    this.isRunning = false;
    this.isStarting = false;
  }

  // Get backend status
  getStatus() {
    return {
      isRunning: this.isRunning,
      isStarting: this.isStarting,
      hasProcess: !!this.backendProcess,
      pythonFound: !!this.pythonCmd,
      backendExists: this.checkBackendExists(),
      startAttempts: this.startAttempts
    };
  }

  // Health check the backend
  async healthCheck() {
    try {
      const response = await fetch('http://127.0.0.1:8001/health');  // Changed from 8000 to 8001
      const data = await response.json();
      return data.status === 'ok';
    } catch (error) {
      return false;
    }
  }

  // Restart backend
  async restartBackend() {
    console.log('🔄 Restarting backend...');
    this.stopBackend();
    
    // Wait a moment for cleanup
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    return this.startBackend();
  }

  // Auto-monitor backend health
  startHealthMonitoring() {
    setInterval(async () => {
      if (this.isRunning) {
        const isHealthy = await this.healthCheck();
        if (!isHealthy && this.startAttempts < this.maxStartAttempts) {
          console.log('💔 Backend health check failed, restarting...');
          await this.restartBackend();
        }
      }
    }, 30000); // Check every 30 seconds
  }
}

// Create and export a singleton instance
const backendManager = new BackendManager();

// Auto-cleanup on app exit
app.on('before-quit', () => {
  console.log('🧹 Cleaning up backend before app exit...');
  backendManager.stopBackend();
});

app.on('window-all-closed', () => {
  backendManager.stopBackend();
});

module.exports = backendManager;
