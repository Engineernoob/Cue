# 🚀 Cue Setup Guide

## One-Command Setup

Cue now automatically manages its Python backend! Just run:

```bash
cd electron-frontend
npm install
npm start
```

That's it! 🎉

## What Happens Automatically

1. **Python Detection**: Automatically finds Python 3 (`python3`, `python`, or `py`)
2. **Dependency Installation**: Installs backend requirements automatically  
3. **Backend Launch**: Starts the Python backend server
4. **Health Monitoring**: Monitors backend and restarts if needed
5. **Graceful Shutdown**: Stops backend when app closes

## Requirements

- **Node.js** (for Electron frontend)
- **Python 3.8+** (automatically detected)
- **Internet connection** (for initial dependency installation)

## Troubleshooting

### Python Not Found
```bash
# Install Python 3
# macOS:
brew install python3

# Ubuntu/Debian:
sudo apt install python3 python3-pip

# Windows:
# Download from python.org
```

### Port Already in Use
If port 8000 is busy:
```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9

# Or change port in config.js
```

### Backend Won't Start
```bash
# Test backend manually
npm run test-backend

# Run backend separately for debugging
npm run backend-only
```

### Dependencies Won't Install
```bash
# Install Python dependencies manually
cd ../backend
pip3 install -r requirements.txt
```

## Advanced Configuration

### Custom Python Path
Edit `electron-frontend/backendManager.js`:
```js
// Line ~15
this.pythonCmd = '/path/to/your/python3';
```

### Custom Backend Port
Edit `electron-frontend/config.js`:
```js
backend: {
  wsUrl: 'ws://127.0.0.1:9000/ws', // Change port here
}
```

### Development Mode
```bash
npm run dev  # Enables additional logging
```

## What's Different Now

**Before** (2 terminals required):
```bash
# Terminal 1
cd backend && python3 main.py

# Terminal 2  
cd electron-frontend && npm start
```

**Now** (1 command):
```bash
cd electron-frontend && npm start
```

## Status Indicators

- 🟢 **Green dot**: Backend connected and healthy
- 🔵 **Blue dot**: AI monitoring active  
- 🟡 **Yellow dot**: Backend starting/reconnecting
- 🔴 **Red dot**: Backend error/disconnected

## First Run

On first startup, you'll see:
1. "Backend starting..." notification
2. Python dependency installation (30-60 seconds)
3. "AI backend connected" when ready
4. All features available!

## Production Build

```bash
npm run build  # Creates packaged app with integrated backend
```

---

**Your invisible AI coding mentor is now plug-and-play! 🥷✨**