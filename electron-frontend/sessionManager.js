const Store = require('electron-store');
const { ipcMain } = require('electron');

const sessionStore = new Store();
let currentSession = null;
let ipcRegistered = false;

function startSession(type = 'interview') {
  const session = {
    id: Date.now(),
    type,
    startedAt: new Date().toISOString()
  };
  currentSession = session;
  sessionStore.set('lastSession', session);
  return session;
}

function stopSession() {
  if (currentSession) {
    currentSession.endedAt = new Date().toISOString();
    sessionStore.set(`session-${currentSession.id}`, currentSession);
    currentSession = null;
  }
}

function registerSessionHandlers() {
  if (ipcRegistered) return; // Prevent double registration
  ipcRegistered = true;

  ipcMain.handle('session:start', (_e, type) => {
    return startSession(type);
  });

  ipcMain.handle('session:stop', () => {
    stopSession();
    return true;
  });
}

module.exports = {
  startSession,
  stopSession,
  registerSessionHandlers
};