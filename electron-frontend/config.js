// Configuration management for Cue
const Store = require('electron-store');

const configStore = new Store({
  defaults: {
    backend: {
      wsUrl: 'ws://127.0.0.1:8000/ws',
      reconnectDelay: 5000,
      maxReconnectAttempts: 10
    },
    audio: {
      chunkInterval: 500, // ms - longer for better performance
      enabled: true
    },
    screen: {
      captureInterval: 2000, // ms - less frequent screen capture
      enabled: true
    },
    ui: {
      opacity: 0.9,
      alwaysOnTop: true,
      position: { x: 'center', y: 30 }
    },
    neurodivergent: {
      adhd: {
        focusReminders: true,
        breakReminders: true,
        hyperfocusWarnings: true,
        reminderInterval: 1800000 // 30 minutes
      },
      autism: {
        socialCueHelp: true,
        structuredResponses: true,
        stressTolerance: 'medium'
      },
      anxiety: {
        groundingTechniques: true,
        confidenceBoosts: true,
        breathingReminders: true
      }
    },
    coaching: {
      codingMode: true,
      interviewMode: true,
      debuggingHelp: true,
      algorithmHints: true
    }
  }
});

function getConfig(key = null) {
  return key ? configStore.get(key) : configStore.store;
}

function setConfig(key, value) {
  configStore.set(key, value);
}

function resetConfig() {
  configStore.clear();
}

module.exports = {
  getConfig,
  setConfig,
  resetConfig
};