// Configuration management for Cue
const Store = require('electron-store');

const configStore = new Store({
  defaults: {
    backend: {
      wsUrl: 'ws://127.0.0.1:8001/ws',  // Changed from 8000 to 8001
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
      position: { x: 'center', y: 'bottom-40' },
      stealthMode: false, // When true, completely invisible like Cluely
      showOnlyWhenNeeded: false, // Only show UI when guidance is needed
      hideDuringScreenShare: true // Hide when screen sharing is detected
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
      algorithmHints: true,
      answerStyle: 'PAR' // Preferred answer structure: PAR, STAR, or SCQA
    },
    hotkeys: {
      pushToTalk: {
        start: 'CommandOrControl+Shift+Space',
        stop: 'CommandOrControl+Shift+K'
      },
      autoCoachToggle: 'CommandOrControl+Shift+A',
      answerStyleCycle: 'CommandOrControl+Shift+R'
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
