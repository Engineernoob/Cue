// Neurodivergent support features for Cue
const { ipcMain, BrowserWindow } = require('electron');
const { getConfig, setConfig } = require('./config');

class NeurodivergentSupport {
  constructor(mainWindow) {
    this.mainWindow = mainWindow;
    this.timers = {
      focus: null,
      break: null,
      hyperfocus: null,
      breathing: null
    };
    this.state = {
      sessionStartTime: null,
      lastBreakTime: null,
      typingActivity: [],
      stressLevel: 'normal', // normal, medium, high
      currentMode: 'default' // default, coding, interview, debugging
    };
  }

  // ADHD Support Features
  startADHDSupport() {
    const config = getConfig('neurodivergent.adhd');
    
    if (config.focusReminders) {
      this.startFocusReminders();
    }
    
    if (config.breakReminders) {
      this.startBreakReminders();
    }
    
    if (config.hyperfocusWarnings) {
      this.startHyperfocusMonitoring();
    }
  }

  startFocusReminders() {
    const interval = getConfig('neurodivergent.adhd.reminderInterval');
    
    this.timers.focus = setInterval(() => {
      const prompts = [
        "How's your focus? Take a moment to check in with yourself.",
        "Remember to save your work regularly.",
        "Are you still on track with your main goal?",
        "Quick focus check: Is this still the most important task?"
      ];
      
      this.sendGentleReminder(
        prompts[Math.floor(Math.random() * prompts.length)],
        'focus-reminder'
      );
    }, interval);
  }

  startBreakReminders() {
    this.timers.break = setInterval(() => {
      const timeSinceBreak = Date.now() - (this.state.lastBreakTime || this.state.sessionStartTime);
      
      if (timeSinceBreak > 3600000) { // 1 hour
        this.sendGentleReminder(
          "You've been working hard! Time for a 5-minute break to recharge.",
          'break-reminder'
        );
      }
    }, 600000); // Check every 10 minutes
  }

  startHyperfocusMonitoring() {
    this.timers.hyperfocus = setInterval(() => {
      const sessionTime = Date.now() - this.state.sessionStartTime;
      
      if (sessionTime > 7200000) { // 2 hours of continuous work
        this.sendGentleReminder(
          "You've been in the zone for 2+ hours! Remember to eat, hydrate, and move around.",
          'hyperfocus-warning',
          'high'
        );
      }
    }, 900000); // Check every 15 minutes
  }

  // Autism Support Features
  provideStructuredSupport(context) {
    const socialCues = this.detectSocialCues(context);
    const structuredResponse = this.generateStructuredResponse(context);
    
    if (socialCues.length > 0) {
      this.mainWindow?.webContents.send('social-cue-help', {
        cues: socialCues,
        suggestions: this.getSocialSuggestions(socialCues)
      });
    }
    
    return structuredResponse;
  }

  detectSocialCues(context) {
    const cues = [];
    const text = context.text?.toLowerCase() || '';
    
    if (text.includes('what do you think') || text.includes('your opinion')) {
      cues.push('opinion_requested');
    }
    
    if (text.includes('any questions') || text.includes('questions for us')) {
      cues.push('questions_expected');
    }
    
    if (text.includes('tell us about yourself') || text.includes('introduce yourself')) {
      cues.push('self_introduction');
    }
    
    if (text.includes('walk through') || text.includes('explain your approach')) {
      cues.push('detailed_explanation');
    }
    
    return cues;
  }

  getSocialSuggestions(cues) {
    const suggestions = {
      opinion_requested: "It's okay to take a moment to think. You can start with 'That's an interesting question...'",
      questions_expected: "Prepare 2-3 thoughtful questions about the role, team, or technical challenges.",
      self_introduction: "Use a structured approach: background, current role, why you're interested in this position.",
      detailed_explanation: "Break it down step by step. It's fine to say 'Let me walk through my thinking process.'"
    };
    
    return cues.map(cue => suggestions[cue] || '');
  }

  generateStructuredResponse(context) {
    return {
      structure: "1. Understand → 2. Plan → 3. Execute → 4. Verify",
      breakdown: this.breakDownProblem(context.text),
      timeEstimate: this.estimateTime(context.text)
    };
  }

  breakDownProblem(text) {
    if (!text) return [];
    
    // Simple problem breakdown for coding problems
    if (this.isCodeRelated(text)) {
      return [
        "1. Read and understand the problem requirements",
        "2. Identify input/output format and constraints", 
        "3. Think of approach and data structures needed",
        "4. Write pseudocode or outline",
        "5. Implement the solution",
        "6. Test with examples and edge cases"
      ];
    }
    
    return [
      "1. Break down the main question or task",
      "2. Identify what information you need",
      "3. Organize your thoughts logically", 
      "4. Present your answer clearly"
    ];
  }

  // Anxiety Support Features
  startAnxietySupport() {
    const config = getConfig('neurodivergent.anxiety');
    
    if (config.breathingReminders) {
      this.startBreathingReminders();
    }
    
    this.monitorStressLevel();
  }

  startBreathingReminders() {
    this.timers.breathing = setInterval(() => {
      if (this.state.stressLevel === 'high') {
        this.sendGentleReminder(
          "Take 3 deep breaths: In for 4, hold for 4, out for 4. You've got this.",
          'breathing-reminder',
          'urgent'
        );
      }
    }, 300000); // Every 5 minutes when stressed
  }

  monitorStressLevel() {
    // Monitor typing patterns for stress indicators
    const recentActivity = this.state.typingActivity.slice(-10);
    
    if (recentActivity.length >= 5) {
      const avgSpeed = recentActivity.reduce((sum, act) => sum + act.speed, 0) / recentActivity.length;
      const pauseCount = recentActivity.filter(act => act.pauseDuration > 3000).length;
      
      if (avgSpeed < 20 && pauseCount > 3) {
        this.state.stressLevel = 'high';
        this.offerAnxietySupport();
      } else if (pauseCount > 1) {
        this.state.stressLevel = 'medium';
      } else {
        this.state.stressLevel = 'normal';
      }
    }
  }

  offerAnxietySupport() {
    const supportMessages = [
      "Feeling stuck? It's normal in coding interviews. Take a breath and think out loud.",
      "Remember: they want to see your thought process, not just the perfect answer.",
      "You're doing better than you think. Trust your instincts.",
      "It's okay to ask for clarification or hint. That shows good communication skills."
    ];
    
    this.sendGentleReminder(
      supportMessages[Math.floor(Math.random() * supportMessages.length)],
      'anxiety-support'
    );
  }

  // Utility Methods
  sendGentleReminder(message, type, priority = 'normal') {
    this.mainWindow?.webContents.send('neurodivergent-support', {
      message,
      type,
      priority,
      timestamp: Date.now()
    });
  }

  isCodeRelated(text) {
    const codeKeywords = [
      'algorithm', 'function', 'variable', 'array', 'loop', 'condition',
      'debug', 'error', 'syntax', 'coding', 'programming', 'leetcode'
    ];
    
    return codeKeywords.some(keyword => 
      text.toLowerCase().includes(keyword)
    );
  }

  recordTypingActivity(speed, pauseDuration) {
    this.state.typingActivity.push({
      timestamp: Date.now(),
      speed,
      pauseDuration
    });
    
    // Keep only last 20 entries
    if (this.state.typingActivity.length > 20) {
      this.state.typingActivity.shift();
    }
  }

  startSession(mode = 'default') {
    this.state.sessionStartTime = Date.now();
    this.state.lastBreakTime = Date.now();
    this.state.currentMode = mode;
    
    const config = getConfig('neurodivergent');
    
    if (config.adhd.focusReminders || config.adhd.breakReminders || config.adhd.hyperfocusWarnings) {
      this.startADHDSupport();
    }
    
    if (config.anxiety.breathingReminders || config.anxiety.groundingTechniques) {
      this.startAnxietySupport();
    }
  }

  stopSession() {
    Object.values(this.timers).forEach(timer => {
      if (timer) clearInterval(timer);
    });
    
    this.state.sessionStartTime = null;
    this.state.typingActivity = [];
    this.state.stressLevel = 'normal';
  }

  estimateTime(text) {
    // Simple time estimation for coding problems
    if (this.isCodeRelated(text)) {
      if (text.includes('easy') || text.length < 200) return "15-20 minutes";
      if (text.includes('hard') || text.length > 500) return "45-60 minutes";
      return "25-35 minutes";
    }
    return "Take your time";
  }
}

function registerNeurodivergentHandlers(mainWindow) {
  const support = new NeurodivergentSupport(mainWindow);
  
  ipcMain.handle('neurodivergent:start-session', (_event, mode) => {
    support.startSession(mode);
    return true;
  });
  
  ipcMain.handle('neurodivergent:stop-session', () => {
    support.stopSession();
    return true;
  });
  
  ipcMain.on('neurodivergent:typing-activity', (_event, data) => {
    support.recordTypingActivity(data.speed, data.pauseDuration);
  });
  
  return support;
}

module.exports = {
  NeurodivergentSupport,
  registerNeurodivergentHandlers
};