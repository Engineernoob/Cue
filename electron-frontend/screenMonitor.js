// Real-time Screen Monitoring for Coding Problems
const { desktopCapturer, screen } = require('electron');
const { getConfig } = require('./config');

class ScreenMonitor {
  constructor(mainWindow) {
    this.mainWindow = mainWindow;
    this.isMonitoring = false;
    this.monitoringInterval = null;
    this.lastScreenshot = null;
    this.currentProblem = null;
    this.solutionProgress = {
      stage: 'understanding', // understanding, planning, implementing, testing, optimizing
      hints_given: 0,
      stuck_time: 0,
      last_progress: Date.now()
    };
    this.captureSettings = {
      interval: 3000, // 3 seconds - balance between responsiveness and performance
      quality: 0.8,
      maxWidth: 1920,
      maxHeight: 1080
    };
  }

  async startMonitoring() {
    if (this.isMonitoring) return;
    
    this.isMonitoring = true;
    console.log('🔍 Starting screen monitoring for coding problems...');
    
    // Initial capture
    await this.captureAndAnalyze();
    
    // Set up periodic monitoring
    this.monitoringInterval = setInterval(async () => {
      await this.captureAndAnalyze();
    }, this.captureSettings.interval);

    this.mainWindow?.webContents.send('screen-monitoring-status', { 
      monitoring: true,
      message: 'AI is now watching for coding problems on your screen'
    });
  }

  stopMonitoring() {
    if (!this.isMonitoring) return;
    
    this.isMonitoring = false;
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    
    console.log('⏹️ Stopped screen monitoring');
    this.mainWindow?.webContents.send('screen-monitoring-status', { 
      monitoring: false,
      message: 'Screen monitoring stopped'
    });
  }

  async captureAndAnalyze() {
    try {
      // Get all available screens
      const sources = await desktopCapturer.getSources({
        types: ['screen'],
        thumbnailSize: {
          width: this.captureSettings.maxWidth,
          height: this.captureSettings.maxHeight
        }
      });

      const primaryScreen = sources[0];
      if (!primaryScreen) return;

      const screenshot = primaryScreen.thumbnail.toDataURL();
      
      // Skip if screenshot hasn't changed significantly (avoid unnecessary processing)
      if (this.lastScreenshot === screenshot) return;
      
      this.lastScreenshot = screenshot;

      // Convert to base64 for backend processing
      const base64Data = screenshot.split(',')[1];
      
      // Send to backend for analysis
      this.sendToBackendForAnalysis(base64Data);

    } catch (error) {
      console.error('Error capturing screen:', error);
    }
  }

  sendToBackendForAnalysis(base64Data) {
    // Send via WebSocket to backend for OCR and problem analysis
    this.mainWindow?.webContents.send('send-screen-for-analysis', {
      imageData: base64Data,
      timestamp: Date.now(),
      monitoring_context: 'coding_problem_detection'
    });
  }

  // Called when backend detects a coding problem
  onProblemDetected(problemData) {
    console.log('🧮 Coding problem detected:', problemData.analysis);
    
    this.currentProblem = {
      text: problemData.text,
      analysis: problemData.analysis,
      detected_at: Date.now(),
      platform: this.detectPlatform(problemData.text)
    };

    this.solutionProgress = {
      stage: 'understanding',
      hints_given: 0,
      stuck_time: 0,
      last_progress: Date.now()
    };

    // Immediately provide initial guidance
    this.provideInitialGuidance();
  }

  detectPlatform(problemText) {
    const text = problemText.toLowerCase();
    const platforms = {
      'leetcode': ['leetcode', 'submissions', 'runtime:', 'memory:'],
      'hackerrank': ['hackerrank', 'sample input', 'sample output', 'function signature'],
      'codility': ['codility', 'lesson', 'task score', 'correctness'],
      'codesignal': ['codesignal', 'arcade', 'interview practice'],
      'geeksforgeeks': ['geeksforgeeks', 'gfg', 'practice problems']
    };

    for (const [platform, indicators] of Object.entries(platforms)) {
      if (indicators.some(indicator => text.includes(indicator))) {
        return platform;
      }
    }
    return 'unknown';
  }

  provideInitialGuidance() {
    const problem = this.currentProblem;
    if (!problem) return;

    const guidance = {
      type: 'problem_guidance',
      stage: 'initial_detection',
      platform: problem.platform,
      message: this.generateInitialMessage(),
      analysis: problem.analysis,
      next_steps: this.getInitialSteps(),
      timestamp: Date.now()
    };

    this.mainWindow?.webContents.send('coding-guidance', guidance);
  }

  generateInitialMessage() {
    const problem = this.currentProblem;
    const platform = problem.platform;
    
    const platformMessages = {
      'leetcode': '🎯 LeetCode problem detected! Let\'s break this down step by step.',
      'hackerrank': '🚀 HackerRank challenge spotted! Focus on the input/output format first.',
      'codility': '⚡ Codility assessment detected! Correctness and efficiency are key here.',
      'unknown': '🧮 Coding problem detected! Let\'s solve this together.'
    };

    const baseMessage = platformMessages[platform] || platformMessages['unknown'];
    
    if (problem.analysis.patterns.length > 0) {
      const patterns = problem.analysis.patterns.join(', ').replace(/_/g, ' ');
      return `${baseMessage} I can see this involves ${patterns} - perfect, I know exactly how to help you approach this!`;
    }
    
    return `${baseMessage} Let me help you understand the problem first.`;
  }

  getInitialSteps() {
    const problem = this.currentProblem;
    const analysis = problem.analysis;
    
    const steps = [
      "📖 Read the problem statement carefully (I'll help you understand)",
      "🎯 Identify the input and output format",
      "💡 Work through the given examples manually"
    ];

    if (analysis.patterns.length > 0) {
      const mainPattern = analysis.patterns[0].replace('_', ' ');
      steps.push(`🔧 This looks like a ${mainPattern} problem - I'll guide you through the approach`);
    } else {
      steps.push("🚀 Start with a brute force approach (I'll help you optimize later)");
    }

    steps.push("✅ Test your solution with the examples");

    return steps;
  }

  // Monitor user progress and provide contextual help
  checkProgress() {
    const now = Date.now();
    const timeSinceLastProgress = now - this.solutionProgress.last_progress;
    
    // If user seems stuck (no progress for 5+ minutes)
    if (timeSinceLastProgress > 300000 && this.solutionProgress.hints_given < 3) {
      this.provideProgressiveHint();
    }
    
    // Check for signs of stress/overwhelm
    if (timeSinceLastProgress > 600000) { // 10 minutes
      this.provideEncouragement();
    }
  }

  provideProgressiveHint() {
    if (!this.currentProblem) return;
    
    const hintLevel = this.solutionProgress.hints_given;
    const analysis = this.currentProblem.analysis;
    
    let hint = '';
    let codeTemplate = '';

    if (analysis.patterns.length > 0) {
      const pattern = analysis.patterns[0];
      hint = this.getPatternHint(pattern, hintLevel);
      codeTemplate = this.getPatternTemplate(pattern, hintLevel);
    } else {
      hint = this.getGeneralHint(hintLevel);
    }

    const guidance = {
      type: 'progressive_hint',
      hint_level: hintLevel + 1,
      message: hint,
      code_template: codeTemplate,
      encouragement: this.getEncouragement(hintLevel),
      timestamp: Date.now()
    };

    this.solutionProgress.hints_given++;
    this.solutionProgress.last_progress = Date.now();

    this.mainWindow?.webContents.send('coding-guidance', guidance);
  }

  getPatternHint(pattern, level) {
    const hints = {
      'two_pointers': [
        "Try using two pointers: one starting from the beginning, one from the end.",
        "Move your pointers based on comparing the values they point to.",
        "Here's the pattern: left = 0, right = len(array) - 1, then move them inward based on your condition."
      ],
      'sliding_window': [
        "Think about maintaining a 'window' that slides across your data.",
        "Expand the window by moving the right pointer, contract by moving the left.",
        "The key insight: when should you expand vs contract the window?"
      ],
      'dynamic_programming': [
        "This looks like a problem where you can break it into smaller subproblems.",
        "Think: what's the recurrence relation? dp[i] = ?",
        "Start with the base case, then build up your solution step by step."
      ],
      'dfs_bfs': [
        "Tree/graph problems often need traversal. Are you going depth-first or breadth-first?",
        "DFS uses recursion/stack, BFS uses a queue. Which fits your problem better?",
        "Don't forget to track visited nodes to avoid cycles!"
      ]
    };

    return hints[pattern]?.[level] || "Break this problem down into smaller parts. What's the first step?";
  }

  getPatternTemplate(pattern, level) {
    const templates = {
      'two_pointers': [
        '',
        `# Two pointers approach
left, right = 0, len(arr) - 1
while left < right:
    # Compare arr[left] and arr[right]
    # Move pointers based on condition`,
        `def two_pointer_solution(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        if condition_met(arr[left], arr[right]):
            return True  # or your result
        elif arr[left] + arr[right] < target:
            left += 1
        else:
            right -= 1
    return False`
      ],
      'sliding_window': [
        '',
        `# Sliding window pattern
left = 0
for right in range(len(arr)):
    # Expand window
    while window_invalid():
        # Contract window
        left += 1`,
        `def sliding_window_solution(arr):
    left = 0
    max_length = 0
    window_sum = 0
    
    for right in range(len(arr)):
        window_sum += arr[right]
        
        while window_sum > target:
            window_sum -= arr[left]
            left += 1
            
        max_length = max(max_length, right - left + 1)
    
    return max_length`
      ]
    };

    return templates[pattern]?.[level] || '';
  }

  getGeneralHint(level) {
    const hints = [
      "Let's start simple: what happens with the smallest possible input?",
      "Try working through the example step by step. What pattern do you see?",
      "Sometimes the brute force solution reveals the optimized approach. What would brute force look like here?"
    ];

    return hints[level] || "You're making progress! Keep thinking about the core logic.";
  }

  getEncouragement(hintLevel) {
    const encouragements = [
      "You've got this! Every expert was once a beginner.",
      "Problem-solving is a skill - you're building it right now.",
      "It's normal to get stuck. The thinking process is what matters most.",
      "You're doing great! Sometimes the best insights come after sitting with a problem."
    ];

    return encouragements[hintLevel] || "Keep going - you're closer than you think!";
  }

  provideEncouragement() {
    const guidance = {
      type: 'encouragement',
      message: "I notice you've been working on this for a while. That's totally normal! Sometimes the best approach is to step back and think about the problem differently.",
      suggestions: [
        "Try explaining the problem out loud in your own words",
        "Draw out the problem with a simple example",
        "Take a 2-minute break to reset your mind",
        "Look for a simpler version of this problem to solve first"
      ],
      timestamp: Date.now()
    };

    this.mainWindow?.webContents.send('coding-guidance', guidance);
  }

  // Update progress when user makes changes
  updateProgress(stage) {
    this.solutionProgress.stage = stage;
    this.solutionProgress.last_progress = Date.now();
    this.solutionProgress.stuck_time = 0;
  }

  // Get solution walkthrough (when user explicitly asks)
  provideSolutionWalkthrough() {
    if (!this.currentProblem) return;

    const walkthrough = this.generateSolutionWalkthrough();
    
    const guidance = {
      type: 'solution_walkthrough',
      message: "Here's how to approach this problem step by step:",
      walkthrough: walkthrough,
      warning: "This is the complete solution approach. Make sure you understand each step!",
      timestamp: Date.now()
    };

    this.mainWindow?.webContents.send('coding-guidance', guidance);
  }

  generateSolutionWalkthrough() {
    const analysis = this.currentProblem.analysis;
    
    if (analysis.patterns.length === 0) {
      return {
        approach: "Brute force approach",
        steps: [
          "1. Understand what the problem is asking for",
          "2. Try all possible combinations/solutions",
          "3. Check each one to see if it meets the criteria",
          "4. Return the best result found"
        ],
        complexity: "This might not be the most efficient, but it's a good starting point!",
        next_step: "Once this works, think about how to optimize it."
      };
    }

    const mainPattern = analysis.patterns[0];
    return this.getPatternWalkthrough(mainPattern);
  }

  getPatternWalkthrough(pattern) {
    const walkthroughs = {
      'two_pointers': {
        approach: "Two Pointers Technique",
        steps: [
          "1. Set up two pointers: left at start (0), right at end (length-1)",
          "2. While left < right, compare the values at both pointers",
          "3. Move the appropriate pointer based on your comparison",
          "4. Continue until pointers meet or condition is satisfied"
        ],
        complexity: "Time: O(n), Space: O(1) - very efficient!",
        next_step: "This pattern works great for sorted arrays and palindrome problems."
      },
      'sliding_window': {
        approach: "Sliding Window Technique",
        steps: [
          "1. Initialize left pointer at 0, iterate right pointer",
          "2. Expand window by including arr[right] in your calculation",
          "3. If window becomes invalid, contract from left until valid",
          "4. Track the best window seen so far"
        ],
        complexity: "Time: O(n), Space: O(1) - each element visited at most twice",
        next_step: "Perfect for subarray/substring problems with constraints."
      }
    };

    return walkthroughs[pattern] || {
      approach: "Step-by-step approach",
      steps: [
        "1. Break down the problem into smaller parts",
        "2. Solve each part individually", 
        "3. Combine the parts into a complete solution",
        "4. Test with the given examples"
      ],
      complexity: "Focus on correctness first, then optimize",
      next_step: "Every problem has a pattern - you'll start recognizing them!"
    };
  }
}

function createScreenMonitor(mainWindow) {
  return new ScreenMonitor(mainWindow);
}

module.exports = {
  ScreenMonitor,
  createScreenMonitor
};