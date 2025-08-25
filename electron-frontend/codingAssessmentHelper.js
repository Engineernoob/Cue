// Coding Assessment Platform Helper for Cue
const { ipcMain } = require('electron');
const { getConfig } = require('./config');

class CodingAssessmentHelper {
  constructor(mainWindow) {
    this.mainWindow = mainWindow;
    this.currentPlatform = null;
    this.problemContext = {
      title: '',
      difficulty: '',
      topics: [],
      timeLimit: null,
      startTime: null
    };
    this.algorithmPatterns = this.initializePatterns();
  }

  // Detect coding assessment platforms
  detectPlatform(url, title = '') {
    const platforms = {
      'leetcode.com': 'leetcode',
      'hackerrank.com': 'hackerrank', 
      'codility.com': 'codility',
      'codesignal.com': 'codesignal',
      'codingbat.com': 'codingbat',
      'hackerearth.com': 'hackerearth',
      'geeksforgeeks.org': 'geeksforgeeks',
      'codeforces.com': 'codeforces',
      'atcoder.jp': 'atcoder',
      'topcoder.com': 'topcoder',
      'interviewing.io': 'interviewing_io',
      'pramp.com': 'pramp'
    };

    for (const [domain, platform] of Object.entries(platforms)) {
      if (url.includes(domain)) {
        this.currentPlatform = platform;
        this.adaptToPlatform(platform);
        return platform;
      }
    }
    
    // Check if it looks like a coding problem based on content
    if (this.looksLikeCodingProblem(title)) {
      this.currentPlatform = 'generic_coding';
      return 'generic_coding';
    }
    
    return null;
  }

  looksLikeCodingProblem(text) {
    const codingIndicators = [
      'algorithm', 'data structure', 'implement', 'function', 'return',
      'array', 'string', 'tree', 'graph', 'linked list', 'hash',
      'time complexity', 'space complexity', 'optimize', 'efficient',
      'leetcode', 'coding challenge', 'programming problem'
    ];
    
    const lowerText = text.toLowerCase();
    return codingIndicators.some(indicator => lowerText.includes(indicator));
  }

  // Platform-specific adaptations
  adaptToPlatform(platform) {
    const platformSettings = {
      leetcode: {
        timeWarning: 'LeetCode problems: Start with brute force, then optimize',
        commonPatterns: ['two_pointers', 'sliding_window', 'dynamic_programming', 'tree_traversal'],
        tipFrequency: 600000 // 10 minutes
      },
      hackerrank: {
        timeWarning: 'HackerRank: Read input/output format carefully',
        commonPatterns: ['implementation', 'mathematics', 'greedy', 'graph'],
        tipFrequency: 900000 // 15 minutes
      },
      codility: {
        timeWarning: 'Codility: Focus on correctness and performance',
        commonPatterns: ['arrays', 'sorting', 'counting', 'prefix_sums'],
        tipFrequency: 1200000 // 20 minutes
      },
      generic_coding: {
        timeWarning: 'Coding assessment detected: Break down the problem step by step',
        commonPatterns: ['basic_algorithms', 'data_structures'],
        tipFrequency: 480000 // 8 minutes
      }
    };

    const settings = platformSettings[platform] || platformSettings.generic_coding;
    
    this.mainWindow?.webContents.send('coding-platform-detected', {
      platform,
      message: settings.timeWarning,
      patterns: settings.commonPatterns
    });
  }

  // Algorithm pattern recognition
  initializePatterns() {
    return {
      // Array patterns
      two_pointers: {
        keywords: ['two pointers', 'left right', 'palindrome', 'sorted array', 'pair sum'],
        hint: 'Try two pointers: one from start, one from end, move based on comparison',
        template: 'left = 0, right = len(arr) - 1\nwhile left < right:\n    # compare arr[left] and arr[right]'
      },
      sliding_window: {
        keywords: ['subarray', 'substring', 'window', 'consecutive', 'maximum sum'],
        hint: 'Sliding window: maintain a window and slide it across the array',
        template: 'left = 0\nfor right in range(len(arr)):\n    # expand window\n    while window_condition_violated:\n        # shrink window from left'
      },
      
      // Tree patterns
      tree_traversal: {
        keywords: ['binary tree', 'traverse', 'inorder', 'preorder', 'postorder', 'dfs', 'bfs'],
        hint: 'Tree problems: Consider DFS (recursive) or BFS (queue) traversal',
        template: 'def dfs(node):\n    if not node:\n        return\n    # process node\n    dfs(node.left)\n    dfs(node.right)'
      },
      
      // Dynamic Programming
      dynamic_programming: {
        keywords: ['optimal', 'maximum', 'minimum', 'count ways', 'fibonacci', 'climb stairs'],
        hint: 'DP pattern: Break into subproblems, find recurrence relation, memoize',
        template: 'dp = [0] * (n + 1)\ndp[0] = base_case\nfor i in range(1, n + 1):\n    dp[i] = recurrence_relation'
      },
      
      // Graph patterns
      graph_bfs: {
        keywords: ['graph', 'shortest path', 'level order', 'minimum steps'],
        hint: 'Graph BFS: Use queue, track visited nodes, good for shortest paths',
        template: 'from collections import deque\nqueue = deque([start])\nvisited = set([start])\nwhile queue:\n    node = queue.popleft()'
      },
      
      // String patterns
      string_matching: {
        keywords: ['string', 'pattern', 'subsequence', 'anagram', 'palindrome'],
        hint: 'String problems: Consider hash maps for counting, or two pointers',
        template: 'char_count = {}\nfor char in string:\n    char_count[char] = char_count.get(char, 0) + 1'
      },
      
      // Mathematical patterns
      mathematics: {
        keywords: ['prime', 'factorial', 'gcd', 'modulo', 'mathematical'],
        hint: 'Math problems: Look for patterns, edge cases, and mathematical properties',
        template: 'result = 1\nfor i in range(1, n + 1):\n    result = (result * i) % MOD'
      }
    };
  }

  // Analyze problem and suggest approach
  analyzeProblem(problemText) {
    const analysis = {
      detectedPatterns: [],
      difficulty: this.estimateDifficulty(problemText),
      suggestions: [],
      timeEstimate: '',
      breakdown: []
    };

    // Pattern detection
    const lowerText = problemText.toLowerCase();
    for (const [pattern, data] of Object.entries(this.algorithmPatterns)) {
      if (data.keywords.some(keyword => lowerText.includes(keyword))) {
        analysis.detectedPatterns.push({
          pattern,
          hint: data.hint,
          template: data.template
        });
      }
    }

    // Generate suggestions based on detected patterns
    analysis.suggestions = this.generateSuggestions(analysis.detectedPatterns, problemText);
    analysis.timeEstimate = this.estimateTime(analysis.difficulty, analysis.detectedPatterns.length);
    analysis.breakdown = this.breakDownProblem(problemText, analysis.detectedPatterns);

    return analysis;
  }

  generateSuggestions(patterns, problemText) {
    const suggestions = [
      "Start by restating the problem in your own words",
      "Identify the input and output format clearly",
      "Think about edge cases: empty input, single element, duplicates"
    ];

    if (patterns.length === 0) {
      suggestions.push("No clear pattern detected - try a brute force approach first");
      suggestions.push("Draw out a small example to understand the problem");
    } else if (patterns.length === 1) {
      suggestions.push(`This looks like a ${patterns[0].pattern.replace('_', ' ')} problem`);
      suggestions.push(patterns[0].hint);
    } else {
      suggestions.push(`Multiple patterns detected: consider ${patterns.map(p => p.pattern.replace('_', ' ')).join(', ')}`);
      suggestions.push("Start with the most familiar pattern to you");
    }

    // Neurodivergent-specific suggestions
    suggestions.push("Take your time - understanding is more important than speed");
    suggestions.push("Write out your approach in comments before coding");

    return suggestions;
  }

  estimateDifficulty(text) {
    const difficultyIndicators = {
      easy: ['simple', 'basic', 'easy', 'straightforward'],
      medium: ['optimize', 'efficient', 'medium', 'better'],
      hard: ['hard', 'complex', 'advanced', 'minimum', 'maximum', 'optimal']
    };

    const lowerText = text.toLowerCase();
    
    for (const [level, indicators] of Object.entries(difficultyIndicators)) {
      if (indicators.some(indicator => lowerText.includes(indicator))) {
        return level;
      }
    }

    // Estimate based on length and complexity
    if (text.length > 1000) return 'hard';
    if (text.length > 500) return 'medium';
    return 'easy';
  }

  estimateTime(difficulty, patternCount) {
    const baseTime = {
      easy: 15,
      medium: 30,
      hard: 45
    };

    const time = baseTime[difficulty] + (patternCount * 5);
    return `${time}-${time + 15} minutes`;
  }

  breakDownProblem(problemText, patterns) {
    const steps = [
      "1. Read and understand the problem statement",
      "2. Identify input/output format and constraints",
      "3. Work through a small example manually",
      "4. Choose your approach based on patterns identified"
    ];

    if (patterns.length > 0) {
      const mainPattern = patterns[0];
      steps.push(`5. Implement ${mainPattern.pattern.replace('_', ' ')} solution`);
      steps.push("6. Test with the example cases");
      steps.push("7. Consider edge cases and optimize if needed");
    } else {
      steps.push("5. Start with brute force approach");
      steps.push("6. Test and verify correctness");
      steps.push("7. Look for optimization opportunities");
    }

    return steps;
  }

  // Provide contextual help during coding
  provideHint(currentCode, stuckTime) {
    const hints = [];

    if (stuckTime > 300000) { // 5 minutes
      hints.push("🤔 Stuck for 5+ minutes? Try explaining the problem out loud");
      hints.push("💡 Consider starting with a simpler version of the problem");
    }

    if (stuckTime > 600000) { // 10 minutes
      hints.push("🧠 ADHD tip: Take a 2-minute break to reset your focus");
      hints.push("📝 Write down what you know and what you need to figure out");
    }

    if (stuckTime > 1200000) { // 20 minutes
      hints.push("🔍 Look up the algorithm pattern - learning is more important than solving alone");
      hints.push("🗣️ If this is an interview, it's okay to ask for a hint");
    }

    return hints;
  }

  // Time management for assessments
  startAssessment(timeLimit = null) {
    this.problemContext.startTime = Date.now();
    this.problemContext.timeLimit = timeLimit;

    if (timeLimit) {
      // Set up time warnings
      const quarterTime = timeLimit * 0.25;
      const halfTime = timeLimit * 0.5;
      const threeQuarterTime = timeLimit * 0.75;

      setTimeout(() => {
        this.sendTimeWarning("25% time elapsed - make sure you have a working solution");
      }, quarterTime);

      setTimeout(() => {
        this.sendTimeWarning("50% time elapsed - focus on getting something working");
      }, halfTime);

      setTimeout(() => {
        this.sendTimeWarning("75% time elapsed - finalize your solution and test");
      }, threeQuarterTime);
    }
  }

  sendTimeWarning(message) {
    this.mainWindow?.webContents.send('time-warning', {
      message,
      timestamp: Date.now()
    });
  }

  // Generate problem-solving template
  generateTemplate(patterns) {
    if (patterns.length === 0) {
      return `# Problem-solving template
# 1. Understand the problem
# 2. Plan your approach
# 3. Implement step by step
# 4. Test with examples

def solve(input_params):
    # Your solution here
    pass

# Test cases
# solve(example1) == expected1
# solve(example2) == expected2`;
    }

    const mainPattern = patterns[0];
    return `# ${mainPattern.pattern.replace('_', ' ').toUpperCase()} Pattern Detected
# Hint: ${mainPattern.hint}

${mainPattern.template}

def solve(input_params):
    # Implement your solution here
    pass`;
  }
}

function registerCodingAssessmentHandlers(mainWindow) {
  const assessmentHelper = new CodingAssessmentHelper(mainWindow);

  ipcMain.handle('coding:detect-platform', (_event, url, title) => {
    return assessmentHelper.detectPlatform(url, title);
  });

  ipcMain.handle('coding:analyze-problem', (_event, problemText) => {
    return assessmentHelper.analyzeProblem(problemText);
  });

  ipcMain.handle('coding:get-hint', (_event, currentCode, stuckTime) => {
    return assessmentHelper.provideHint(currentCode, stuckTime);
  });

  ipcMain.handle('coding:start-assessment', (_event, timeLimit) => {
    assessmentHelper.startAssessment(timeLimit);
    return true;
  });

  ipcMain.handle('coding:generate-template', (_event, patterns) => {
    return assessmentHelper.generateTemplate(patterns);
  });

  return assessmentHelper;
}

module.exports = {
  CodingAssessmentHelper,
  registerCodingAssessmentHandlers
};