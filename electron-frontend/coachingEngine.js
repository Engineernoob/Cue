const { ipcMain } = require("electron");
const { getConfig } = require("./config");

const coachingPrompts = {
  interview: [
    "Try the STAR method: Situation, Task, Action, Result.",
    "Breathe — clarify the question if needed.",
    "Highlight your impact, not just your task.",
    "It's okay to think out loud - walk through your process.",
    "Ask for clarification if the question seems ambiguous."
  ],
  
  coding_interview: [
    "Start by understanding the problem - restate it in your own words.",
    "Think about edge cases: empty input, single element, duplicates.",
    "Consider time/space complexity - can you optimize?",
    "Trace through your solution with a simple example.",
    "Don't forget to test your solution step by step.",
    "It's fine to start with a brute force approach, then optimize."
  ],
  
  debugging: [
    "Check your variable names and types first.",
    "Add console.log or print statements to track values.",
    "Look for off-by-one errors in loops and arrays.",
    "Verify your conditional logic - are you checking the right conditions?",
    "Step through the code line by line in your mind."
  ],
  
  algorithm: [
    "What's the pattern here? Sorting, searching, graph traversal?",
    "Can you solve a simpler version of this problem first?",
    "Think about data structures: arrays, hash maps, trees, graphs.",
    "Consider if this is a dynamic programming problem.",
    "Sometimes the naive solution reveals the optimized approach."
  ],
  
  // Neurodivergent-specific prompts
  adhd_focus: [
    "Take a 30-second break to reset your focus.",
    "Write down the key points so far to clear your working memory.",
    "Break this problem into smaller, concrete steps.",
    "You've been coding for a while - time for a movement break?"
  ],
  
  autism_structure: [
    "Let's organize this problem into clear, sequential steps.",
    "Here's the structure: Input → Process → Output.",
    "Use the same problem-solving pattern you know works.",
    "Take time to process - there's no rush to answer immediately."
  ],
  
  anxiety_support: [
    "You know more than you think you do.",
    "It's normal to feel uncertain - that means you're learning.",
    "Take three deep breaths. You've got this.",
    "Focus on what you do know, not what you're unsure about.",
    "Progress over perfection - any step forward counts."
  ],
  
  default: [
    "Take a moment to break down the problem.",
    "Focus on one step at a time.",
    "You're doing great - trust your instincts."
  ]
};

// Context-aware prompt selection
function getPrompt(type = "default", context = {}) {
  const config = getConfig('neurodivergent');
  let selectedType = type;
  
  // Detect coding context
  if (context.text && isCodeRelated(context.text)) {
    if (context.text.includes('debug') || context.text.includes('error')) {
      selectedType = 'debugging';
    } else if (context.text.includes('algorithm') || context.text.includes('optimize')) {
      selectedType = 'algorithm';
    } else if (type === 'interview') {
      selectedType = 'coding_interview';
    }
  }
  
  // Apply neurodivergent-specific support
  if (context.stressLevel === 'high' && config.anxiety.confidenceBoosts) {
    selectedType = 'anxiety_support';
  }
  
  const prompts = coachingPrompts[selectedType] || coachingPrompts.default;
  return prompts[Math.floor(Math.random() * prompts.length)];
}

// Helper function to detect code-related content
function isCodeRelated(text) {
  const codeKeywords = [
    'function', 'class', 'variable', 'array', 'object', 'loop', 'condition',
    'algorithm', 'data structure', 'complexity', 'debug', 'error', 'syntax',
    'compile', 'runtime', 'javascript', 'python', 'java', 'cpp', 'coding',
    'programming', 'leetcode', 'hackerrank', 'interview question'
  ];
  
  return codeKeywords.some(keyword => 
    text.toLowerCase().includes(keyword)
  );
}

function registerCoachingHandlers() {
  if (!ipcMain.eventNames().includes("coaching:get-prompt")) {
    ipcMain.handle("coaching:get-prompt", (_event, type) => {
      return getPrompt(type);
    });
  }
}

module.exports = { getPrompt, registerCoachingHandlers };