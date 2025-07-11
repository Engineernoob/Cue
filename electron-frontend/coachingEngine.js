// coachingEngine.js
const coachingPrompts = {
    interview: [
      "Try the STAR method: Situation, Task, Action, Result.",
      "Breathe — clarify the question if needed.",
      "Highlight your impact, not just your task.",
    ],
    default: [
      "Take a moment to break down the problem.",
      "Focus on one step at a time.",
    ],
  };
  
  function getPrompt(type = "default") {
    const prompts = coachingPrompts[type] || coachingPrompts.default;
    return prompts[Math.floor(Math.random() * prompts.length)];
  }
  
  module.exports = { getPrompt };