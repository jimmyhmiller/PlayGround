# Multiple Choice Question Feature - Testing Guide

## Overview
We've implemented a multiple choice question feature that allows Claude to ask users questions and wait for their answers, similar to Claude Code's AskQuestion tool.

## Implementation Summary

### 1. **Frontend (App.jsx)**
- **ChatMessage Component**: Now renders multiple choice questions with clickable option buttons
- **handleAnswerQuestion**: Sends the user's answer to backend and marks question as answered
- **Question Event Listener**: Listens for `claude-question` events and displays questions in chat

### 2. **Backend (main.js)**
- **AskQuestion Tool**: MCP tool that Claude can invoke to ask questions
- **IPC Handler**: `claude-answer-question` receives user answers and resolves pending promises
- **Question Callback**: Sends questions to UI via `claude-question` event
- **Pending Questions Map**: Stores promises that resolve when user answers

### 3. **Dashboard Tools (dashboard-tools.js)**
- **createGeneralTools**: Creates MCP server with AskQuestion tool
- **Tool Logic**: Creates promise, sends to UI, waits for answer, returns result to Claude

### 4. **Preload (preload.js)**
- **answerQuestion**: IPC method to submit answers
- **onQuestion/offQuestion**: Event listeners for receiving questions

## How It Works

```
1. Claude calls AskQuestion tool with question and options
   ↓
2. Tool creates a pending promise and questionId
   ↓
3. Backend sends 'claude-question' event to UI
   ↓
4. UI displays question as message with option buttons
   ↓
5. User clicks an option
   ↓
6. Frontend calls answerQuestion IPC
   ↓
7. Backend resolves the pending promise
   ↓
8. Tool returns user's answer to Claude
   ↓
9. Claude continues with the answer
```

## Testing Instructions

### Test 1: Basic Multiple Choice
Ask Claude in the chat widget:
```
Can you ask me a multiple choice question? Like what's my favorite color: red, blue, or green?
```

Expected behavior:
- Claude should use the AskQuestion tool
- A message appears with the question text
- Three buttons appear below: "red", "blue", "green"
- When you click one, it becomes disabled
- Your answer appears as a user message
- Claude responds based on your answer

### Test 2: Yes/No Question
Ask Claude:
```
Ask me if I want to continue with yes or no options
```

Expected behavior:
- Question appears with two buttons: "Yes" and "No"
- Clicking either sends the answer
- Claude acknowledges your choice

### Test 3: Complex Options
Ask Claude:
```
Give me a choice between implementing feature A (faster), feature B (more features), or feature C (better UX)
```

Expected behavior:
- Question with three descriptive options
- Each option is clearly labeled
- Claude uses your selection to inform next steps

### Test 4: Multiple Questions
Ask Claude:
```
Ask me two questions: first about my preferred programming language (JavaScript, Python, or Rust), then about my experience level (beginner, intermediate, or advanced)
```

Expected behavior:
- First question appears with three options
- After answering, second question appears
- Claude uses both answers in its response

## Debugging

If issues occur, check these logs:

**Browser Console:**
- `[Chat UI] Received question from Claude:` - Question received
- `[Chat UI] Question answered:` - Answer submitted

**Electron Main Process:**
- `[Claude] Sending question to UI:` - Question sent to frontend
- `[Claude] Received answer for question:` - Answer received
- `[Claude] Question answered successfully` - Promise resolved

**Common Issues:**
1. **Buttons don't appear**: Check if `msg.multipleChoice` is set in message
2. **Answer doesn't submit**: Check `answerQuestion` IPC handler
3. **Claude doesn't receive answer**: Check if promise in `pendingQuestions` Map is being resolved
4. **Questions disappear on refresh**: They're saved to chat history as messages with `multipleChoice` field

## Code Locations

- **ChatMessage rendering**: `src/App.jsx` lines 1776-1881
- **Answer handler**: `src/App.jsx` lines 2217-2244
- **Question listener**: `src/App.jsx` lines 2199-2228
- **AskQuestion tool**: `dashboard-tools.js` lines 543-587
- **IPC handler**: `main.js` lines 752-772
- **IPC API**: `preload.js` lines 196-210

## Future Enhancements

1. **Show selected answer**: Highlight which option was selected after answering
2. **Validation**: Allow specifying correct/incorrect answers
3. **Custom styling**: Per-option colors or icons
4. **Keyboard shortcuts**: Number keys to select options
5. **Timeout**: Auto-answer after certain duration
6. **History**: Show previous answers when viewing old conversations
