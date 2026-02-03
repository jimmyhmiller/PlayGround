# 🐦 Flappy Bird - Premium Edition

A modern, full-stack implementation of Flappy Bird with real-time statistics, leaderboards, and a native macOS app.

## Features

### Frontend
- Beautiful, responsive game interface
- Real-time score tracking
- Smooth animations and graphics
- Keyboard and mouse controls
- Live connection status

### Backend
- RESTful API with Express.js
- SQLite database for persistent storage
- WebSocket support for real-time updates
- Leaderboard system
- Game session tracking
- Global statistics
- Dynamic game parameter management

### Native App
- Native macOS application using Swift/SwiftUI
- Embedded WebView for seamless gameplay
- Built-in server management
- One-click launch and play

## Architecture

```
flappy/
├── frontend/          # Game frontend (HTML/JS)
│   ├── index.html    # Main game interface
│   └── game.js       # Game logic and API integration
├── backend/          # Node.js backend server
│   └── server.js     # Express API + WebSocket server
├── app/              # macOS native app
│   └── Sources/
│       └── FlappyBird/
│           └── FlappyBirdApp.swift
└── flappy.db         # SQLite database (created on first run)
```

## Installation

### Prerequisites
- Node.js 16+ and npm
- Swift 6.1+ (for native app)

### Setup

1. Install dependencies:
```bash
npm install
```

2. That's it! The database will be created automatically on first run.

## Running the Game

### Option 1: Native macOS App (Recommended)
```bash
npm run app
```

This will:
- Launch the native macOS application
- Automatically start the backend server
- Open the game in an embedded WebView
- Handle server lifecycle management

### Option 2: Web Browser
```bash
npm start
```

Then open your browser to `http://localhost:3000`

### Option 3: Development Mode
```bash
npm run dev
```

## How to Play

1. Click anywhere or press **Space** to start
2. Click or press **Space** to make the bird jump
3. Avoid the pipes!
4. Try to get the highest score
5. Submit your score to the leaderboard

## API Endpoints

### Leaderboard
- `GET /api/leaderboard?limit=10` - Get top scores
- `POST /api/leaderboard` - Submit a new score

### Game Sessions
- `POST /api/sessions` - Create a new game session
- `GET /api/sessions/:sessionId` - Get session details
- `PUT /api/sessions/:sessionId` - Update session stats

### Statistics
- `GET /api/stats` - Get global game statistics

### Game Parameters
- `GET /api/params` - Get current game parameters
- `PUT /api/params` - Update game parameters

## WebSocket Events

The game uses WebSocket for real-time updates:

- `game.started` - Game session started
- `bird.jump` - Bird jumped
- `score.update` - Score increased
- `game.over` - Game ended
- `game.update` - Periodic game state update
- `params-updated` - Game parameters changed

## Database Schema

### Tables
- `leaderboard` - High scores
- `game_sessions` - Individual game sessions
- `game_events` - Real-time event log
- `game_params` - Game configuration

## Customization

You can modify game parameters via the API:

```bash
curl -X PUT http://localhost:3000/api/params \
  -H "Content-Type: application/json" \
  -d '{
    "gravity": 0.8,
    "jumpStrength": -12,
    "pipeSpeed": 4,
    "pipeGap": 140
  }'
```

Changes will be broadcast to all connected clients in real-time.

## Tech Stack

- **Frontend**: Vanilla JavaScript, HTML5 Canvas
- **Backend**: Node.js, Express.js, SQLite, WebSocket (ws)
- **Native App**: Swift, SwiftUI, WebKit
- **Database**: SQLite3

## License

ISC

## Credits

Created as a modern take on the classic Flappy Bird game.
