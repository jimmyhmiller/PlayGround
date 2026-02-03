const express = require('express');
const cors = require('cors');
const path = require('path');
const sqlite3 = require('sqlite3').verbose();
const WebSocket = require('ws');
const http = require('http');

const app = express();
const PORT = process.env.PORT || 3000;

// Create HTTP server
const server = http.createServer(app);

// Create WebSocket server
const wss = new WebSocket.Server({ server });

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../frontend')));

// Initialize SQLite database
const db = new sqlite3.Database('./flappy.db', (err) => {
    if (err) {
        console.error('Error opening database:', err);
    } else {
        console.log('Connected to SQLite database');
        initDatabase();
    }
});

// Initialize database tables
function initDatabase() {
    db.run(`
        CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            score INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    `);

    db.run(`
        CREATE TABLE IF NOT EXISTS game_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            player_name TEXT,
            start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            end_time DATETIME,
            final_score INTEGER,
            total_jumps INTEGER DEFAULT 0,
            max_velocity REAL DEFAULT 0,
            status TEXT DEFAULT 'active'
        )
    `);

    db.run(`
        CREATE TABLE IF NOT EXISTS game_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_data TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    `);

    db.run(`
        CREATE TABLE IF NOT EXISTS game_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gravity REAL DEFAULT 0.6,
            jump_strength REAL DEFAULT -10,
            pipe_speed REAL DEFAULT 3,
            pipe_gap INTEGER DEFAULT 150,
            pipe_interval INTEGER DEFAULT 90,
            bird_size INTEGER DEFAULT 20,
            bird_x INTEGER DEFAULT 100,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    `);

    // Insert default params if none exist
    db.get('SELECT COUNT(*) as count FROM game_params', (err, row) => {
        if (!err && row.count === 0) {
            db.run(`
                INSERT INTO game_params (gravity, jump_strength, pipe_speed, pipe_gap, pipe_interval, bird_size, bird_x)
                VALUES (0.6, -10, 3, 150, 90, 20, 100)
            `);
        }
    });
}

// WebSocket connections
const clients = new Set();

wss.on('connection', (ws) => {
    console.log('New WebSocket client connected');
    clients.add(ws);

    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            handleWebSocketMessage(ws, data);
        } catch (err) {
            console.error('Error parsing WebSocket message:', err);
        }
    });

    ws.on('close', () => {
        console.log('WebSocket client disconnected');
        clients.delete(ws);
    });

    // Send initial connection success
    ws.send(JSON.stringify({ type: 'connected', timestamp: Date.now() }));
});

function handleWebSocketMessage(ws, data) {
    switch (data.type) {
        case 'game-event':
            // Record game event in database
            if (data.sessionId) {
                db.run(
                    'INSERT INTO game_events (session_id, event_type, event_data) VALUES (?, ?, ?)',
                    [data.sessionId, data.eventType, JSON.stringify(data.data)]
                );
            }
            // Broadcast to all connected clients
            broadcastToClients({ type: 'game-update', data });
            break;
    }
}

function broadcastToClients(message) {
    const messageStr = JSON.stringify(message);
    clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(messageStr);
        }
    });
}

// API Routes

// Get leaderboard
app.get('/api/leaderboard', (req, res) => {
    const limit = parseInt(req.query.limit) || 10;
    db.all(
        'SELECT player_name, score, created_at FROM leaderboard ORDER BY score DESC LIMIT ?',
        [limit],
        (err, rows) => {
            if (err) {
                res.status(500).json({ error: err.message });
            } else {
                res.json({ leaderboard: rows });
            }
        }
    );
});

// Add score to leaderboard
app.post('/api/leaderboard', (req, res) => {
    const { player_name, score } = req.body;
    
    if (!player_name || score === undefined) {
        return res.status(400).json({ error: 'Player name and score are required' });
    }

    db.run(
        'INSERT INTO leaderboard (player_name, score) VALUES (?, ?)',
        [player_name, score],
        function(err) {
            if (err) {
                res.status(500).json({ error: err.message });
            } else {
                res.json({ 
                    success: true, 
                    id: this.lastID,
                    message: 'Score added to leaderboard' 
                });
            }
        }
    );
});

// Create new game session
app.post('/api/sessions', (req, res) => {
    const { player_name } = req.body;
    const session_id = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    db.run(
        'INSERT INTO game_sessions (session_id, player_name) VALUES (?, ?)',
        [session_id, player_name || 'Anonymous'],
        function(err) {
            if (err) {
                res.status(500).json({ error: err.message });
            } else {
                res.json({ 
                    success: true, 
                    session_id,
                    id: this.lastID
                });
            }
        }
    );
});

// Update game session
app.put('/api/sessions/:sessionId', (req, res) => {
    const { sessionId } = req.params;
    const { final_score, total_jumps, max_velocity, status } = req.body;

    const updates = [];
    const params = [];

    if (final_score !== undefined) {
        updates.push('final_score = ?');
        params.push(final_score);
    }
    if (total_jumps !== undefined) {
        updates.push('total_jumps = ?');
        params.push(total_jumps);
    }
    if (max_velocity !== undefined) {
        updates.push('max_velocity = ?');
        params.push(max_velocity);
    }
    if (status !== undefined) {
        updates.push('status = ?');
        params.push(status);
    }
    if (status === 'completed') {
        updates.push('end_time = CURRENT_TIMESTAMP');
    }

    params.push(sessionId);

    if (updates.length === 0) {
        return res.status(400).json({ error: 'No valid fields to update' });
    }

    db.run(
        `UPDATE game_sessions SET ${updates.join(', ')} WHERE session_id = ?`,
        params,
        function(err) {
            if (err) {
                res.status(500).json({ error: err.message });
            } else {
                res.json({ success: true, changes: this.changes });
            }
        }
    );
});

// Get session statistics
app.get('/api/sessions/:sessionId', (req, res) => {
    const { sessionId } = req.params;

    db.get(
        'SELECT * FROM game_sessions WHERE session_id = ?',
        [sessionId],
        (err, row) => {
            if (err) {
                res.status(500).json({ error: err.message });
            } else if (!row) {
                res.status(404).json({ error: 'Session not found' });
            } else {
                res.json({ session: row });
            }
        }
    );
});

// Get game statistics
app.get('/api/stats', (req, res) => {
    db.get(
        `SELECT 
            COUNT(*) as total_games,
            AVG(final_score) as avg_score,
            MAX(final_score) as max_score,
            AVG(total_jumps) as avg_jumps
        FROM game_sessions 
        WHERE status = 'completed'`,
        (err, row) => {
            if (err) {
                res.status(500).json({ error: err.message });
            } else {
                res.json({ stats: row });
            }
        }
    );
});

// Get current game parameters
app.get('/api/params', (req, res) => {
    db.get('SELECT * FROM game_params ORDER BY id DESC LIMIT 1', (err, row) => {
        if (err) {
            res.status(500).json({ error: err.message });
        } else {
            const params = {
                gravity: row.gravity,
                jumpStrength: row.jump_strength,
                pipeSpeed: row.pipe_speed,
                pipeGap: row.pipe_gap,
                pipeInterval: row.pipe_interval,
                birdSize: row.bird_size,
                birdX: row.bird_x
            };
            res.json({ params });
        }
    });
});

// Update game parameters
app.put('/api/params', (req, res) => {
    const { gravity, jumpStrength, pipeSpeed, pipeGap, pipeInterval, birdSize, birdX } = req.body;

    const updates = [];
    const params = [];

    if (gravity !== undefined) {
        updates.push('gravity = ?');
        params.push(gravity);
    }
    if (jumpStrength !== undefined) {
        updates.push('jump_strength = ?');
        params.push(jumpStrength);
    }
    if (pipeSpeed !== undefined) {
        updates.push('pipe_speed = ?');
        params.push(pipeSpeed);
    }
    if (pipeGap !== undefined) {
        updates.push('pipe_gap = ?');
        params.push(pipeGap);
    }
    if (pipeInterval !== undefined) {
        updates.push('pipe_interval = ?');
        params.push(pipeInterval);
    }
    if (birdSize !== undefined) {
        updates.push('bird_size = ?');
        params.push(birdSize);
    }
    if (birdX !== undefined) {
        updates.push('bird_x = ?');
        params.push(birdX);
    }

    updates.push('updated_at = CURRENT_TIMESTAMP');

    if (updates.length === 1) {
        return res.status(400).json({ error: 'No valid fields to update' });
    }

    db.run(
        `UPDATE game_params SET ${updates.join(', ')} WHERE id = (SELECT MAX(id) FROM game_params)`,
        params,
        function(err) {
            if (err) {
                res.status(500).json({ error: err.message });
            } else {
                // Fetch the complete updated params and broadcast
                db.get('SELECT * FROM game_params ORDER BY id DESC LIMIT 1', (err, row) => {
                    if (!err && row) {
                        const fullParams = {
                            gravity: row.gravity,
                            jumpStrength: row.jump_strength,
                            pipeSpeed: row.pipe_speed,
                            pipeGap: row.pipe_gap,
                            pipeInterval: row.pipe_interval,
                            birdSize: row.bird_size,
                            birdX: row.bird_x
                        };
                        console.log('Broadcasting params update:', fullParams);
                        broadcastToClients({ 
                            type: 'params-updated', 
                            params: fullParams 
                        });
                    }
                });
                res.json({ success: true, changes: this.changes });
            }
        }
    );
});

// Serve the game
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

// Start server
server.listen(PORT, () => {
    console.log(`Flappy Bird server running on http://localhost:${PORT}`);
    console.log(`WebSocket server ready`);
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\nShutting down gracefully...');
    db.close((err) => {
        if (err) {
            console.error('Error closing database:', err);
        } else {
            console.log('Database closed');
        }
        process.exit(0);
    });
});
