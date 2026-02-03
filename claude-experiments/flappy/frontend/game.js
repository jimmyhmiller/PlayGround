// API and WebSocket configuration
const API_URL = window.location.origin;
const WS_URL = `ws://${window.location.hostname}:${window.location.port || 3000}`;

let ws = null;
let sessionId = null;
let reconnectInterval = null;

// Canvas and game elements
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const scoreElement = document.getElementById('score');
const gameOverElement = document.getElementById('gameOver');
const finalScoreElement = document.getElementById('finalScore');
const instructionsElement = document.getElementById('instructions');
const highScoreMsg = document.getElementById('highScoreMsg');

// Game parameters - MUCH BETTER TUNING
let gameParams = {
    gravity: 0.45,
    jumpStrength: -9,
    pipeSpeed: 2.5,
    pipeGap: 180,
    pipeInterval: 100,
    birdSize: 24,
    birdX: 100
};

// Particle system
class Particle {
    constructor(x, y, color, vx, vy) {
        this.x = x;
        this.y = y;
        this.color = color;
        this.vx = vx;
        this.vy = vy;
        this.life = 1.0;
        this.size = Math.random() * 4 + 2;
    }
    
    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.vy += 0.2;
        this.life -= 0.02;
    }
    
    draw(ctx) {
        ctx.globalAlpha = this.life;
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
    }
    
    isDead() {
        return this.life <= 0;
    }
}

// Trail system for bird
class TrailParticle {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.life = 1.0;
        this.size = 12;
    }
    
    update() {
        this.life -= 0.05;
        this.size *= 0.95;
    }
    
    draw(ctx) {
        ctx.globalAlpha = this.life * 0.3;
        ctx.fillStyle = '#FFD700';
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
    }
    
    isDead() {
        return this.life <= 0;
    }
}

// Cloud decoration
class Cloud {
    constructor() {
        this.x = canvas.width + Math.random() * 200;
        this.y = Math.random() * canvas.height * 0.6;
        this.speed = 0.2 + Math.random() * 0.3;
        this.scale = 0.5 + Math.random() * 0.8;
    }
    
    update() {
        this.x -= this.speed;
        if (this.x < -100) {
            this.x = canvas.width + Math.random() * 200;
            this.y = Math.random() * canvas.height * 0.6;
        }
    }
    
    draw(ctx) {
        ctx.globalAlpha = 0.6;
        ctx.fillStyle = '#ffffff';
        
        const size = 30 * this.scale;
        ctx.beginPath();
        ctx.arc(this.x, this.y, size, 0, Math.PI * 2);
        ctx.arc(this.x + size * 0.8, this.y, size * 0.8, 0, Math.PI * 2);
        ctx.arc(this.x + size * 1.5, this.y, size * 0.9, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.globalAlpha = 1;
    }
}

// Game state
let gameState = {
    bird: { y: 300, velocity: 0, rotation: 0 },
    pipes: [],
    particles: [],
    trailParticles: [],
    clouds: [],
    score: 0,
    gameStarted: false,
    gameOver: false,
    frameCount: 0,
    totalJumps: 0,
    maxVelocity: 0,
    cameraShake: 0,
    flashAlpha: 0
};

// Initialize clouds
for (let i = 0; i < 5; i++) {
    const cloud = new Cloud();
    cloud.x = Math.random() * canvas.width;
    gameState.clouds.push(cloud);
}

// Bird object
function createBird() {
    return {
        x: gameParams.birdX,
        y: canvas.height / 2,
        velocity: 0,
        size: gameParams.birdSize,
        rotation: 0
    };
}

// Pipe object
function createPipe() {
    const minHeight = 80;
    const maxHeight = canvas.height - gameParams.pipeGap - 180;
    const height = Math.random() * (maxHeight - minHeight) + minHeight;
    
    return {
        x: canvas.width,
        topHeight: height,
        bottomY: height + gameParams.pipeGap,
        width: 65,
        passed: false
    };
}

// Create particles
function createParticles(x, y, color, count = 20) {
    for (let i = 0; i < count; i++) {
        const angle = Math.random() * Math.PI * 2;
        const speed = Math.random() * 4 + 2;
        const vx = Math.cos(angle) * speed;
        const vy = Math.sin(angle) * speed;
        gameState.particles.push(new Particle(x, y, color, vx, vy));
    }
}

// Initialize game
async function init() {
    gameState.bird = createBird();
    gameState.pipes = [];
    gameState.particles = [];
    gameState.trailParticles = [];
    gameState.score = 0;
    gameState.gameStarted = false;
    gameState.gameOver = false;
    gameState.frameCount = 0;
    gameState.totalJumps = 0;
    gameState.maxVelocity = 0;
    gameState.cameraShake = 0;
    gameState.flashAlpha = 0;
    
    scoreElement.textContent = '0';
    gameOverElement.style.display = 'none';
    instructionsElement.style.display = 'block';
    highScoreMsg.style.display = 'none';
    
    updateSessionStats();
    
    // Create new session
    await createSession();
}

// Jump function with juice
function jump() {
    if (!gameState.gameStarted) {
        gameState.gameStarted = true;
        instructionsElement.style.display = 'none';
        sendGameEvent('game.started');
    }
    
    if (!gameState.gameOver) {
        gameState.bird.velocity = gameParams.jumpStrength;
        gameState.totalJumps++;
        
        // Play jump sound
        if (window.audioSystem) {
            window.audioSystem.playJump();
        }
        
        // Create upward particles on jump
        createParticles(
            gameState.bird.x + gameState.bird.size / 2,
            gameState.bird.y + gameState.bird.size,
            '#FFD700',
            8
        );
        
        // Small screen shake
        gameState.cameraShake = 2;
        
        updateSessionStats();
        sendGameEvent('bird.jump', { 
            y: gameState.bird.y, 
            velocity: gameState.bird.velocity,
            totalJumps: gameState.totalJumps
        });
    }
}

// Update game state
function update() {
    if (!gameState.gameStarted || gameState.gameOver) return;

    // Update bird physics
    gameState.bird.velocity += gameParams.gravity;
    gameState.bird.y += gameState.bird.velocity;
    
    // Update bird rotation based on velocity
    gameState.bird.rotation = Math.min(Math.max(gameState.bird.velocity * 3, -30), 90);
    
    // Add trail particles
    if (gameState.frameCount % 3 === 0) {
        gameState.trailParticles.push(
            new TrailParticle(
                gameState.bird.x + gameState.bird.size / 2,
                gameState.bird.y + gameState.bird.size / 2
            )
        );
    }
    
    // Track max velocity
    if (Math.abs(gameState.bird.velocity) > Math.abs(gameState.maxVelocity)) {
        gameState.maxVelocity = gameState.bird.velocity;
        updateSessionStats();
    }

    // Generate pipes
    if (gameState.frameCount % gameParams.pipeInterval === 0) {
        gameState.pipes.push(createPipe());
    }

    // Update pipes
    for (let i = gameState.pipes.length - 1; i >= 0; i--) {
        const pipe = gameState.pipes[i];
        pipe.x -= gameParams.pipeSpeed;

        // Check if pipe passed
        if (!pipe.passed && pipe.x + pipe.width < gameState.bird.x) {
            pipe.passed = true;
            gameState.score++;
            scoreElement.textContent = gameState.score;
            document.getElementById('currentScore').textContent = gameState.score;
            
            // Play score sound
            if (window.audioSystem) {
                window.audioSystem.playScore();
            }
            
            // Score particle effect
            createParticles(pipe.x + pipe.width, canvas.height / 2, '#00ff00', 15);
            gameState.flashAlpha = 0.15;
            
            sendGameEvent('score.update', { score: gameState.score });
        }

        // Remove off-screen pipes
        if (pipe.x + pipe.width < 0) {
            gameState.pipes.splice(i, 1);
        }

        // Check collision with tighter hitbox
        if (checkCollision(pipe)) {
            endGame();
        }
    }

    // Check bounds
    if (gameState.bird.y + gameState.bird.size > canvas.height * 0.8 || gameState.bird.y < 0) {
        endGame();
    }

    // Update particles
    gameState.particles = gameState.particles.filter(p => {
        p.update();
        return !p.isDead();
    });
    
    // Update trail particles
    gameState.trailParticles = gameState.trailParticles.filter(p => {
        p.update();
        return !p.isDead();
    });
    
    // Update clouds
    gameState.clouds.forEach(cloud => cloud.update());
    
    // Decay camera shake
    gameState.cameraShake *= 0.8;
    if (gameState.cameraShake < 0.1) gameState.cameraShake = 0;
    
    // Decay flash
    gameState.flashAlpha *= 0.9;

    gameState.frameCount++;
    
    // Send periodic updates
    if (gameState.frameCount % 30 === 0) {
        sendGameEvent('game.update', {
            birdY: gameState.bird.y,
            velocity: gameState.bird.velocity,
            score: gameState.score,
            pipeCount: gameState.pipes.length
        });
    }
}

// Check collision with better hitbox
function checkCollision(pipe) {
    const bird = gameState.bird;
    
    // Smaller hitbox for more forgiving gameplay
    const hitboxPadding = 3;
    const birdLeft = bird.x + hitboxPadding;
    const birdRight = bird.x + bird.size - hitboxPadding;
    const birdTop = bird.y + hitboxPadding;
    const birdBottom = bird.y + bird.size - hitboxPadding;
    
    if (birdRight > pipe.x && birdLeft < pipe.x + pipe.width) {
        if (birdTop < pipe.topHeight || birdBottom > pipe.bottomY) {
            return true;
        }
    }
    
    return false;
}

// End game with effects
async function endGame() {
    if (gameState.gameOver) return;
    
    gameState.gameOver = true;
    finalScoreElement.textContent = gameState.score;
    
    // Play death sound
    if (window.audioSystem) {
        window.audioSystem.playDeath();
    }
    
    // Big explosion of particles
    createParticles(
        gameState.bird.x + gameState.bird.size / 2,
        gameState.bird.y + gameState.bird.size / 2,
        '#ff0000',
        40
    );
    
    // Screen shake
    gameState.cameraShake = 15;
    gameState.flashAlpha = 0.3;
    
    // Check if it's a new high score
    const leaderboard = await fetchLeaderboard();
    const topScore = leaderboard.length > 0 ? leaderboard[0].score : 0;
    if (gameState.score > topScore) {
        highScoreMsg.style.display = 'block';
    }
    
    setTimeout(() => {
        gameOverElement.style.display = 'block';
    }, 500);
    
    sendGameEvent('game.over', { 
        score: gameState.score,
        totalJumps: gameState.totalJumps,
        maxVelocity: gameState.maxVelocity
    });
    
    // Update session
    await updateSession({
        final_score: gameState.score,
        total_jumps: gameState.totalJumps,
        max_velocity: gameState.maxVelocity,
        status: 'completed'
    });
    
    // Refresh stats
    fetchGlobalStats();
}

// Restart game
function restartGame() {
    init();
}

// Submit score to leaderboard
async function submitScore() {
    const playerName = document.getElementById('playerName').value.trim() || 'Anonymous';
    
    try {
        const response = await fetch(`${API_URL}/api/leaderboard`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                player_name: playerName,
                score: gameState.score
            })
        });
        
        const data = await response.json();
        if (data.success) {
            alert('Score submitted successfully!');
            fetchLeaderboard();
            restartGame();
        }
    } catch (err) {
        console.error('Error submitting score:', err);
        alert('Failed to submit score. Please try again.');
    }
}

// Draw everything with much better graphics
function draw() {
    // Apply camera shake
    ctx.save();
    if (gameState.cameraShake > 0) {
        const shakeX = (Math.random() - 0.5) * gameState.cameraShake;
        const shakeY = (Math.random() - 0.5) * gameState.cameraShake;
        ctx.translate(shakeX, shakeY);
    }
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw clouds
    gameState.clouds.forEach(cloud => cloud.draw(ctx));
    
    // Draw trail particles
    gameState.trailParticles.forEach(particle => particle.draw(ctx));

    // Draw bird with rotation and better graphics
    ctx.save();
    ctx.translate(
        gameState.bird.x + gameState.bird.size / 2,
        gameState.bird.y + gameState.bird.size / 2
    );
    ctx.rotate((gameState.bird.rotation * Math.PI) / 180);
    
    // Bird shadow
    ctx.globalAlpha = 0.3;
    ctx.fillStyle = '#000';
    ctx.beginPath();
    ctx.arc(2, 2, gameState.bird.size / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    
    // Bird body with gradient
    const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, gameState.bird.size / 2);
    gradient.addColorStop(0, '#FFE55C');
    gradient.addColorStop(1, '#FFD700');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(0, 0, gameState.bird.size / 2, 0, Math.PI * 2);
    ctx.fill();
    
    // Bird outline
    ctx.strokeStyle = '#FFA500';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Wing
    ctx.fillStyle = '#FFA500';
    ctx.beginPath();
    ctx.ellipse(-3, 0, 6, 10, Math.sin(gameState.frameCount * 0.2) * 0.3, 0, Math.PI * 2);
    ctx.fill();
    
    // Eye
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(6, -3, 5, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.fillStyle = '#000';
    ctx.beginPath();
    ctx.arc(7, -3, 3, 0, Math.PI * 2);
    ctx.fill();
    
    // Eye shine
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(8, -4, 1.5, 0, Math.PI * 2);
    ctx.fill();

    // Beak
    ctx.fillStyle = '#FF6347';
    ctx.beginPath();
    ctx.moveTo(gameState.bird.size / 2, 0);
    ctx.lineTo(gameState.bird.size / 2 + 10, -2);
    ctx.lineTo(gameState.bird.size / 2 + 10, 2);
    ctx.closePath();
    ctx.fill();
    
    // Beak outline
    ctx.strokeStyle = '#CC4A37';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    ctx.restore();

    // Draw pipes with better graphics
    for (const pipe of gameState.pipes) {
        // Pipe gradient
        const pipeGradient = ctx.createLinearGradient(pipe.x, 0, pipe.x + pipe.width, 0);
        pipeGradient.addColorStop(0, '#2d9e2d');
        pipeGradient.addColorStop(0.5, '#228B22');
        pipeGradient.addColorStop(1, '#1a6b1a');
        
        // Top pipe
        ctx.fillStyle = pipeGradient;
        ctx.fillRect(pipe.x, 0, pipe.width, pipe.topHeight);
        
        // Pipe highlights
        ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.fillRect(pipe.x, 0, pipe.width * 0.3, pipe.topHeight);
        
        // Pipe outline
        ctx.strokeStyle = '#1a6b1a';
        ctx.lineWidth = 3;
        ctx.strokeRect(pipe.x, 0, pipe.width, pipe.topHeight);
        
        // Pipe cap (top)
        const capGradient = ctx.createLinearGradient(pipe.x - 5, 0, pipe.x + pipe.width + 5, 0);
        capGradient.addColorStop(0, '#3db83d');
        capGradient.addColorStop(0.5, '#2d9e2d');
        capGradient.addColorStop(1, '#1a8b1a');
        ctx.fillStyle = capGradient;
        ctx.fillRect(pipe.x - 5, pipe.topHeight - 25, pipe.width + 10, 25);
        ctx.strokeRect(pipe.x - 5, pipe.topHeight - 25, pipe.width + 10, 25);

        // Bottom pipe
        ctx.fillStyle = pipeGradient;
        ctx.fillRect(pipe.x, pipe.bottomY, pipe.width, canvas.height - pipe.bottomY);
        
        // Pipe highlights
        ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.fillRect(pipe.x, pipe.bottomY, pipe.width * 0.3, canvas.height - pipe.bottomY);
        
        ctx.strokeStyle = '#1a6b1a';
        ctx.strokeRect(pipe.x, pipe.bottomY, pipe.width, canvas.height - pipe.bottomY);
        
        // Pipe cap (bottom)
        ctx.fillStyle = capGradient;
        ctx.fillRect(pipe.x - 5, pipe.bottomY, pipe.width + 10, 25);
        ctx.strokeRect(pipe.x - 5, pipe.bottomY, pipe.width + 10, 25);
    }
    
    // Draw particles
    gameState.particles.forEach(particle => particle.draw(ctx));

    // Draw ground with grass texture
    const groundY = canvas.height * 0.8;
    const groundGradient = ctx.createLinearGradient(0, groundY, 0, canvas.height);
    groundGradient.addColorStop(0, '#8B4513');
    groundGradient.addColorStop(1, '#654321');
    ctx.fillStyle = groundGradient;
    ctx.fillRect(0, groundY, canvas.width, canvas.height - groundY);
    
    // Grass on top of ground
    ctx.strokeStyle = '#228B22';
    ctx.lineWidth = 2;
    for (let x = 0; x < canvas.width; x += 10) {
        const offset = Math.sin(x * 0.1 + gameState.frameCount * 0.05) * 3;
        ctx.beginPath();
        ctx.moveTo(x, groundY);
        ctx.lineTo(x + 3, groundY - 8 + offset);
        ctx.stroke();
    }
    
    // Flash effect
    if (gameState.flashAlpha > 0) {
        ctx.fillStyle = `rgba(255, 255, 255, ${gameState.flashAlpha})`;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    
    ctx.restore();
}

// Game loop
function gameLoop() {
    update();
    draw();
    requestAnimationFrame(gameLoop);
}

// Event listeners
canvas.addEventListener('click', jump);
document.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
        e.preventDefault();
        jump();
    }
});

// API Functions
async function createSession() {
    try {
        const response = await fetch(`${API_URL}/api/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ player_name: 'Anonymous' })
        });
        const data = await response.json();
        if (data.success) {
            sessionId = data.session_id;
        }
    } catch (err) {
        console.error('Error creating session:', err);
    }
}

async function updateSession(updates) {
    if (!sessionId) return;
    
    try {
        await fetch(`${API_URL}/api/sessions/${sessionId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updates)
        });
    } catch (err) {
        console.error('Error updating session:', err);
    }
}

async function fetchLeaderboard() {
    try {
        const response = await fetch(`${API_URL}/api/leaderboard?limit=10`);
        const data = await response.json();
        displayLeaderboard(data.leaderboard);
        return data.leaderboard;
    } catch (err) {
        console.error('Error fetching leaderboard:', err);
        return [];
    }
}

function displayLeaderboard(leaderboard) {
    const list = document.getElementById('leaderboard-list');
    list.innerHTML = '';
    
    leaderboard.forEach((entry, index) => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span>#${index + 1} ${entry.player_name}</span>
            <span style="font-weight: bold">${entry.score}</span>
        `;
        list.appendChild(li);
    });
    
    if (leaderboard.length === 0) {
        list.innerHTML = '<li style="text-align: center; color: #999;">No scores yet</li>';
    }
}

async function fetchGlobalStats() {
    try {
        const response = await fetch(`${API_URL}/api/stats`);
        const data = await response.json();
        const stats = data.stats;
        
        document.getElementById('totalGames').textContent = stats.total_games || 0;
        document.getElementById('avgScore').textContent = stats.avg_score ? stats.avg_score.toFixed(1) : '0';
        document.getElementById('maxScore').textContent = stats.max_score || 0;
    } catch (err) {
        console.error('Error fetching stats:', err);
    }
}

async function fetchGameParams() {
    try {
        const response = await fetch(`${API_URL}/api/params`);
        const data = await response.json();
        Object.assign(gameParams, data.params);
        updateDebugParams();
        console.log('Fetched params:', gameParams);
    } catch (err) {
        console.error('Error fetching params:', err);
    }
}

function updateSessionStats() {
    document.getElementById('currentScore').textContent = gameState.score;
    document.getElementById('totalJumps').textContent = gameState.totalJumps;
    document.getElementById('maxVelocity').textContent = gameState.maxVelocity.toFixed(2);
}

function updateDebugParams() {
    document.getElementById('debugGravity').textContent = gameParams.gravity;
    document.getElementById('debugJumpStrength').textContent = gameParams.jumpStrength;
    document.getElementById('debugPipeSpeed').textContent = gameParams.pipeSpeed;
}

// WebSocket Functions
function connectWebSocket() {
    ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        updateWSStatus(true);
        clearInterval(reconnectInterval);
        reconnectInterval = null;
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (err) {
            console.error('Error parsing WebSocket message:', err);
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        updateWSStatus(false);
        
        // Attempt to reconnect
        if (!reconnectInterval) {
            reconnectInterval = setInterval(() => {
                console.log('Attempting to reconnect...');
                connectWebSocket();
            }, 5000);
        }
    };
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'connected':
            console.log('WebSocket handshake complete');
            break;
        case 'params-updated':
            console.log('Parameters updated via WebSocket:', data.params);
            Object.assign(gameParams, data.params);
            updateDebugParams();
            
            // Show visual feedback
            gameState.flashAlpha = 0.3;
            
            // If game not started, show updated message
            if (!gameState.gameStarted) {
                const msg = document.createElement('div');
                msg.textContent = `⚙️ Parameters Updated! Gravity: ${gameParams.gravity}`;
                msg.style.cssText = `
                    position: absolute;
                    top: 100px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: rgba(102, 126, 234, 0.95);
                    color: white;
                    padding: 15px 30px;
                    border-radius: 8px;
                    font-size: 18px;
                    font-weight: bold;
                    z-index: 1000;
                    animation: fadeOut 2s forwards;
                `;
                document.getElementById('gameSection').appendChild(msg);
                setTimeout(() => msg.remove(), 2000);
            }
            break;
        case 'game-update':
            // Handle real-time updates from other clients if needed
            break;
    }
}

function sendGameEvent(eventType, data = {}) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'game-event',
            sessionId: sessionId,
            eventType: eventType,
            data: data,
            timestamp: Date.now()
        }));
    }
}

function updateWSStatus(connected) {
    const statusDot = document.getElementById('wsStatus');
    const statusText = document.getElementById('wsStatusText');
    
    if (connected) {
        statusDot.classList.remove('disconnected');
        statusDot.classList.add('connected', 'pulse');
        statusText.textContent = 'Connected';
        statusText.style.color = '#28a745';
    } else {
        statusDot.classList.remove('connected', 'pulse');
        statusDot.classList.add('disconnected');
        statusText.textContent = 'Disconnected';
        statusText.style.color = '#dc3545';
    }
}

// Initialize everything
async function initializeApp() {
    await fetchGameParams();
    await fetchLeaderboard();
    await fetchGlobalStats();
    connectWebSocket();
    init();
    gameLoop();
    
    // Refresh leaderboard and stats periodically
    setInterval(fetchLeaderboard, 10000);
    setInterval(fetchGlobalStats, 15000);
}

// Start the app
initializeApp();

// Expose functions globally
window.restartGame = restartGame;
window.submitScore = submitScore;
window.getGameState = () => gameState;
window.getGameParams = () => gameParams;
