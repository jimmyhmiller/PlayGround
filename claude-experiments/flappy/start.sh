#!/bin/bash

echo "🐦 Starting Flappy Bird Premium Edition..."
echo ""

# Kill any existing server
pkill -f "node.*backend/server.js" 2>/dev/null

# Start the server
cd "$(dirname "$0")"
node backend/server.js &
SERVER_PID=$!

echo "✅ Server started (PID: $SERVER_PID)"
echo "🌐 Game running at: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Wait for Ctrl+C
trap "echo ''; echo 'Stopping server...'; kill $SERVER_PID 2>/dev/null; exit 0" INT

# Keep script running
wait $SERVER_PID
