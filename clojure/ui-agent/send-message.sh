#!/bin/bash

# UI Agent Message Sender
# Usage: ./send-message.sh "Your message here"
# Example: ./send-message.sh "Here's some sales data: Q1: 100, Q2: 150, Q3: 200"

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"Your message here\""
    echo "Example: $0 \"Here's some data to visualize: A=10, B=20, C=30\""
    exit 1
fi

MESSAGE="$1"
SERVER_URL="${UI_AGENT_URL:-http://localhost:8080}"

echo "Sending message to UI Agent at $SERVER_URL/message"
echo "Message: $MESSAGE"
echo "---"

# Check if httpie is installed
if ! command -v http &> /dev/null; then
    echo "Error: httpie (http command) is not installed"
    echo "Install it with: brew install httpie"
    exit 1
fi

# Send the message using httpie  
echo "{\"message\": \"$MESSAGE\", \"sender\": \"script\", \"timestamp\": $(date +%s)}" | \
http POST "$SERVER_URL/message" \
    Content-Type:application/json \
    --print=HhBb

echo ""
echo "Message sent! Check the UI window for any generated interface."