#!/bin/bash

echo "🤖 Setting up Claude CLI"
echo "========================"
echo ""

# Check if claude command exists
if command -v claude >/dev/null 2>&1; then
    echo "✅ Claude CLI is already installed!"
    echo "   Version: $(claude --version 2>/dev/null || echo 'Unknown')"
    echo ""
    echo "🚀 You're ready to run the app:"
    echo "   swift run"
else
    echo "❌ Claude CLI not found"
    echo ""
    echo "📦 To install Claude CLI, you have several options:"
    echo ""
    echo "1. Install via npm (if you have Node.js):"
    echo "   npm install -g @anthropics/claude-cli"
    echo ""
    echo "2. Download binary from:"
    echo "   https://github.com/anthropics/claude-cli/releases"
    echo ""
    echo "3. Or install via your package manager"
    echo ""
    echo "After installation, make sure 'claude' is in your PATH"
fi

echo ""
echo "📖 The app will use the 'claude' command with these flags:"
echo "   claude --format json --stream [your_message]"
echo ""
echo "💡 Make sure you're authenticated with Claude CLI before running the app"